/**
  * Created by jrlimingyang on 2017/7/25.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD

import java.text.SimpleDateFormat
import java.lang.String
import java.util.concurrent.TimeUnit
import java.util.Locale


object SparkSQLCSV {
  case class Data(fullName: String, sex: String, age: Integer, dob: Long)
  case class NewData(fullName: String, occupation: String, address: String)

  private def readDataCsv(sc: SparkContext): RDD[Data] = {
    val data = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\data.csv")

    val header = data.first()
    val cleanData = data.filter(line => !line.equals(header))

    val parsedData = cleanData.map(line => {
      val fields = line.split(",")
      val age = fields(2).trim.toInt
      val dformat = new SimpleDateFormat("dd/mm/yyyy", Locale.ENGLISH)
      val date = dformat.parse(fields(3).trim)

      Data(fields(0).trim, fields(1).trim, age , date.getTime)
    })
    parsedData
  }

  private def readNewDataCsv(sc: SparkContext): RDD[NewData] = {
    val data = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\newdata.csv")

    val header = data.first()
    val cleanData = data.filter(line => !line.equals(header))

    val parsedData = cleanData.map(line => {
      val fields = line.split(",")
      val fullName = fields(0).trim
      val occupation = fields(1).trim
      val address = fields(2).trim

      NewData(fullName, occupation, address)
    })
    parsedData
  }



  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlCt = new SQLContext(sc)

    val parsedData = readDataCsv(sc)
    val newParsedData = readNewDataCsv(sc)

    // create dataframes and register the two tables
    val df = sqlCt.createDataFrame(parsedData)
    df.registerTempTable("data")

    val dfNew = sqlCt.createDataFrame(newParsedData)
    dfNew.registerTempTable("newdata")

    println("The data read from the file: ")
    parsedData.foreach(println)
    println

    println("Number of males: " + df.filter("sex='male'").count()+"\n")

    val row = sqlCt.sql("select avg(age) from data").collect()(0)
    println("Average age: " + row.getDouble(0) + "\n" )

    println("Select name and age: ")
    val nameAndAge = sqlCt.sql("select fullName, age from data")
    nameAndAge.show
    println

    val jeff = df.filter("fullName='Jeff Briton'").collect()(0)
    val tom = df.filter("fullName='Tom Soyer'").collect()(0)

    val ms = Math.abs(jeff.getLong(3) - tom.getLong(3))
    val days = TimeUnit.MILLISECONDS.toDays(ms)
    println("Year difference in days between Jeff and Tom: "+ days + "\n")

    val maxAge = sqlCt.sql("select MAX(age) from data")
    println("Maximum age: " + maxAge.collect()(0).getInt(0) + "\n")

    println("Data ordered by age: ")
    df.orderBy(desc("age")).show
    println

    println("Join on data and newdata tables where the sex is male")
    val joinDataOrderedByAge = sqlCt.sql(
      """
         select a.fullName,
                a.age,
                b.occupation
         from data a
         join newdata b
         on a.fullName = b.fullName
         where a.sex = 'male'
      """)

    joinDataOrderedByAge.show

  }

}
