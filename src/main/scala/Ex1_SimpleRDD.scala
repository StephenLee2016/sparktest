/**
  * Created by jrlimingyang on 2017/7/21.
  */
import org.apache.spark.{SparkConf, SparkContext}

object Ex1_SimpleRDD {
  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.6.5")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    //put some data into an RDD
    val numbers = 1 to 10
    val numbersRDD = sc.parallelize(numbers, 4)
    println("Print each element of the original RDD")
    numbersRDD.foreach(println)

    //trivially operate on the numbers
    val stillAnRDD = numbersRDD.map(n => n.toDouble / 10)

    //get the data back out
    val nowAnArray = stillAnRDD.collect()
    println("Now print each element of the transformed array")
    nowAnArray.foreach(println)

    //explore RDD properties
    val partitions = stillAnRDD.glom()
    println("We _should_ have 4 partitions")
    println(partitions.count())
    partitions.foreach(a => {
      println("Partition contents:"+ a.foldLeft("")(((s,e) => s + " " + e)))
    })
  }
}
