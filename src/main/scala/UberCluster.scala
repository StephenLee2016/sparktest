/**
  * Created by jrlimingyang on 2017/7/25.
  */
import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object UberCluster {

  case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setMaster("local[2]").setAppName("UberCluster")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    val spark: SparkSession = SparkSession.builder().appName("uber").getOrCreate()

    import spark.implicits._

    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)
    ))

    // Spark 2.1.0
    val df: Dataset[Uber] = spark.read.option("inferSchema", "false").schema(schema).csv("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\uber.csv").as[Uber]

    df.cache
    df.show
    //df.schema
    df.printSchema()

    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), 5043)

    val kmeans = new KMeans().setK(20).setFeaturesCol("features").setMaxIter(5)
    val model = kmeans.fit(trainingData)
    println("Final Centers: ")
    model.clusterCenters.foreach(println)

    val categories = model.transform(testData)

    categories.show
    categories.createOrReplaceTempView("uber")

    categories.select(month($"dt").alias("month"), dayofmonth($"dt").alias("day"), hour($"dt").alias("hour"), $"prediction").
                          groupBy("month", "day", "hour", "prediction").agg(count("prediction").alias("count")).orderBy("day","hour","prediction").show

    categories.select(hour($"dt").alias("hour"), $"prediction").groupBy("hour", "prediction").agg(count("prediction").alias("count")).
      orderBy(desc("count")).show

    categories.groupBy("prediction").count().show

    spark.sql("select prediction, count(prediction) as count from uber group by prediction").show
    spark.sql("select hour(uber.dt) as hr, count(prediction) as ct from uber group by hour(uber.dt)").show

    /**
      * uncomment below for various functionality:
      */
    //to save the model
    model.write.overwrite().save("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\model\\savemodel")
    // model can be re-loaded like this
    // val sameModel = KMeansModel.load("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\model\\savemodel")
    //
    // to save the categories dataframe as json data
    val res = spark.sql("select dt, lat, lon ,base, prediction as cid from uber order by dt")
    res.write.format("json").save("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\model\\uber.json")
  }
}