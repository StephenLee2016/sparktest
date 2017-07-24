/**
  * Created by jrlimingyang on 2017/7/21.
  */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}

import scala.util.Random

object KMeanTest {
  def main(args: Array[String]): Unit ={
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.6.5")
    val conf = new SparkConf().setAppName("KMeans").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    val k = 5
    val dimension = 10
    val recordNum = 1000
    val sparsity = 1
    val iterations = 1000
    val means = 1
    val parNumber = 100

    val data: RDD[Vector] = sc.parallelize(1 to recordNum, parNumber).map(i => {
      val ran = new Random()
      val indexArr = ran.shuffle((0 until dimension).toList).take((dimension * sparsity).toInt).sorted.toArray
      val valueArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextDouble()).sorted.toArray
      val vec: Vector = new SparseVector(dimension, indexArr, valueArr)
      vec
    }).cache()
    println(args.mkString(", "))
    println(data.count() + " records generated")

    val st = System.nanoTime()

    val model = if(means == "my") {
      println("running scalable kmeans")
      val model = new KMeans()
        .setK(k)
        .setInitializationMode("random")
        .setMaxIterations(iterations)
        .run(data)
      model
    } else {
      println("running mllib kmeans")
      val model = new KMeans()
        .setK(k)
        .setInitializationMode("random")
        .setMaxIterations(iterations)
        .run(data)
      model
    }

    println((System.nanoTime() - st) / 1e9 + " seconds cost")
    println("final clusters: " + model.clusterCenters.length)
    println(model.clusterCenters.map(v => v.numNonzeros).mkString("\n"))

    sc.stop()

  }

}
