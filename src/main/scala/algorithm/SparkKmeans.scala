package algorithm

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}


/**
  * Created by jrlimingyang on 2017/8/11.
  */
object SparkKmeans {
  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\iris.txt")

    // 将RDD[String]转化为RDD[Vector]
    val trainingData = rawData.map(line => {Vectors.dense(line.split(",").filter(
      p => p.matches("\\d*(\\.?)\\d*")).map(_.toDouble))}).cache()

    val model: KMeansModel = KMeans.train(trainingData, 3, 100, 5)

    model.clusterCenters.foreach(center => {
      println("Clustering Center: " + center)
    })

    trainingData.collect().foreach(
      sample => {
        val predictedCluster = model.predict(sample)
        println(sample.toString + " belongs to cluster " + predictedCluster)
      }
    )

  }

}
