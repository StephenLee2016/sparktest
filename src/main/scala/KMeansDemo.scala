/**
  * Created by jrlimingyang on 2017/7/24.
  */
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object KMeansDemo {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.6.5")
    val conf = new SparkConf().setAppName("KMeansDemo").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    val log = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\train.txt")
    val model = build_model(log, 10, 100)
    val predictedCluster = model.predict(featurize("My mother is very great"))
    println("-"*30)
    println(predictedCluster.toString())
    println("-"*30)
  }

  def build_model(text: RDD[String], numClusters: Int, numIterations: Int): KMeansModel = {
    //Caches the vectors since it will be used many times by KMeans
    val vectors = text.map(featurize).cache()
    vectors.count() //calls an action to create the cache
    KMeans.train(vectors, numClusters, numIterations)
  }

  def featurize(s: String): Vector = {
    val tf = new HashingTF(1000)
    val bigram = s.sliding(2).toSeq
    tf.transform(bigram)
  }

}
