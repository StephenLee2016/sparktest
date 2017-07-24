/**
  * Created by jrlimingyang on 2017/7/21.
  */
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

object Ex0WordCount {

  def main(args: Array[String]): Unit ={
    val pathToFile = "file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\wordcount.txt"

    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.6.5")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val tweets = sc.textFile(pathToFile).flatMap(_.split(" ")).map(word => (word,1)).reduceByKey(_ + _)
    tweets.foreach(println)
  }


}
