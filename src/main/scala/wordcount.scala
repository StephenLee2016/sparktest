/**
  * Created by jrlimingyang on 2017/7/21.
  */
import org.apache.spark.{SparkContext, SparkConf}

object wordcount {
  def main(args: Array[String]): Unit={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val text = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\word.txt")
  }
}
