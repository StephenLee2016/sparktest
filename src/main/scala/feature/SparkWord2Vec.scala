package feature

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Word2Vec

/**
  * Created by jrlimingyang on 2017/8/10.
  */


object SparkWord2Vec {
  def main(args: Array[String]): Unit={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("Word2Vec").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val documentDF = sqlContext.createDataFrame(Seq(
      "我 今天 吃 了 早饭".split(" "),
      "早饭 吃 了 面包 和 牛奶".split(" "),
      "今天 忽然 有点 激动 啊".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)

    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)

    result.select("result").take(3).foreach(println)

  }
}
