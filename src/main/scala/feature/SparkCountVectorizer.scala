package feature

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

/**
  * Created by jrlimingyang on 2017/8/11.
  */
object SparkCountVectorizer {
  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("CountVectorizer").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val df = sqlContext.createDataFrame(Seq(
      (0, Array("我", "是", "中国", "人")),
      (1, Array("他", "是", "日本", "人", "啊")),
      (2, Array("中国", "游客", "在", "日本", "失踪"))
    )).toDF("id", "words")

    val cvModel : CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(5)
      .setMinDF(2)
      .fit(df)

    cvModel.transform(df).show

    // 指定一个先验词表
    val cvm = new CountVectorizerModel(Array("我", "是", "中国", "游客"))
      .setInputCol("words")
      .setOutputCol("features")

    cvm.transform(df).select("features").rdd.foreach{println}

  }
}
