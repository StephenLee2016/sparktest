package feature

/**
  * Created by jrlimingyang on 2017/8/9.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{SparkSession, SQLContext}

object Spark_TFIdf {
  def main(args: Array[String]): Unit ={

    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("TF-Idf").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._   //开启RDD的隐式转换

    val sentenceData = sqlContext.createDataFrame(Seq(
      (0, "我 今天 吃 猪肉 了"),
      (0, "我 去年 买 表 了"),
      (1, "王者荣耀 全民 都 在 玩")
    )).toDF("label", "sentence")

    //用tokenizer对句子进行分词
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF().setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(2000)  //设置哈希表的桶数为2000

    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.select("rawFeatures").show

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(2).foreach(println)

  }
}
