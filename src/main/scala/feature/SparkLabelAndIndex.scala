package feature

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, IndexToString}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Created by jrlimingyang on 2017/8/9.
  */
object SparkLabelAndIndex {
  def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("TF-Idf").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._ //开启RDD的隐式转换

//    // StringIndexer
//    val df1 = sqlContext.createDataFrame(Seq(
//      (0, "a"),
//      (1, "b"),
//      (2, "c"),
//      (3, "a"),
//      (4, "b"),
//      (5, "c")
//    )).toDF("id", "category")
//
//    val indexer = new StringIndexer().
//      setInputCol("category").
//      setOutputCol("categoryIndex")
//
//    val model = indexer.fit(df1)
//    val indexed1 = model.transform(df1)

//    // 对于未出现过的类别的处理方式
//    val df2 = sqlContext.createDataFrame(Seq(
//      (0, "a"),
//      (1, "b"),
//      (2, "c"),
//      (3, "a"),
//      (4, "b"),
//      (5, "d")
//    )).toDF("id", "category")
//
//    val indexed2 = model.setHandleInvalid("skip").transform(df2)
//    indexed2.show


//    // Index转换成原来的类别
//    val converter = new IndexToString().
//      setInputCol("categoryIndex").
//      setOutputCol("originalCategory")
//
//    val converted = converter.transform(indexed1)
//    converted.show


//    // OneHotEncoding
//    val encoder = new OneHotEncoder().
//      setInputCol("categoryIndex").
//      setOutputCol("categoryVec")
//
//    val encoded = encoder.transform(indexed1)
//    encoded.show

    // VectorIndexer
    val data = Seq(
      Vectors.dense(-1.0, 1.0, 1.0),
      Vectors.dense(-1.0, 3.0, 1.0),
      Vectors.dense(0.0, 5.0, 1.0)
    )

    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val indexer = new VectorIndexer().
      setInputCol("features").
      setOutputCol("indexed").
      setMaxCategories(2)

    val indexerModel = indexer.fit(df)
    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet

    println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(","))

    val indexed = indexerModel.transform(df)
    indexed.show





  }
}
