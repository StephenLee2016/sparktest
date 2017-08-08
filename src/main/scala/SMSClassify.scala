/**
  * Created by jrlimingyang on 2017/8/4.
  */
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

object SMSClassify {
  final val VECTOR_SIZE = 100
  def main(args: Array[String]): Unit ={

    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("SMSClassify").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlCtx = new SQLContext(sc)
    val parsedRDD = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\data.txt")
      .map(_.split("\t")).map(eachRow => {(eachRow(0), eachRow(1).split(" "))})

    val msgDF = sqlCtx.createDataFrame(parsedRDD).toDF("label", "message")
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(msgDF)

    val word2Vec =new Word2Vec()
      .setInputCol("message")
      .setOutputCol("features")
      .setVectorSize(VECTOR_SIZE)
      .setMinCount(1)

    val layers = Array[Int](VECTOR_SIZE, 6, 4, 2)
    val mlpc = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(512)
      .setSeed(1234L)
      .setMaxIter(128)
      .setFeaturesCol("features")
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val Array(trainingData, testData) = msgDF.randomSplit(Array(0.8, 0.2))

    val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, mlpc, labelConverter))
    val model = pipeline.fit(trainingData)

    val predictionResultDF = model.transform(testData)
    //below 2 lines are for debug use
    predictionResultDF.printSchema
    predictionResultDF.select("message", "label", "predictedLabel").show(30)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
    println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")
    sc.stop

  }

}
