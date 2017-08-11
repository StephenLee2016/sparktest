package algorithm

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, IndexToString, StringIndexer, Tokenizer, VectorIndexer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.sql.functions

/**
  * Created by jrlimingyang on 2017/8/11.
  */
object SparkLogisticRegression {
  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)

  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val data = sqlContext.sparkContext.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\iris.txt")
      .map(_.split(","))
      .map(p => Iris(Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble), p(4).toString()))
      .toDF()

    data.createOrReplaceTempView("iris")
    //val df = sqlContext.sql("select * from iris where label != 'setosa'")
    val df = sqlContext.sql("select * from iris")
    df.map(t => t(1) + ":" + t(0)).collect().foreach(println)

    //
    // 构建ML的Pipeline
    //
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(df)

    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val lrPipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))

    val lrPipelineModel = lrPipeline.fit(trainingData)
    val lrPredictions = lrPipelineModel.transform(testData)

    lrPredictions.select("predictedLabel", "label", "features", "probability")
      .collect()
      .foreach {case Row(predictedLabel: String, label: String, features: Vector, prob: Vector) =>
       println(s"($label, $features) --> prob=$prob, predicted Label=$predictedLabel")}

    //
    // 模型评估
    //
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
//
//    val lrAccuracy = evaluator.evaluate(lrPredictions)
//    println("Test Error = " + (1.0 - lrAccuracy))
//
//    val lrModel = lrPipelineModel
//      .stages(2)
//      .asInstanceOf[LogisticRegressionModel]
//
//    println("Coefficients: " + lrModel.coefficients + " Intercept:" + lrModel.intercept +
//      " numClasses: " + lrModel.numClasses + " numFeatures: " + lrModel.numFeatures)
//
//    //
//    // 模型评估
//    //
//    val trainingSummary = lrModel.summary
//    val objectiveHistory = trainingSummary.objectiveHistory
//    objectiveHistory.foreach(loss => println(loss))
//
//    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
//    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")
//
//    val fMeasure = binarySummary.fMeasureByThreshold
//    val maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0)
//
//    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
//      .select("threshold")
//      .head()
//      .getDouble(0)
//
//    lrModel.setThreshold(bestThreshold)


    //
    // 用多项逻辑回归解决 二分类(多) 问题
    //
    val mlr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")

    val mlrPipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, mlr, labelConverter))
    val mlrPipelineModel = mlrPipeline.fit(trainingData)
    val mlrPredictions = mlrPipelineModel.transform(testData)

    mlrPredictions.select("predictedLabel", "label", "features", "probability")
      .collect()
      .foreach { case Row(predictedLabel: String, label: String, features: Vector, prob: Vector) =>
      println(s"($label, $features) --> prob=$prob, predictedLabel=$predictedLabel")}


    val mlrAccuracy = evaluator.evaluate(mlrPredictions)
    println("Test Error = " + (1.0 - mlrAccuracy))

    val mlrModel = mlrPipelineModel.stages(2)
      .asInstanceOf[LogisticRegressionModel]

    println("Multinomial coefficients: " + mlrModel.coefficientMatrix +
    " Multinomial intercepts; " + mlrModel.interceptVector + " numClasses: " +
    mlrModel.numClasses + " numFeatures: " + mlrModel.numFeatures)




  }
}
