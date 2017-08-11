package algorithm

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

/**
  * Created by jrlimingyang on 2017/8/11.
  */
object SparkDecisionTree {
  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)

  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("DecisionTree").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    // 载入数据
    val data = sqlContext.sparkContext.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\iris.txt")
      .map(_.split(","))
      .map(p => Iris(Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble), p(4).toString()))
      .toDF()

    data.createOrReplaceTempView("iris")
    val df = sqlContext.sql("select * from iris")

    //df.map(t => t(1) + ":" + t(0)).collect().foreach(println)

    // 分别获取标签列和特征列，进行索引，并进行重命名
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(df)

    // 这里我们设置一个labelConverter, 目的是把预测的类别重新转成字符型的
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // 接下来，将数据集随机分成训练集和测试集，其中训练集占70%
    val Array(trainingData, testData) = data.randomSplit(Array(0.7,0.3))


    // 训练决策树模型，这里可以通过setter的方法来设置决策树的参数，也可以通过
    // ParamMap来设置。具体的可以设置的参数可以通过explainParams()来获取
    val dtClassifier = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // 在Pipeline中进行设置
    val pipelinedClassifier = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dtClassifier, labelConverter))

    // 训练决策树模型
    val modelClassifier = pipelinedClassifier.fit(trainingData)

    // 进行预测
    val predictionsClassifier = modelClassifier.transform(testData)

    // 查看部分预测结果
    predictionsClassifier.select("predictedLabel", "label", "features").show

    //
    // 评估决策树分类模型
    //
    val evaluatorClassifier = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluatorClassifier.evaluate(predictionsClassifier)
    println("Test Error = " + (1.0 - accuracy))

    val treeModelClassifier = modelClassifier.stages(2)
      .asInstanceOf[DecisionTreeClassificationModel]

    println("Learned classification tree model: \n" + treeModelClassifier.toDebugString)

    // --------------------------
    // 训练决策树回归模型
    // --------------------------
    val dtRegressor = new DecisionTreeRegressor()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // 在pipeline中进行设置
    val pipelineRegressor = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dtRegressor, labelConverter))

    // 训练决策树模型
    val modelRegressor = pipelineRegressor.fit(trainingData)
    // 进行预测
    val predictionRegressor = modelRegressor.transform(testData)

    // 查看部分预测结果
    predictionRegressor.select("predictedLabel", "label", "features").show

    //
    // 评估决策树回归模型
    //
    val evaluationRegressor = new RegressionEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    var rmse = evaluationRegressor.evaluate(predictionRegressor)
    println("Root Mean Squred Error (RMSE) on test data = " + rmse)

    val treeModelRegressor = modelRegressor.stages(2).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModelRegressor.toDebugString)
  }

}
