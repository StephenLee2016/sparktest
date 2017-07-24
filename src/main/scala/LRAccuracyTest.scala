/**
  * Created by jrlimingyang on 2017/7/21.
  */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf,SparkContext}

object LRAccuracyTest {
  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.6.5")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)
    val data = MLUtils.loadLibSVMFile(sc, "file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\sample_libsvm_data.txt").map(
      l => LabeledPoint(l.label, l.features.toSparse)
    )

    //split data into training (60%) and test(40%)
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    //run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS().setNumClasses(5).run(training)

    //compute raw scores on the test set
    val predictionAndLabels = test.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)}

    //get evaluation metrics
    val metrics = new MulticlassMetrics(predictionAndLabels)

    val precision = metrics.precision
    println("Precision = " + precision)

  }
}
