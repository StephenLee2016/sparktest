package algorithm

/**
  * Created by jrlimingyang on 2017/8/21.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object SparkCollaborativeFiltering {
  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("WordCount").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)

    val data = sc.textFile("file:///D:\\spark-2.1.0-bin-hadoop2.7\\data\\mllib\\als\\test.data")
    val ratings = data.map(_.split(",") match{
      case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)
    })

    val splits = ratings.randomSplit(Array(0.8, 0.2))
    val training = splits(0)
    val test = splits(1)

    /*
    用ALS训练数据建立推荐模型
    - numBlocks 用于并行化计算的分块个数（设置为-1， 为自动配置）
    - rank 模型中隐语义因子的个数
    - iterations 是迭代的次数
    - lambda 是ALS的正则化参数
    - implicitPrefs 决定了是用显性反馈ALS的版本还是使用隐性反馈数据集的版本
    - alpha 针对隐性反馈ALS版本的参数，这个参数决定了偏好行为强度的基准
     */

    val rank = 10
    val numIterations = 10
    //val model = ALS.train(training, rank, numIterations, 0.01)
    val model = new ALS()
      .setRank(rank)
      .setIterations(numIterations)
      .setLambda(0.01)
      .setUserBlocks(-1)
      .run(training)

    val testUsersProducts = test.map {case Rating(user, product, rate) => (user, product)}

    val predictions = model.predict(testUsersProducts).map{case Rating(user, product, rate) => ((user,product), rate)}
    val ratesAndPreds = test.map{case Rating(user, product, rate) => ((user, product), rate)}.join(predictions)

    //ratesAndPreds.foreach(println)
    val MSE = ratesAndPreds.map{case ((user, product), (r1, r2)) => val err = (r1-r2)
      err * err}.mean()
    println("Mean Squared Error = " + MSE)
  }
}
