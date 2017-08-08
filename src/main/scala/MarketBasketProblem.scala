import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{FloatType, _}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object MarketBasketProblem {

  /*Init Spark Context*/
  def initSparkContext(): Tuple2[SparkContext, SQLContext] = {
    /*  Initialize Spark Context*/
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("ShoppingListFrequentItems").setMaster("local[2]")
    conf.set("spark.testing.memory", "50147480000")

    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    return (sc, sqlContext)
  }

  def main(args: Array[String]) {

    /* input file */
    val input_file = "file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\OnlineRetailData.csv"

    /*Init Spark Context*/
    val (sc, sqlContext) = initSparkContext()

    val SqlDataFrame = new PurchaseDataFrame(sqlContext, input_file)


    /* Extract Data*/
    val purchases = SqlDataFrame.readPurchases()
    val purchasesRDD = purchases.rdd.map(v => (v.getString(0), v.getString(1)))
      .groupByKey().map(t => t._2.toArray.distinct)


    /* Initialize FPGrowth */
    val fpg = new FPGrowth()
      .setMinSupport(0.01)
      .setNumPartitions(100)

    /* Learn FPGrowth */
    val model = fpg.run(purchasesRDD)

    /* Get Association Rules from Model */
    val minConfidence = 0.8
    val rules = model.generateAssociationRules(minConfidence).zipWithUniqueId()

    /* Make a result in readable format*/
    val antecedentRDD = rules.flatMap(r => r._1.antecedent.map(t => Row(r._2, t)))
    val consequentRDD = rules.flatMap(r => r._1.consequent.map(t => Row(r._2, t)))

    val product = SqlDataFrame.readPurchasesWithDescription()
    var contAnt = SqlDataFrame.readAntecSchema(antecedentRDD)
    var contCons = SqlDataFrame.readAntecSchema(consequentRDD)

    var joinResultAnt = contAnt.join(product, contAnt.col("AntecStockCode") === product.col("StockCode"), "left").select("RuleID", "StockCode", "Description").distinct()
    var joinResultCons = contCons.join(product, contCons.col("AntecStockCode") === product.col("StockCode"), "left").select("RuleID", "StockCode", "Description").distinct()

    joinResultAnt.show()
    joinResultCons.show()

    var totalResult = joinResultAnt.rdd.map(v => (v(0), v(2))).groupByKey().map(t => (t._1, t._2.toArray.mkString(", ")))
    var totalResultCons = joinResultCons.rdd.map(v => (v(0), v(2))).groupByKey().map(t => (t._1, t._2.toArray.mkString(", ")))

    /* Save Results*/
    var result = totalResult.cogroup(totalResultCons).map(t => "Rule:" + t._1 + " [" + t._2._1.toArray.mkString(", ") + "] = > " + t._2._2.toArray.mkString(", "))
    result.coalesce(1).saveAsTextFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\result.txt")
    result.foreach(println)
  }
}


