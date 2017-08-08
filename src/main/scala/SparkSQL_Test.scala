/**
  * Created by jrlimingyang on 2017/7/25.
  */
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

object SparkSQL_Test {
  case class gd_py_corp_sharehd_info(id: String,
                                     batch_seq_num: String,
                                     name: String,
                                     contributiveFund: String,
                                     contributivePercent: String,
                                     currency: String,
                                     contributiveDate: String,
                                     corp_basic_info_id: String,
                                     query_time: String)

  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("Customers_Statistics").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //RDD 隐式转换成DataFrame
    import sqlContext.implicits._
    //读取本地文件
    val gd_py_corp_sharehd_infoDF = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\gd_py_corp_sharehd_info.txt")
      .map(_.split("\\t"))
      .map(d => gd_py_corp_sharehd_info(d(0),d(1),d(2),d(3),d(4),d(5),d(6),d(7),d(8)))
      .toDF()

    //注册表
    gd_py_corp_sharehd_infoDF.registerTempTable("gd_py_corp_sharehd_info")

    /**
      * 分渠道进件数量统计并按进件数量降序排列
      */
    val result= sqlContext.sql("select * from gd_py_corp_sharehd_info limit 10")
      .toDF().show

  }
}
