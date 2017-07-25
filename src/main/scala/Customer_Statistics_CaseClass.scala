/**
  * Created by jrlimingyang on 2017/7/25.
  */
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object Customer_Statistics_CaseClass {
  /**
    * 使用模板类描述元数据信息
    * @param chnl_code
    * @param id_num
    */
  case class blb_intpc_info(chnl_code: String, id_num: String)

  def main(args: Array[String]): Unit ={
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val conf = new SparkConf().setAppName("Customers_Statistics").setMaster("local[2]")
    conf.set("spark.testing.memory", "2147480000")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //RDD 隐士转换成DataFrame
    import sqlContext.implicits._
    //读取本地文件
    val blb_intpc_infoDF = sc.textFile("file:///C:\\Users\\jrlimingyang\\IdeaProjects\\sparktest\\src\\main\\scala\\data\\blb_intpc_info_10000_2.txt")
      .map(_.split("\\t"))
      .map(d => blb_intpc_info(d(0), d(1))).toDF()

    //注册表
    blb_intpc_infoDF.registerTempTable("blb_intpc_info")

    //分渠道进件数量统计并按进件数量降序排列
    sqlContext.sql("" + "select chnl_code, count(*) as intpc_sum from blb_intpc_info group by chnl_code")
      .toDF()
      .sort($"intpc_sum".desc)
      .show

  }
}
