import os
import time
import shutil
import subprocess
import warnings
import matplotlib.pyplot as plt
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, avg, count, window, current_timestamp, to_timestamp, lit, concat, \
    coalesce, format_number
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

# 设置HDFS用户
os.environ['HADOOP_USER_NAME'] = 'hadoop'

# 设置日志配置，解决SLF4J绑定警告
os.environ['SPARK_DIST_CLASSPATH'] = subprocess.check_output("hadoop classpath", shell=True).decode("utf-8")

# 完全忽略所有UserWarning警告
warnings.filterwarnings("ignore", category=UserWarning)

# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False

# 配置matplotlib日志，抑制字体查找警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 尝试设置中文字体，使用更简洁的方式
try:
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
except:
    pass

# 定义数据结构
schema = StructType([
    StructField("商品ID", StringType(), True),
    StructField("商品名称", StringType(), True),
    StructField("商品类别", StringType(), True),
    StructField("价格", DoubleType(), True),
    StructField("销量", IntegerType(), True),
    StructField("上架日期", StringType(), True),
    StructField("库存", IntegerType(), True),
    StructField("评分", DoubleType(), True),
    StructField("评论数", IntegerType(), True),
    StructField("上架周期", IntegerType(), True)
])


def visualize_alerts(df, batch_id):
    print(f"开始可视化批次 {batch_id} 的预警数据...")
    pandas_df = df.toPandas()
    alert_counts = pandas_df.groupby('alert_type').size()

    plt.figure(figsize=(12, 8))
    ax = alert_counts.plot(kind='bar', color='skyblue')

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.title(f'批次 {batch_id} 预警类型统计', fontsize=14)
    plt.xlabel('预警类型', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()

    local_path = f"/tmp/alert_visualization_batch_{batch_id}.png"
    svg_path = f"/tmp/alert_visualization_batch_{batch_id}.svg"
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    plt.close()

    try:
        hdfs_base_path = "hdfs://master:9000/shuju/real_time_alert/hot_products"
        hdfs_path = f"{hdfs_base_path}/visualizations/alert_batch_{batch_id}.png"
        hdfs_svg_path = f"{hdfs_base_path}/visualizations/alert_batch_{batch_id}.svg"

        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", f"{hdfs_base_path}/visualizations"], check=True)
        subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_path], check=True)
        subprocess.run(["hdfs", "dfs", "-put", "-f", svg_path, hdfs_svg_path], check=True)

        print(f"可视化图表已保存至: {hdfs_path}")
    except Exception as e:
        print(f"保存到HDFS失败: {e}")


def main():
    print("正在创建SparkSession...")
    spark = SparkSession.builder \
        .appName("实时预警系统") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,org.slf4j:slf4j-log4j12:1.7.36") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark_checkpoint") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.streaming.metricsEnabled", "true") \
        .getOrCreate()

    # 设置日志级别为INFO，便于调试
    spark.sparkContext.setLogLevel("INFO")
    print("SparkSession创建成功")

    hdfs_base_path = "hdfs://master:9000/shuju/real_time_alert/hot_products"
    checkpoint_path = f"{hdfs_base_path}/checkpoints"

    print(f"清理检查点目录: {checkpoint_path}")
    try:
        subprocess.run(["hdfs", "dfs", "-rm", "-r", checkpoint_path], check=False)
        print(f"检查点目录已清理")
    except Exception as e:
        print(f"检查点目录清理错误: {e}")

    # 从Kafka主题读取数据
    kafka_bootstrap_servers = "slave1:9092"
    kafka_topic = "test_topic"
    print(f"尝试从Kafka主题读取数据: {kafka_topic}")

    # 添加Kafka连接超时设置
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .option("kafka.request.timeout.ms", "60000") \
        .option("kafka.session.timeout.ms", "30000") \
        .load()

    print("开始解析Kafka消息...")
    parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")) \
        .select("data.*") \
        .na.fill(0, ["价格", "库存", "销量", "评论数", "上架周期", "评分"]) \
        .withColumn("商品类别", coalesce(col("商品类别"), lit("未知类别"))) \
        .withColumn("timestamp", current_timestamp())

    print("执行窗口聚合操作...")
    windowed_data = parsed_df \
        .withWatermark("timestamp", "5 minutes") \
        .groupBy(
        window(col("timestamp"), "10 minutes", "5 minutes"),
        col("商品ID"),
        col("商品名称"),
        col("商品类别")
    ) \
        .agg(
        avg("销量").alias("avg_sales"),
        count("*").alias("transaction_count"),
        avg("库存").alias("avg_inventory"),
        avg("评分").alias("avg_rating"),
        avg("评论数").alias("avg_comments"),
        avg("价格").alias("avg_price")
    )

    print("生成预警数据...")
    hot_products = windowed_data \
        .filter("avg_sales > 20 AND transaction_count > 2") \
        .withColumn("alert_type", concat(col("商品类别"), lit(" 爆款商品"))) \
        .select(
        col("商品ID"), col("商品名称"), col("商品类别"), col("alert_type"),
        col("window.end").alias("时间戳"),
        col("window.start").alias("窗口开始时间"),
        col("window.end").alias("窗口结束时间"),
        lit(0.0).alias("额外指标"),
        format_number(col("avg_sales"), 2).alias("平均销量"),
        col("transaction_count").alias("交易次数"),
        format_number(col("avg_price"), 2).alias("平均价格")
    )

    inventory_alert = windowed_data \
        .filter("avg_inventory < 20") \
        .withColumn("alert_type", concat(col("商品类别"), lit(" 库存预警"))) \
        .select(
        col("商品ID"), col("商品名称"), col("商品类别"), col("alert_type"),
        col("window.end").alias("时间戳"),
        col("window.start").alias("窗口开始时间"),
        col("window.end").alias("窗口结束时间"),
        format_number(col("avg_inventory"), 2).alias("额外指标"),
        format_number(col("avg_inventory"), 2).alias("平均库存"),
        col("transaction_count").alias("交易次数"),
        format_number(col("avg_price"), 2).alias("平均价格")
    )

    high_rating_alert = windowed_data \
        .filter("avg_rating > 4.5 AND avg_sales > 10") \
        .withColumn("alert_type", concat(col("商品类别"), lit(" 高评分爆款商品"))) \
        .select(
        col("商品ID"), col("商品名称"), col("商品类别"), col("alert_type"),
        col("window.end").alias("时间戳"),
        col("window.start").alias("窗口开始时间"),
        col("window.end").alias("窗口结束时间"),
        format_number(col("avg_rating"), 2).alias("额外指标"),
        format_number(col("avg_rating"), 2).alias("平均评分"),
        col("transaction_count").alias("交易次数"),
        format_number(col("avg_sales"), 2).alias("平均销量")
    )

    comment_alert = windowed_data \
        .filter("avg_comments > avg_sales") \
        .withColumn("alert_type", concat(col("商品类别"), lit(" 评论数异常（评论数 > 销量）"))) \
        .select(
        col("商品ID"), col("商品名称"), col("商品类别"), col("alert_type"),
        col("window.end").alias("时间戳"),
        col("window.start").alias("窗口开始时间"),
        col("window.end").alias("窗口结束时间"),
        format_number(col("avg_comments"), 2).alias("额外指标"),
        format_number(col("avg_comments"), 2).alias("平均评论数"),
        col("transaction_count").alias("交易次数"),
        format_number(col("avg_sales"), 2).alias("平均销量")
    )

    all_alerts = hot_products.union(inventory_alert).union(high_rating_alert).union(comment_alert)

    def print_alert(df, batch_id):
        print(f"\n===== 批次 {batch_id} 预警结果 =====")
        if df.rdd.isEmpty():
            print("此批次没有预警数据")
            return

        print(f"批次 {batch_id} 有 {df.count()} 条预警数据")
        df.show(truncate=False, n=20)

        hdfs_base_path = "hdfs://master:9000/shuju/real_time_alert/hot_products"
        output_path = f"{hdfs_base_path}/alerts/batch_{batch_id}"

        try:
            df.write.mode("overwrite").csv(
                output_path,
                header=True,
                sep="\t",
                quoteAll=True
            )
            print(f"预警数据已保存至: {output_path}")
        except Exception as e:
            print(f"保存预警数据失败: {e}")

        visualize_alerts(df, batch_id)

    print("启动流处理...")
    query = all_alerts.writeStream \
        .outputMode("update") \
        .foreachBatch(print_alert) \
        .trigger(processingTime="15 seconds") \
        .option("checkpointLocation", f"{checkpoint_path}/console") \
        .option("failOnDataLoss", "false") \
        .start()

    print("流处理已启动，等待数据...")
    print(f"查询ID: {query.id}")
    print(f"查询名称: {query.name}")
    print(f"查询状态: {query.status}")

    try:
        print("等待流处理终止...")
        query.awaitTermination()
    except KeyboardInterrupt:
        print("用户中断流处理")
        query.stop()
    finally:
        print("关闭SparkSession...")
        spark.stop()
        print("作业完成")


if __name__ == "__main__":
    main()
