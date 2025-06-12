from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_date, current_timestamp, when, isnan, regexp_replace, trim, udf
from pyspark.sql.types import *
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import matplotlib.dates as mdates

# 创建 SparkSession
spark = (
    SparkSession.builder
    .appName("KafkaProductDataCleaningAndMining")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

# 调整日志级别
spark.sparkContext.setLogLevel("ERROR")

# 定义数据 Schema
product_schema = StructType([
    StructField("商品ID", StringType(), True),
    StructField("商品名称", StringType(), True),
    StructField("商品类别", StringType(), True),
    StructField("价格", StringType(), True),
    StructField("库存", StringType(), True),
    StructField("销量", StringType(), True),
    StructField("评分", StringType(), True),
    StructField("评论数", StringType(), True),
    StructField("上架周期", StringType(), True),
    StructField("上架日期", StringType(), True),
])

# 从 Kafka 读取流数据
kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "192.168.152.128:9092")
    .option("subscribe", "your_topic_name")
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
)

# 解析 JSON 数据
parsed_df = kafka_df.select(from_json(col("value").cast("string"), product_schema).alias("data")).select("data.*")

# 增强型数据清洗函数
def clean_data(df):
    # 字符串清理
    cleaned_strings_df = df.select(
        col("商品ID").alias("product_id"),
        trim(regexp_replace(col("商品名称"), r'[^\w\u4e00-\u9fa5\s]', '')).alias("product_name"),
        trim(col("商品类别")).alias("category"),
        trim(col("价格")).alias("价格"),
        trim(col("库存")).alias("库存"),
        trim(col("销量")).alias("销量"),
        trim(col("评分")).alias("评分"),
        trim(col("评论数")).alias("评论数"),
        trim(col("上架周期")).alias("上架周期"),
        trim(col("上架日期")).alias("上架日期")
    )

    # 数据类型转换
    type_converted_df = cleaned_strings_df.select(
        col("product_id"),
        col("product_name"),
        col("category"),
        when(col("价格").rlike(r'^\d+(\.\d+)?$'), col("价格").cast(DoubleType())).otherwise(None).alias("价格"),
        when(col("库存").rlike(r'^\d+$'), col("库存").cast(IntegerType())).otherwise(None).alias("库存"),
        when(col("销量").rlike(r'^\d+$'), col("销量").cast(IntegerType())).otherwise(None).alias("销量"),
        when(col("评论数").rlike(r'^\d+$'), col("评论数").cast(IntegerType())).otherwise(None).alias("评论数"),
        when(
            col("评分").rlike(r'^\d+(\.\d+)?$') &
            (col("评分").cast(DoubleType()) >= 0) &
            (col("评分").cast(DoubleType()) <= 5),
            col("评分").cast(DoubleType())
        ).otherwise(None).alias("评分"),
        when(col("上架周期").rlike(r'^\d+$'), col("上架周期").cast(IntegerType())).otherwise(None).alias("上架周期"),
        col("上架日期")
    )

    # 类别标准化
    category_mapping = {
        "数码配件": "数码配件",
        "数码产品配件": "数码配件",
        "存储设备": "存储设备",
        "存储器": "存储设备",
        "电脑办公": "电脑办公",
        "计算机办公": "电脑办公"
    }

    @udf(StringType())
    def standardize_category(cat):
        if cat in category_mapping:
            return category_mapping[cat]
        return cat

    category_standardized_df = type_converted_df.withColumn("category", standardize_category(col("category")))

    # 处理缺失值
    filled_df = category_standardized_df.fillna({
        "价格": 0.0,
        "库存": 0,
        "销量": 0,
        "评分": 0.0,
        "评论数": 0,
        "上架周期": 0
    })

    return filled_df

# 应用数据清洗
cleaned_df = (
    parsed_df
    .transform(clean_data)
    .withColumn("shelf_date", to_date("上架日期", "yyyy/M/d"))
    .withColumn("processing_time", current_timestamp())
    .where(col("product_id").isNotNull())
    .filter(col("价格") > 0)
    .filter(col("category").isin(["数码配件", "存储设备", "电脑办公"]))
)