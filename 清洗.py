# 导入必要的库
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns  # 基于matplotlib的统计数据可视化库
import os  # 提供与操作系统进行交互的功能
import pandas as pd  # 用于数据处理和分析
from pyspark.sql import SparkSession  # 用于创建SparkSession对象，与Spark集群进行交互
from pyspark.sql.functions import (
    col, from_json, to_date, current_timestamp, when, isnan, regexp_replace, trim, udf
)  # 导入Spark SQL的函数
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType  # 导入Spark SQL的数据类型
from pyspark.ml.feature import VectorAssembler  # 用于将多个列合并为一个向量列
import matplotlib.dates as mdates  # 用于日期格式化

# 创建SparkSession对象，它是与Spark集群交互的入口点
spark = (
    SparkSession.builder
    .appName("KafkaProductDataCleaningAndMining")  # 设置应用程序的名称，方便在Spark UI中识别
    .config("spark.sql.shuffle.partitions", "4")  # 配置数据洗牌时的分区数，可根据集群资源调整
    .getOrCreate()
)
# 设置日志级别为ERROR，只显示错误信息，减少不必要的日志输出
spark.sparkContext.setLogLevel("ERROR")

# 定义数据的Schema，描述从Kafka读取的数据结构
product_schema = StructType([
    # 商品ID，字符串类型，允许为空
    StructField("商品ID", StringType(), True),
    # 商品名称，字符串类型，允许为空
    StructField("商品名称", StringType(), True),
    # 商品类别，字符串类型，允许为空
    StructField("商品类别", StringType(), True),
    # 商品价格，字符串类型，允许为空
    StructField("价格", StringType(), True),
    # 商品库存，字符串类型，允许为空
    StructField("库存", StringType(), True),
    # 商品销量，字符串类型，允许为空
    StructField("销量", StringType(), True),
    # 商品评分，字符串类型，允许为空
    StructField("评分", StringType(), True),
    # 商品评论数，字符串类型，允许为空
    StructField("评论数", StringType(), True),
    # 商品上架周期，字符串类型，允许为空
    StructField("上架周期", StringType(), True),
    # 商品上架日期，字符串类型，允许为空
    StructField("上架日期", StringType(), True),
])

# 从Kafka读取流数据并进行解析
KAFKA_TOPIC = "your_topic_name"  # 替换为实际的Kafka主题名称
kafka_df = (
    spark.readStream
    .format("kafka")  # 指定数据源格式为Kafka
    .option("kafka.bootstrap.servers", "192.168.111.129:9092")  # 指定Kafka服务器的地址
    .option("subscribe", "your_topic_name")  # 指定要订阅的Kafka主题
    .option("startingOffsets", "earliest")  # 从最早的偏移量开始读取数据
    .option("failOnDataLoss", "false")  # 数据丢失时不失败
    .load()
)
# 从Kafka读取的数据是以二进制形式存储在"value"列中，使用from_json函数将其解析为符合定义Schema的DataFrame
parsed_df = kafka_df.select(
    from_json(col("value").cast("string"), product_schema).alias("data")
).select("data.*")

# 增强型数据清洗函数，包含多个数据清洗阶段，并输出每个阶段的结果
def clean_data(df):
    # 1. 字符串清理阶段
    # 去除字符串字段中的特殊字符，并去除首尾空格，以保证数据的一致性
    cleaned_strings_df = df.select(
        col("商品ID").alias("商品ID"),
        # 去除商品名称中的特殊字符，并去除首尾空格
        trim(regexp_replace(col("商品名称"), r'[^\w\u4e00-\u9fa5\s]', '')).alias("商品名称"),
        # 去除商品类别字段的首尾空格
        trim(col("商品类别")).alias("商品类别"),
        # 去除价格字段的首尾空格
        trim(col("价格")).alias("价格"),
        # 去除库存字段的首尾空格
        trim(col("库存")).alias("库存"),
        # 去除销量字段的首尾空格
        trim(col("销量")).alias("销量"),
        # 去除评分字段的首尾空格
        trim(col("评分")).alias("评分"),
        # 去除评论数字段的首尾空格
        trim(col("评论数")).alias("评论数"),
        # 去除上架周期字段的首尾空格
        trim(col("上架周期")).alias("上架周期"),
        # 去除上架日期字段的首尾空格
        trim(col("上架日期")).alias("上架日期")
    )
    # 将字符串清理后的结果输出到控制台，方便查看清理效果
    cleaned_strings_df.writeStream \
        .queryName("cleaned_strings")  # 查询名称，用于在Spark UI中识别
        .outputMode("append")  # 输出模式为追加，即每次有新数据到来时添加到输出中
        .format("console")  # 输出格式为控制台
        .option("truncate", "false")  # 不截断输出，完整显示数据
        .option("numRows", 5)  # 每次输出5行数据
        .start()

    # 2. 数据类型转换阶段
    # 将字符串类型的字段转换为合适的数值类型，并处理非法值
    type_converted_df = cleaned_strings_df.select(
        col("商品ID"),
        col("商品名称"),
        col("商品类别"),
        # 将价格字段转换为Double类型，如果不符合数字格式则置为None
        when(col("价格").rlike(r'^\d+(\.\d+)?$'), col("价格").cast(DoubleType())).otherwise(None).alias("价格"),
        # 将库存字段转换为Integer类型，如果不符合整数格式则置为None
        when(col("库存").rlike(r'^\d+$'), col("库存").cast(IntegerType())).otherwise(None).alias("库存"),
        # 将销量字段转换为Integer类型，如果不符合整数格式则置为None
        when(col("销量").rlike(r'^\d+$'), col("销量").cast(IntegerType())).otherwise(None).alias("销量"),
        # 将评论数字段转换为Integer类型，如果不符合整数格式则置为None
        when(col("评论数").rlike(r'^\d+$'), col("评论数").cast(IntegerType())).otherwise(None).alias("评论数"),
        # 将评分字段转换为Double类型，同时检查评分范围是否在0到5之间，不符合则置为None
        when(
            col("评分").rlike(r'^\d+(\.\d+)?$') &
            (col("评分").cast(DoubleType()) >= 0) &
            (col("评分").cast(DoubleType()) <= 5),
            col("评分").cast(DoubleType())
        ).otherwise(None).alias("评分"),
        # 将上架周期字段转换为Integer类型，如果不符合整数格式则置为None
        when(col("上架周期").rlike(r'^\d+$'), col("上架周期").cast(IntegerType())).otherwise(None).alias("上架周期"),
        col("上架日期")
    )
    # 将类型转换后的结果输出到控制台，方便查看转换效果
    type_converted_df.select(
        col("商品ID"),
        col("商品名称"),
        col("商品类别"),
        col("价格").alias("价格(数值)"),
        col("库存").alias("库存(数值)"),
        col("销量").alias("销量(数值)"),
        col("评分").alias("评分(数值)"),
        col("评论数").alias("评论数(数值)"),
        col("上架周期").alias("上架周期(数值)"),
        col("上架日期")
    ).writeStream \
        .queryName("type_converted")  # 查询名称，用于在Spark UI中识别
        .outputMode("append")  # 输出模式为追加
        .format("console")  # 输出格式为控制台
        .option("truncate", "false")  # 不截断输出
        .option("numRows", 5)  # 每次输出5行数据
        .start()

    # 3. 类别标准化阶段
    # 定义类别映射关系，将相似的类别统一为标准类别
    category_mapping = {
        "数码配件": "数码配件",
        "数码产品配件": "数码配件",
        "存储设备": "存储设备",
        "存储器": "存储设备",
        "电脑办公": "电脑办公",
        "计算机办公": "电脑办公"
    }
    # 定义用户自定义函数（UDF），用于标准化商品类别
    @udf(StringType())
    def standardize_category(cat):
        return category_mapping.get(cat, cat)
    # 应用UDF对商品类别进行标准化
    category_standardized_df = type_converted_df.withColumn(
        "商品类别", standardize_category(col("商品类别"))
    )
    # 将类别标准化后的结果输出到控制台，方便查看标准化效果
    category_standardized_df.select("商品ID", "商品名称", "商品类别").writeStream \
        .queryName("category_standardized")  # 查询名称，用于在Spark UI中识别
        .outputMode("append")  # 输出模式为追加
        .format("console")  # 输出格式为控制台
        .option("truncate", "false")  # 不截断输出
        .option("numRows", 5)  # 每次输出5行数据
        .start()

  

# 应用数据清洗函数，对解析后的数据进行清洗
cleaned_df = (
    parsed_df
    .transform(clean_data)  # 调用clean_data函数进行数据清洗
    .withColumn("上架日期", to_date("上架日期", "yyyy/M/d"))  # 将上架日期转换为日期类型
    .withColumn("处理时间", current_timestamp())  # 添加处理时间列，记录数据处理的时间
    .where(col("商品ID").isNotNull())  # 过滤掉商品ID为空的记录
    .filter(col("价格") > 0)  # 过滤掉价格小于等于0的记录
    .filter(col("商品类别").isin(["数码配件", "存储设备", "电脑办公"]))  # 过滤掉商品类别不在指定范围内的记录
)

# 输出最终清洗结果到控制台
final_cleaned_query = cleaned_df.select(
    "商品ID", "商品名称", "商品类别",
    "价格", "销量", "上架日期", "处理时间"
).writeStream \
    .queryName("final_cleaned")  # 查询名称，用于在Spark UI中识别
    .outputMode("append")  # 输出模式为追加
    .format("console")  # 输出格式为控制台
    .option("truncate", "false")  # 不截断输出
    .option("numRows", 5)  # 每次输出5行数据
    .start()

# 收集数据到内存
collected_data = []
# 定义foreachBatch函数，用于处理每个批次的数据
def foreach_batch_function(batch_df, batch_id):
    global collected_data
    # 将每个批次的数据转换为Pandas DataFrame，并转换为字典列表添加到collected_data中
    collected_data.extend(batch_df.toPandas().to_dict('records'))
    print(f"=== 批次 {batch_id} 清洗数据已收集 ===")

# 启动流式查询，将每个批次的数据传递给foreach_batch_function处理
query = cleaned_df.writeStream.foreachBatch(foreach_batch_function).start()
try:
    # 等待收集数据，设置超时时间为10秒
    query.awaitTermination(timeout=10)
except Exception as e:
    print(f"数据收集过程中出现异常: {e}")
finally:
    try:
        # 停止最终清洗结果的查询
        final_cleaned_query.stop()
        # 停止数据收集的查询
        query.stop()
    except Exception as e:
        print(f"停止查询时出现异常: {e}")
    print("数据收集完成！")

    if not collected_data:
        print(" ")
        # 停止SparkSession，释放资源
        spark.stop()
        exit()

    # 转换为Pandas DataFrame并整理列名
    collected_df = pd.DataFrame(collected_data)
    collected_df.rename(columns={
        'product_id': '商品ID',
        'product_name': '商品名称',
        'category': '商品类别',
        '价格': '价格',
        '销量': '销量',
        'shelf_date': '上架日期',
        'processing_time': '处理时间'
    }, inplace=True)

# 停止SparkSession，释放资源
spark.stop()
