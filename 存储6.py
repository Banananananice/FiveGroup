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

# 收集数据到内存
collected_data = []

def foreach_batch_function(batch_df, batch_id):
    global collected_data
    collected_data.extend(batch_df.toPandas().to_dict('records'))
    print(f"=== 批次 {batch_id} 清洗数据已收集 ===")

print("开始收集数据...")
query = cleaned_df.writeStream.foreachBatch(foreach_batch_function).start()

try:
    # 等待一段时间收集数据（实际生产中建议使用触发器）
    query.awaitTermination(timeout=10)
except:
    print("手动停止数据收集")
finally:
    query.stop()
    print("数据收集完成！")

    # 转换为 Pandas DataFrame
    if not collected_data:
        print("未收集到任何数据！请检查Kafka连接和数据来源。")
        spark.stop()
        exit()

    collected_df = pd.DataFrame(collected_data)

    # ============ 数据分析报告 ============
    print("\n\n===== 数据分析报告 =====")

    # 1. 时序模式挖掘
    print("\n1. 时序模式挖掘")
    print("-" * 50)
    try:
        daily_sales = collected_df.groupby('上架日期')['销量'].sum().reset_index()
        daily_sales['上架日期'] = pd.to_datetime(daily_sales['上架日期'])
        daily_sales.set_index('上架日期', inplace=True)
        daily_sales = daily_sales.asfreq('D').fillna(0)

        print("\n销量趋势:")
        print(daily_sales.tail())

        model = ARIMA(daily_sales['销量'], order=(0, 1, 1))
        model_fit = model.fit()
        print('\nARIMA 模型摘要:')
        print(model_fit.summary())

        forecast = model_fit.forecast(steps=7)
        print("\n未来7天销量预测:")
        print(forecast)
    except Exception as e:
        print(f'时序分析错误: {e}')

    # ============ 新增：KMeans价格聚类分析 ============
    print("\n\n2. 商品价格区间聚类分析")
    print("-" * 50)
    categories = ["数码配件", "存储设备", "电脑办公"]
    for category in categories:
        category_df = spark.createDataFrame(collected_df[collected_df['category'] == category])
        if category_df.count() == 0:
            print(f"[警告] {category}类别数据不足，无法进行聚类分析")
            continue

        assembler = VectorAssembler(inputCols=["价格"], outputCol="features")
        vector_df = assembler.transform(category_df)

        kmeans = KMeans(k=3, seed=42)
        model = kmeans.fit(vector_df)
        centers = model.clusterCenters()
        centers.sort(key=lambda x: x[0])

        price_ranges = []
        min_price = category_df.agg({"价格": "min"}).collect()[0][0]
        max_price = category_df.agg({"价格": "max"}).collect()[0][0]
        for i in range(3):
            if i == 0:
                lower, upper = min_price, centers[i][0]
            elif i == 2:
                lower, upper = centers[i - 1][0], max_price
            else:
                lower, upper = centers[i - 1][0], centers[i][0]
            price_ranges.append((round(lower, 2), round(upper, 2)))

        print(f"\n{category}价格区间分布（分为3类）:")
        for i, (lower, upper) in enumerate(price_ranges):
            print(f"  区间{i + 1}: {lower}元 - {upper}元")
        main_range = price_ranges[1]
        print(f"  主要价格区间: {main_range[0]}元 - {main_range[1]}元")
        print(f"  建议定价策略: 该类别商品价格集中在{main_range[0]}元 - {main_range[1]}元，"
              f"定价时可重点参考此区间，或针对高低区间开发差异化产品")

    # ============ 新增：数据可视化报告 ============
    print("\n\n===== 数据可视化报告 =====")
    output_dir = "/home/hadoop/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 1. 各类别价格与销量散点图
    plt.figure(figsize=(15, 5))
    for i, category in enumerate(["数码配件", "存储设备", "电脑办公"], start=1):
        plt.subplot(1, 3, i)
        category_data = collected_df[collected_df['category'] == category]
        sns.scatterplot(x='价格', y='销量', data=category_data,
                        color=['blue', 'green', 'red'][i - 1], alpha=0.6)
        plt.title(f'{category} - 价格与销量关系')
        plt.xlabel('价格')
        plt.ylabel('销量')
        plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_price_sales_scatter.png")
    print(f"[可视化] 已生成类别价格销量散点图: {output_dir}/category_price_sales_scatter.png")

    # 2. 销量预测图
    try:
        # 销量预测图
        forecast_df = pd.DataFrame({
            '销量': forecast,
            '日期': pd.date_range(start=daily_sales.index[-1], periods=8)[1:]
        })
        forecast_df.set_index('日期', inplace=True)

        plt.figure(figsize=(12, 6))
        plt.plot(daily_sales.index, daily_sales['销量'],
                 marker='o', linestyle='-', color='green', label='历史销量')
        plt.plot(forecast_df.index, forecast_df['销量'],
                 marker='x', linestyle='--', color='red', label='预测销量')
        plt.title('销量历史与未来7天预测')
        plt.xlabel('日期')
        plt.ylabel('销量')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sales_forecast.png")
        print(f"[可视化] 已生成销量预测图: {output_dir}/sales_forecast.png")
    except Exception as e:
        print(f"[可视化] 销量趋势/预测图生成失败: {e}")

# 停止 SparkSession
spark.stop()