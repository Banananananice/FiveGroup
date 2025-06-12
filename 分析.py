
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler



# 定义数据收集函数
collected_data = []
def foreach_batch_function(batch_df, batch_id):
    global collected_data
    collected_data.extend(batch_df.toPandas().to_dict('records'))
    print(f"=== 批次 {batch_id} 清洗数据已收集 ===")

# 启动流处理并收集数据（移至函数外部）
print("开始收集数据...")
query = cleaned_df.writeStream.foreachBatch(foreach_batch_function).start()

# 等待一段时间收集数据（实际生产中建议使用触发器）
query.awaitTermination(timeout=10)

query.stop()
print("数据收集完成！")

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

# 1. 时序模式挖掘
print("\n1. 时序模式挖掘")
print("-" * 50)

daily_sales = collected_df.groupby('上架日期')['销量'].sum().reset_index()
daily_sales['上架日期'] = pd.to_datetime(daily_sales['上架日期'])
daily_sales.set_index('上架日期', inplace=True)
daily_sales = daily_sales.asfreq('D').fillna(0)

print("\n销量趋势:")
print(daily_sales.tail())

model = ARIMA(daily_sales['销量'], order=(1, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=7)
print("\n未来7天销量预测:")
print(forecast)

print("\n\n2. 商品价格区间聚类分析")
categories = ["数码配件", "存储设备", "电脑办公"]
for category in categories:
    # 1. 选数据：过滤出当前品类
    category_df = spark.createDataFrame(collected_df[collected_df['category'] == category])
    if category_df.count() == 0:
        print(f"[警告] {category}类别数据不足，无法分析")
        continue

    # 2. 处理特征：把价格转成KMeans需要的格式
    assembler = VectorAssembler(inputCols=["价格"], outputCol="features")
    vector_df = assembler.transform(category_df)

    # 3. 跑KMeans：分3类，找价格区间
    kmeans = KMeans(k=3, seed=42)  # 固定seed让结果可重复
    model = kmeans.fit(vector_df)
    centers = model.clusterCenters()  # 3个类的中心价格
    centers.sort(key=lambda x: x[0])   # 按价格从小到大排

    # 4. 算价格区间：结合最小、最大价格，划分低、中、高区间
    min_price = category_df.agg({"价格": "min"}).collect()[0][0]
    max_price = category_df.agg({"价格": "max"}).collect()[0][0]
    price_ranges = []
    for i in range(3):
        if i == 0:
            # 第一类：最低价格 ~ 第一类中心
            lower, upper = min_price, centers[i][0]
        elif i == 2:
            # 第三类：第二类中心 ~ 最高价格
            lower, upper = centers[i-1][0], max_price
        else:
            # 第二类：第一类中心 ~ 第二类中心
            lower, upper = centers[i-1][0], centers[i][0]
        price_ranges.append((round(lower, 2), round(upper, 2)))

    # 5. 输出结果+建议
    print(f"\n{category}价格区间分布（分3类）:")
    for i, (lower, upper) in enumerate(price_ranges):
        print(f"  区间{i+1}: {lower}元 - {upper}元")
    main_range = price_ranges[1]  # 中间区间是主流
    print(f"  主要价格区间: {main_range[0]}元 - {main_range[1]}元")
    print(f"  建议：大部分商品在{main_range}，定价优先参考这里；高低区间可以做差异化产品（比如高端款、低价引流款）")