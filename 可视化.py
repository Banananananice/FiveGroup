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

# 停止 SparkSession
spark.stop()