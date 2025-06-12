import pandas as pd
import json
from kafka import KafkaProducer
import logging

# 配置日志（启用 INFO 级别，方便调试）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Kafka 生产者，增加错误处理和重试机制
producer = KafkaProducer(
    bootstrap_servers=['192.168.149.129:9092'],
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
    key_serializer=lambda k: json.dumps(k).encode('utf-8'),
    retries=3,  # 重试次数
    max_in_flight_requests_per_connection=1,  # 确保消息按顺序发送
    request_timeout_ms=30000,  # 请求超时时间（毫秒）
    linger_ms=5,  # 批量发送等待时间
    batch_size=16384  # 批次大小
)


def send_product_data(product_data, topic):
    """
    将商品数据发送到 Kafka 主题。

    :param product_data: 从 CSV 文件读取的数据列表
    :param topic: Kafka 主题名称
    """
    success_records = 0
    for index, row in product_data.iterrows():
        key = {"index": index}
        value = row.to_dict()
        try:
            # 同步发送并获取确认，同时获取 RecordMetadata 信息
            future = producer.send(topic, value=value, key=key)
            record_metadata = future.get(timeout=10)  # 等待最多 10 秒
            partition = record_metadata.partition
            logging.info(f"消息发送成功，索引: {index}，发送到分区: {partition}")
            success_records += 1
        except Exception as e:
            logging.error(f"消息发送失败，索引: {index}, 错误: {e}")
    return success_records


if __name__ == "__main__":
    try:
        # 读取商品数据，只读取前 1000 条，修改文件路径为你上传文件的路径
        product_data = pd.read_csv("/home/hadoop/商品_含上架日期.CSV").head(1000)

        if product_data.empty:
            raise ValueError("商品数据为空，请检查 CSV 文件路径和内容")

        # 发送数据
        topic = "your_topic_name"
        total_sent = send_product_data(product_data, topic)

        # 输出详细结果
        print(f"数据发送完成！总共尝试发送 {len(product_data)} 条商品数据")
        print(f"成功发送: {total_sent} 条")

    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
    finally:
        # 确保所有消息都被发送
        producer.flush()
        producer.close()
        logging.info("Kafka 生产者已关闭")