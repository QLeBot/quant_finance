from kafka import KafkaConsumer
import psycopg2
import json

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="yourdb",
    user="youruser",
    password="yourpass",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Kafka Consumer
consumer = KafkaConsumer(
    'country-data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for msg in consumer:
    record = msg.value
    country = record["country"]
    date = record["date"]
    content = record["content"]

    cursor.execute(
        "INSERT INTO scraped_data (country, date, content) VALUES (%s, %s, %s)",
        (country, date, content)
    )
    conn.commit()
