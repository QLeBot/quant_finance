from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Example scraped data
data = {
    "country": "Germany",
    "date": "2025-05-15",
    "content": "Some scraped content..."
}

producer.send("country-data", value=data)
producer.flush()
