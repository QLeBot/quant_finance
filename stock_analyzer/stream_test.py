import json
from datetime import datetime, timedelta
import yfinance as yf
from kafka import KafkaProducer, KafkaConsumer
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
STOCK_TOPIC = 'stock_data'
ENRICHED_STOCK_TOPIC = 'enriched_stock_data'

def get_db_connection():
    """Create a connection to PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

def fetch_stocks_from_db() -> List[Dict]:
    """Fetch stock symbols from PostgreSQL database."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT symbol FROM stocks")
            stocks = cur.fetchall()
            return [dict(stock) for stock in stocks]
    except Exception as e:
        logger.error(f"Error fetching stocks from database: {e}")
        return []
    finally:
        conn.close()

def create_kafka_producer():
    """Create and return a Kafka producer."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

def create_kafka_consumer(topic: str):
    """Create and return a Kafka consumer."""
    return KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )

def fetch_yahoo_finance_data(symbol: str) -> Dict:
    """Fetch stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'symbol': symbol,
            'current_price': info.get('currentPrice'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield'),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
        return None

def save_enriched_data(data: Dict):
    """Save enriched stock data back to PostgreSQL."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_data (
                    symbol, current_price, market_cap, pe_ratio, 
                    dividend_yield, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                data['symbol'],
                data['current_price'],
                data['market_cap'],
                data['pe_ratio'],
                data['dividend_yield'],
                data['timestamp']
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving enriched data to database: {e}")
    finally:
        conn.close()

def producer_process():
    """Kafka producer process that reads from PostgreSQL and sends to Kafka."""
    producer = create_kafka_producer()
    
    while True:
        try:
            stocks = fetch_stocks_from_db()
            for stock in stocks:
                producer.send(STOCK_TOPIC, value=stock)
                logger.info(f"Sent stock {stock['symbol']} to Kafka")
            
            # Wait for 5 minutes before next iteration
            time.sleep(300)
        except Exception as e:
            logger.error(f"Error in producer process: {e}")
            time.sleep(60)

def consumer_process():
    """Kafka consumer process that fetches Yahoo Finance data and saves to PostgreSQL."""
    consumer = create_kafka_consumer(STOCK_TOPIC)
    
    for message in consumer:
        try:
            stock_data = message.value
            enriched_data = fetch_yahoo_finance_data(stock_data['symbol'])
            
            if enriched_data:
                save_enriched_data(enriched_data)
                logger.info(f"Processed and saved data for {stock_data['symbol']}")
        except Exception as e:
            logger.error(f"Error in consumer process: {e}")

if __name__ == "__main__":
    import multiprocessing
    
    # Start producer and consumer processes
    producer = multiprocessing.Process(target=producer_process)
    consumer = multiprocessing.Process(target=consumer_process)
    
    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()
