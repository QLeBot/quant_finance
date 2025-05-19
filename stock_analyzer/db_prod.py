import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    dbname=os.getenv("DB_PROD_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

# replicate dev db to prod db
cursor.execute("""
    CREATE TABLE IF NOT EXISTS country AS SELECT * FROM dev_db.country;
""")