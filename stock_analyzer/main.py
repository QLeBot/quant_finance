import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

# Create a table for stock data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        name VARCHAR(255),
        country VARCHAR(255),
        sector VARCHAR(255),
        industry VARCHAR(255)
    );
""")

#ex_line = "https://finance.yahoo.com/quote/POLL.BA/"
#ex_line_split = ex_line.split("/")
# get the second last element
#print(ex_line_split[-2])


# Read all txt files from the data folder in the raw folder
# Loop through all files in the raw folder and read and insert data into the database
# If file is empty, skip it
for file in os.listdir("stock_analyzer/data/raw"):
    if file.endswith(".txt"):
        with open("stock_analyzer/data/raw/" + file, "r") as file:
            if not file.read():
                continue
            file.seek(0)
            # get country from the file name
            country = file.name.split("_")[-1].split(".")[0]
            # get stock data from the file each line is a stock with symbol to split
            stock_data = [line.split("/")[-2] for line in file.readlines()]

            # create a list of tuples with all the data
            stock_data = [(symbol, None, country, None, None) for symbol in stock_data]
            print(stock_data)

            # insert data into the database
            args_str = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in stock_data)
            cursor.execute("INSERT INTO stock_data (symbol, name, country, sector, industry) VALUES " + (args_str))

# select statement to display output
sql1 = '''select * from stock_data;'''

# executing sql statement
cursor.execute(sql1)

# fetching rows
for i in cursor.fetchall():
    print(i)

conn.commit()

file.close()

cursor.close()
conn.close()
