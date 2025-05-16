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

# default US when loading the page
regions = [
    {"code": "us",
    "name": "United States"
    },
    {"code": "fr",
    "name": "France"
    },
    {"code": "ar",
    "name": "Argentina"
    },
    {"code": "at",
    "name": "Austria"
    },
    {"code": "au",
    "name": "Australia"
    },
    {"code": "be",
    "name": "Belgium"
    },
    {"code": "br",
    "name": "Brazil"
    },
    {"code": "ca",
    "name": "Canada"
    },
    {"code": "ch",
    "name": "Switzerland"
    },
    {"code": "cl",
    "name": "Chile"
    },
    {"code": "cn",
    "name": "China"
    },
    {"code": "cz",
    "name": "Czechia"
    },
    {"code": "de",
    "name": "Germany"
    },
    {"code": "dk",
    "name": "Denmark"
    },
    {"code": "ee",
    "name": "Estonia"
    },
    {"code": "gb",
    "name": "United Kingdom"
    },
    {"code": "gr",
    "name": "Greece"
    },
    {"code": "hk",
    "name": "Hong Kong SAR China"
    },
    {"code": "hu",
    "name": "Hungary"
    },
    {"code": "id",
    "name": "Indonesia"
    },
    {"code": "is",
    "name": "Iceland"
    },
    {"code": "it",
    "name": "Italy"
    },
    {"code": "jp",
    "name": "Japan"
    },
    {"code": "kr",
    "name": "South Korea"
    },
    {"code": "kw",
    "name": "Kuwait"
    },
    {"code": "lk",
    "name": "Sri Lanka"
    },
    {"code": "lt",
    "name": "Lithuania"
    },
    {"code": "lv",
    "name": "Latvia"
    },
    {"code": "mx",
    "name": "Mexico"
    },
    {"code": "my",
    "name": "Malaysia"
    },
    {"code": "nl",
    "name": "Netherlands"
    },
    {"code": "ph",
    "name": "Philippines"
    },
    {"code": "pk",
    "name": "Pakistan"
    },
    {"code": "pl",
    "name": "Poland"
    },
    {"code": "pt",
    "name": "Portugal"
    },
    {"code": "za",
    "name": "South Africa"
    },
    {"code": "sr",
    "name": "Suriname"
    },
    {"code": "th",
    "name": "Thailand"
    },
    {"code": "tr",
    "name": "Turkey"
    },
    {"code": "tw",
    "name": "Taiwan"
    },
    {"code": "ve",
    "name": "Venezuela"
    }
]

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

# Create a table for country data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS country (
        country_id SERIAL PRIMARY KEY,
        country_code VARCHAR(2) NOT NULL,
        country_name VARCHAR(255)
    );
""")

# insert country data into the database
args_str = ','.join(cursor.mogrify("(%s,%s)", (x['code'], x['name'])).decode('utf-8') for x in regions)
cursor.execute("INSERT INTO country (country_code, country_name) VALUES " + (args_str))


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
#sql1 = '''select * from stock_data;'''

# executing sql statement
#cursor.execute(sql1)

# fetching rows
#for i in cursor.fetchall():
#    print(i)

conn.commit()

file.close()

cursor.close()
conn.close()
