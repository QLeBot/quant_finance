import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    dbname=os.getenv("DB_DEV_NAME"),
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

# Drop the stock_data table for testing purposes
cursor.execute("DROP TABLE IF EXISTS stock")

# Create a table for stock data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock (
        stock_id SERIAL PRIMARY KEY,
        stock_symbol VARCHAR(20) NOT NULL,
        stock_name VARCHAR(255),
        country_id INT,
        stock_sector VARCHAR(255),
        stock_industry VARCHAR(255),
        FOREIGN KEY (country_id) REFERENCES country(country_id)
    );
""")



