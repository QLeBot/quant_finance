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

def initialize_country_table():
    """Create a table for country data if it doesn't exist."""
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS country (
            country_code VARCHAR(2) PRIMARY KEY,
            country_name VARCHAR(255)
        );
        """)
        conn.commit()
        print("Country table created or already exists")
    except Exception as e:
        print(f"Error creating country table: {e}")

    # insert country data into the database
    args_str = ','.join(cursor.mogrify("(%s,%s)", (x['code'], x['name'])).decode('utf-8') for x in regions)
    cursor.execute("INSERT INTO country (country_code, country_name) VALUES " + (args_str))
    conn.commit()
    print("Country data inserted into the database")

def initialize_stock_table():
    """Create a table for stock data if it doesn't exist."""
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock (
            symbol VARCHAR(20) PRIMARY KEY,
            name VARCHAR(255),
            country_code VARCHAR(2),
            date_added TIMESTAMP,
            CONSTRAINT fk_country
            FOREIGN KEY (country_code)
            REFERENCES country(country_code)
        );
        """)
        conn.commit()
        print("Stock table created or already exists")
    except Exception as e:
        print(f"Error creating stock table: {e}")

def initialize_financial_ratios_table():
    """Create the financial ratios table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS financial_ratios (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    name VARCHAR(255),
                    country VARCHAR(2),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    market_cap NUMERIC,
                    date DATE,
                    enterprise_value NUMERIC,
                    price_per_share NUMERIC,
                    price_per_earning NUMERIC,
                    enterprise_value_per_ebitda NUMERIC,
                    price_per_free_cash_flow NUMERIC,
                    enterprise_value_per_free_cash_flow NUMERIC,
                    roe NUMERIC,
                    roic NUMERIC,
                    cash_ratio NUMERIC,
                    current_ratio NUMERIC,
                    quick_ratio NUMERIC,
                    debt_to_equity NUMERIC,
                    dividend_yield NUMERIC,
                    payout_ratio NUMERIC,
                    gross_margin NUMERIC,
                    operating_margin NUMERIC,
                    net_margin NUMERIC
                );
            """)
            conn.commit()
            print("Financial ratios table created or already exists")
    except Exception as e:
        print(f"Error creating financial ratios table: {e}")

def main():
    #initialize_country_table()
    #initialize_stock_table()
    initialize_financial_ratios_table()
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()




