import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import yfinance as yf

import scraper

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



def stock_found(ticker):
    """Check if the stock is found from API"""
    ticker = yf.Ticker(ticker)
    return f"{ticker} is not found" if ticker.info is None else f"{ticker} is found"

def get_info(ticker):
    ticker = yf.Ticker(ticker)
    return ticker.info['industry'], ticker.info['sector']

def get_financial_data(ticker):
    # get financial data from the database
    ticker = yf.Ticker(ticker)
    # get the balance sheet
    balance_sheet = ticker.balance_sheet
    # get the income statement
    #income_statement = ticker.financials
    income_statement = ticker.get_income_stmt(freq='yearly')
    # get the cash flow statement
    cash_flow = ticker.cash_flow

    # process the financials and create a dataframe with all the data
    # create a dataframe with all the data
    df = pd.DataFrame(columns=['Date', 'Revenue', 'Net Income', 'Cash Flow', 'Assets', 'Liabilities', 'Equity'])
    for date, data in income_statement.items():
        df = df.append({'Date': date, 'Revenue': data['Revenue'], 'Net Income': data['Net Income'], 'Cash Flow': data['Cash Flow'], 'Assets': data['Assets'], 'Liabilities': data['Liabilities'], 'Equity': data['Equity']}, ignore_index=True)
    print(df)
    # save the dataframe to a csv file
    df.to_csv('stock_analyzer/data/financials.csv', index=False)
    # save the balance sheet to a csv file
    balance_sheet.to_csv('stock_analyzer/data/balance_sheet.csv', index=False)
    # save the cash flow statement to a csv file
    cash_flow.to_csv('stock_analyzer/data/cash_flow.csv', index=False)
    
    return balance_sheet, income_statement, cash_flow


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

            temp = stock_data[0][0]
            print(temp)
            print(get_info(temp))
            print(get_financial_data(temp))
            #for symbol in stock_data:
            #    print(stock_found(symbol))

            # create a list of tuples with all the data
            #stock_data = [(symbol, None, country, None, None) for symbol in stock_data]
            #print(stock_data)

            # insert data into the database
            #args_str = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in stock_data)
            #cursor.execute("INSERT INTO stock_data (symbol, name, country, sector, industry) VALUES " + (args_str))

conn.commit()

file.close()

cursor.close()
conn.close()

from scraper import scrape_stocks

def main():
    # Example 1: Basic usage with default parameters
    results = scrape_stocks()
    print("Basic scraping completed")
    
    # Example 2: Custom market cap range
    # results = scrape_stocks(market_cap_min="500M", market_cap_max="5B")
    
    # Example 3: Specific regions only
    # custom_regions = [
    #     {"code": "us", "name": "United States"},
    #     {"code": "gb", "name": "United Kingdom"},
    #     {"code": "fr", "name": "France"}
    # ]
    # results = scrape_stocks(regions=custom_regions)
    
    # Example 4: Headless mode
    # results = scrape_stocks(headless=True)

if __name__ == "__main__":
    main()
