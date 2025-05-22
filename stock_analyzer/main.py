import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import yfinance as yf

from scraper import scrape_stocks

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
    return ticker.info['industry'], ticker.info['sector'], ticker.info['country'], ticker.info['marketCap']

def get_financial_data(ticker):
    # get financial data from the database
    ticker = yf.Ticker(ticker)
    # Get the financial statements
    balance_sheet = ticker.get_balance_sheet(freq='yearly')
    income_stmt = ticker.get_income_stmt(freq='yearly')
    cash_flow = ticker.get_cash_flow(freq='yearly')
    
    # Transpose the dataframes to get dates as columns
    balance_sheet = balance_sheet.T
    income_stmt = income_stmt.T
    cash_flow = cash_flow.T
    
    # Reset index to make dates a column
    balance_sheet = balance_sheet.reset_index()
    income_stmt = income_stmt.reset_index()
    cash_flow = cash_flow.reset_index()
    
    # Rename the date column
    balance_sheet = balance_sheet.rename(columns={'index': 'Date'})
    income_stmt = income_stmt.rename(columns={'index': 'Date'})
    cash_flow = cash_flow.rename(columns={'index': 'Date'})
    
    # Merge the dataframes
    merged_df = pd.merge(balance_sheet, income_stmt, on='Date', how='outer', suffixes=('_bs', '_is'))
    merged_df = pd.merge(merged_df, cash_flow, on='Date', how='outer', suffixes=('', '_cf'))

    return merged_df
    

def process_stocks():
    # create a unique dataframe
    df = pd.DataFrame()

    # Loop through all files in the raw folder and read and insert data into the database
    # If file is empty, skip it
    for file in os.listdir("stock_analyzer/data/raw"):
        if file.endswith(".csv"):
            stock_data = pd.read_csv("stock_analyzer/data/raw/" + file)
            if not stock_data.empty:
                # extract symbol list from the dataframe and filter out NaN values
                symbol_list = stock_data['symbol'].dropna().tolist()
                country = stock_data['country'][0]
                name_list = stock_data['name'].dropna().tolist()
                print(symbol_list)

                # loop through the symbol list and get the info and financial data
                for symbol in symbol_list:
                    print(symbol)
                    try:
                        industry, sector, country, market_cap = get_info(symbol)
                        fin_df = get_financial_data(symbol)
                        
                        # Get the most recent year's data (first row of financial data)
                        if not fin_df.empty:
                            latest_fin_data = fin_df.iloc[0].to_dict()
                            
                            # Create a dictionary with all the data
                            stock_dict = {
                                'symbol': symbol,
                                'name' : name_list[symbol_list.index(symbol)],
                                'country': country,
                                'sector': sector,
                                'industry': industry,
                                'market_cap': market_cap,
                                **latest_fin_data  # Unpack the financial data into the dictionary
                            }
                            
                            # Create a new row and append it to the main DataFrame
                            new_row = pd.DataFrame([stock_dict])
                            df = pd.concat([df, new_row], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing {symbol}: {str(e)}")
                        continue
                
                #print(df)

    # save the dataframe to a csv file
    df.to_csv('stock_analyzer/data/financials.csv', index=False)

cursor.close()
conn.close()

def main():
    custom_regions = [
        {"code": "us", "name": "United States"},
        {"code": "gb", "name": "United Kingdom"},
        {"code": "fr", "name": "France"}
    ]
    #results = scrape_stocks(market_cap_min="300M", market_cap_max="2B", regions=custom_regions, headless=False)
    #print("Basic scraping completed")
    process_stocks()
    
    # Get financial data for AAPL
    #ticker = yf.Ticker("AAPL")
    #
    ## Get the financial statements
    #balance_sheet = ticker.get_balance_sheet(freq='yearly')
    #income_stmt = ticker.get_income_stmt(freq='yearly')
    #cash_flow = ticker.get_cash_flow(freq='yearly')
    #
    ## Transpose the dataframes to get dates as columns
    #balance_sheet = balance_sheet.T
    #income_stmt = income_stmt.T
    #cash_flow = cash_flow.T
    #
    ## Reset index to make dates a column
    #balance_sheet = balance_sheet.reset_index()
    #income_stmt = income_stmt.reset_index()
    #cash_flow = cash_flow.reset_index()
    #
    ## Rename the date column
    #balance_sheet = balance_sheet.rename(columns={'index': 'Date'})
    #income_stmt = income_stmt.rename(columns={'index': 'Date'})
    #cash_flow = cash_flow.rename(columns={'index': 'Date'})
    #
    ## Merge the dataframes
    #merged_df = pd.merge(balance_sheet, income_stmt, on='Date', how='outer', suffixes=('_bs', '_is'))
    #merged_df = pd.merge(merged_df, cash_flow, on='Date', how='outer', suffixes=('', '_cf'))
    #
    ## Save to CSV
    #merged_df.to_csv('stock_analyzer/data/financials.csv', index=False)
    #print("Financial data saved successfully")

if __name__ == "__main__":
    main()
