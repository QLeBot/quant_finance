import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import yfinance as yf

from data_process import calculate_ratios
from scraper import scrape_stocks
from db_dev import initialize_country_table, initialize_stock_table, initialize_financial_ratios_table

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

def stock_found(ticker):
    """Check if the stock is found from API"""
    ticker = yf.Ticker(ticker)
    return f"{ticker} is not found" if ticker.info is None else f"{ticker} is found"

def store_ratios_in_db(symbol, name, country, ratios_df):
    """Store financial ratios in the database for each year."""
    try:
        with conn.cursor() as cur:
            # For each year in the ratios DataFrame
            for _, row in ratios_df.iterrows():
                # Convert date to datetime if it's not already
                date = pd.to_datetime(row['Date'])
                
                # Insert the data into the database
                cur.execute("""
                    INSERT INTO financial_ratios (
                        symbol, name, country, sector, industry, market_cap, date, enterprise_value,
                        price_per_share, price_per_earning, enterprise_value_per_ebitda,
                        price_per_free_cash_flow, enterprise_value_per_free_cash_flow,
                        roe, roic, cash_ratio, current_ratio, quick_ratio,
                        debt_to_equity, dividend_yield, payout_ratio,
                        gross_margin, operating_margin, net_margin
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    symbol,
                    name,
                    country,
                    row['Sector'],
                    row['Industry'],
                    row['Market cap'],
                    row['Date'],
                    row['Enterprise value'],
                    row['Price per share'],
                    row['Price per earning'],
                    row['Enterprise value per ebitda'],
                    row['Price per free cash flow'],
                    row['Enterprise value per free cash flow'],
                    row['ROE'],
                    row['ROIC'],
                    row['Cash ratio'],
                    row['Current ratio'],
                    row['Quick ratio'],
                    row['Debt to equity'],
                    row['Dividend yield'],
                    row['Payout ratio'],
                    row['Gross margin'],
                    row['Operating margin'],
                    row['Net margin']
                ))
            conn.commit()
            print(f"Successfully stored ratios for {symbol} for years {[pd.to_datetime(d).date() for d in ratios_df['Date']]}")
    except Exception as e:
        print(f"Error storing ratios for {symbol}: {e}")

def data_from_db(region):
    # get the data from the database
    cursor.execute(f"SELECT symbol, name, country_code FROM stock WHERE country_code = '{region['code']}'")
    stock_list = cursor.fetchall()
    symbol_list = [stock[0] for stock in stock_list]
    name_list = [stock[1] for stock in stock_list]
    country = region['code']
    return symbol_list, name_list, country

def data_from_csv(region):
    symbol_list = []
    name_list = []
    country = region['code']
    
    for file in os.listdir("stock_analyzer/data/raw"):
        if file.endswith(".csv") and file.endswith(f"{region['code']}.csv") :
            stock_data = pd.read_csv("stock_analyzer/data/raw/" + file)
            if not stock_data.empty:
                # extract symbol list from the dataframe and filter out NaN values
                symbol_list = stock_data['symbol'].dropna().tolist()
                name_list = stock_data['name'].dropna().tolist()

    return symbol_list, name_list, country

def process_stocks(regions):
    # create a unique dataframe
    df = pd.DataFrame()

    for region in regions:
        #symbol_list, name_list, country = data_from_csv(region)

        symbol_list, name_list, country = data_from_db(region)

        if symbol_list:
            # loop through the symbol list and get the info and financial data
            for symbol in symbol_list:
                print(symbol)
                try:
                    #industry, sector, country, market_cap = get_info(symbol)
                        
                    # Get financial data and calculate ratios
                    ticker = yf.Ticker(symbol)
                    ratios_df = calculate_ratios(ticker)
                    
                    # Store ratios in database
                    store_ratios_in_db(symbol, name_list[symbol_list.index(symbol)], country, ratios_df)
                    
                    # Create a dictionary with all the data for the CSV file
                    stock_dict = {
                        'symbol': symbol,
                        'name': name_list[symbol_list.index(symbol)],
                        'country': country,
                        **ratios_df.iloc[0].to_dict()  # Add the calculated ratios
                    }
                    
                    # Create a new row and append it to the main DataFrame
                    new_row = pd.DataFrame([stock_dict])
                    df = pd.concat([df, new_row], ignore_index=True)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue

    # save the main dataframe to a csv file
    df.to_csv('stock_analyzer/data/financials.csv', index=False)

def main():
    custom_regions = [
        {"code": "gb", "name": "United Kingdom"},
        {"code": "fr", "name": "France"}
    ]
    #results = scrape_stocks(market_cap_min="300M", market_cap_max="2B", regions=regions, headless=True)
    #print("Basic scraping completed")
    process_stocks(regions)
    
    # Get financial data for testing
    #ticker = yf.Ticker("6666.TW")
    #ratios_df = calculate_ratios(ticker)
    #print(ratios_df)

    #pd.DataFrame(ratios_df).to_csv('stock_analyzer/data/ratios.csv', index=False)
    
    #print(ticker.info)
    #print(ticker.get_balance_sheet(freq='yearly'))
    #print(ticker.get_income_stmt(freq='yearly'))
    #print(ticker.get_cash_flow(freq='yearly'))
    #print(ticker.get_financials(freq='yearly'))
    
    #print(f"Balance Sheet: \n{ticker.get_balance_sheet(freq='yearly').index.tolist()} \n")
    #print(f"Income Statement: \n{ticker.get_income_stmt(freq='yearly').index.tolist()} \n")
    #print(f"Cash Flow: \n{ticker.get_cash_flow(freq='yearly').index.tolist()} \n")
    #print(f"Financials: \n{ticker.get_financials(freq='yearly').index.tolist()} \n")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
