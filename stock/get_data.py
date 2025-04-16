import os
from dotenv import load_dotenv
import yfinance as yf

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca client
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)


def alpaca_data(symbol, start_date, end_date, timeframe, adjustment):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment=adjustment
    )
    data = stock_client.get_stock_bars(request_params).df
    return data

def yfinance_data(symbol, start_date, end_date, timeframe):
    # Download data from yfinance
    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    data.dropna(inplace=True)
    return data



