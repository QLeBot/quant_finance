import yfinance as yf
import os
import pandas as pd

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca client
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=False)

active_assets = trading_client.get_all_assets(raw_data=False, filter=GetAssetsRequest(status='active'))

print(active_assets)

AAPL = yf.Ticker('AAPL')
AAPL.history(period='5y', interval='1d')

#print(AAPL.financials)
#print(AAPL.balance_sheet)
#print(AAPL.cashflow)