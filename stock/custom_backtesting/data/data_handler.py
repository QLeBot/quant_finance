import polars as pl
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv

class DataHandler:
    """Class to handle all data functionality for the trading strategy."""
    
    def __init__(self, ticker, start_date, end_date):
        """Initialize the data with data and ticker symbol."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = None

    def yfinance_data(self):
        """Fetch stock data from yfinance."""
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(self.df.head())
        return pl.from_pandas(self.df.reset_index())
    
    def alpaca_data(self):
        """Fetch stock data from Alpaca."""
        self.client = self.load_environment()
        self.df = self.fetch_stock_data()
        print(self.df.head())
        return self.df
    
    def fetch_stock_data(self):
        """Fetch stock data from Alpaca."""
        request_params = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            timeframe=TimeFrame.Day,
            start=self.start_date,
            end=self.end_date,
            adjustment="split"
        )
        bars = self.client.get_stock_bars(request_params)
        print(bars.df.head())
        return pl.from_pandas(bars.df.reset_index())

    def load_environment(self):
        """Load environment variables and initialize Alpaca client."""
        load_dotenv()
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        return client 