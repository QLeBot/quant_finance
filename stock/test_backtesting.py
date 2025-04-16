import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from backtesting import Backtest, Strategy

from ta.momentum import RSIIndicator
from ta.trend import MACD
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca client
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Stock split data for multiple symbols
stock_splits = {
    'NVDA': {
        '2000-06-27': 2.0,
        '2001-09-12': 2.0,
        '2006-04-07': 2.0,
        '2007-09-11': 1.5,
        '2021-07-20': 4.0,
        '2024-06-10': 10.0,
    },
    'TSLA': {
        '2020-08-31': 5.0,
        '2022-08-25': 3.0,
    },
    'MC.PA': {
        '2000-07-03': 5.0,
        '1999-06-21': 1.100011,
        '1994-07-06': 1.10999,
    },
    'ATO': {
        '1985-06-13': 2.0,
        '1994-05-17': 1.5
    },
    'ATOS': {
        '1999-03-24': 2.0,
        '2025-04-24': 0.0001
    }
}

# Parameters
#symbols = ["NVDA", "SPY", "TSLA", "MC.PA", "ATO", "ATOS"]
#symbols = ["NVDA"]
#symbols = ["NVDA", "ATOS"]
#symbols = ["NVDA", "ATOS" , "ATO"]
symbols = ["SPY"]


initial_cash = 10000
start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

# Request 1-hour data for primary analysis and 4-hour data for confirmation
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    adjustment='split'
)

all_data = stock_client.get_stock_bars(request_params).df
# Rename columns to match the expected format
all_data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

print(all_data.head())

all_data = all_data.reset_index()
all_data = all_data[['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
all_data.set_index('timestamp', inplace=True)

print(all_data.head())

print(type(all_data['Close'])) # <class 'pandas.core.series.Series'>
print(type(all_data.index)) # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>


class RSI_MACD_Strategy(Strategy):
    rsi_threshold = 70
    macd_threshold = 30
    rsi_window = 14
    macd_window_slow = 26
    macd_window_fast = 12
    macd_window_sign = 9

    def init(self):
        # Convert the backtesting._util._Array to a Pandas Series
        close_series = pd.Series(self.data.Close, index=self.data.index)
        
        # Initialize indicators with the Pandas Series
        self.rsi = self.I(RSIIndicator(close_series, window=self.rsi_window), name='rsi')
        self.macd = self.I(MACD(close_series, window_slow=self.macd_window_slow, window_fast=self.macd_window_fast, window_sign=self.macd_window_sign), name='macd')

    def next(self):
        if self.rsi > self.rsi_threshold and self.macd.macd() > self.macd.macd_signal():
            self.buy()
        elif self.rsi < self.rsi_threshold and self.macd.macd() < self.macd.macd_signal():
            self.sell()

bt = Backtest(all_data, RSI_MACD_Strategy, cash=initial_cash)

# Run the backtest
output = bt.run()

# Print the results
print(output)

bt.plot()