import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from alpaca.data.timeframe import TimeFrame

from backtesting import Backtest, Strategy

from ta.momentum import RSIIndicator
from ta.trend import MACD

from get_data import alpaca_data

# Parameters
symbols = ["NVDA"]
initial_cash = 10000
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime.now() - datetime.timedelta(days=10)

# Get data
data_1h = alpaca_data(symbol=symbols, start_date=start_date, end_date=end_date, timeframe=TimeFrame.Hour, ajustment='split')
data_day = alpaca_data(symbol=symbols, start_date=start_date, end_date=end_date, timeframe=TimeFrame.Day, ajustment='split')
data_week = alpaca_data(symbol=symbols, start_date=start_date, end_date=end_date, timeframe=TimeFrame.Week, ajustment='split')

data_1h.reset_index()
data_day.reset_index()
data_week.reset_index()

data_1h['timestamp'] = pd.to_datetime(data_1h['timestamp'])
data_day['timestamp'] = pd.to_datetime(data_day['timestamp'])
data_week['timestamp'] = pd.to_datetime(data_week['timestamp'])

def get_risk_free_rate_tnx():
    """
    Fetches the latest U.S. 10-Year Treasury yield using yfinance.
    This is often used as a proxy for the risk-free rate in USD.
    """
    # '^TNX' is the Yahoo Finance symbol for the 10-Year Treasury Note yield (multiplied by 100)
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100  # convert from percentage
        return latest_yield
    else:
        raise ValueError("No data found for ^TNX.")
    
# Function to compute indicators for a given dataframe
def compute_indicators(df):
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = RSIIndicator(df['close'], window=10).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    return df

data_1h = compute_indicators(data_1h)
data_day = compute_indicators(data_day)
data_week = compute_indicators(data_week)

macd_hist_1h = data_1h['macd_hist'].iloc[-1]
macd_hist_day = data_day['macd_hist'].iloc[-1]
macd_hist_week = data_week['macd_hist'].iloc[-1]


class MTF_RSI_MACD_Strategy(Strategy):
    rsi_threshold = 70
    macd_threshold = 30
    rsi_window = 14
    macd_window_slow = 26
    macd_window_fast = 12
    macd_window_sign = 9

    df_1h = data_1h[data_1h['symbol'] == symbol].copy()
    df_day = data_day[data_day['symbol'] == symbol].copy()
    df_week = data_week[data_week['symbol'] == symbol].copy()

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

bt = Backtest(data, MTF_RSI_MACD_Strategy, cash=initial_cash)

# Run the backtest
output = bt.run()

# Print the results
print(output)

bt.plot()