import truststore
truststore.inject_into_ssl()

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

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

# Parameters
symbols = ["NVDA"]
initial_cash = 10000
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime.now() - datetime.timedelta(days=10)

# Request data for different timeframes
request_params_1h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_1h = stock_client.get_stock_bars(request_params_1h).df
all_data_1h = all_data_1h.reset_index()

request_params_4h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_4h = stock_client.get_stock_bars(request_params_4h).df
all_data_4h = all_data_4h.reset_index()

request_params_day = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_day = stock_client.get_stock_bars(request_params_day).df
all_data_day = all_data_day.reset_index()

request_params_week = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Week,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_week = stock_client.get_stock_bars(request_params_week).df
all_data_week = all_data_week.reset_index()

# Convert timestamps to datetime
all_data_1h['timestamp'] = pd.to_datetime(all_data_1h['timestamp'])
all_data_4h['timestamp'] = pd.to_datetime(all_data_4h['timestamp'])
all_data_day['timestamp'] = pd.to_datetime(all_data_day['timestamp'])
all_data_week['timestamp'] = pd.to_datetime(all_data_week['timestamp'])

def get_risk_free_rate_tnx():
    """Fetches the latest U.S. 10-Year Treasury yield using yfinance."""
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100
        return latest_yield
    else:
        raise ValueError("No data found for ^TNX.")

def compute_indicators(df):
    """Compute technical indicators for a given dataframe."""
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = RSIIndicator(df['close'], window=10).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    return df

# Compute indicators for all timeframes
all_data_1h = compute_indicators(all_data_1h)
all_data_4h = compute_indicators(all_data_4h)
all_data_day = compute_indicators(all_data_day)
all_data_week = compute_indicators(all_data_week)

class MultiTimeframeRSIMACDStrategy(Strategy):
    rsi_threshold = 30
    
    def init(self):
        # Convert the backtesting._util._Array to a Pandas Series
        close_series = pd.Series(self.data.Close, index=self.data.index)
        
        # Initialize indicators with the Pandas Series
        self.rsi = self.I(RSIIndicator(close_series, window=10), name='rsi')
        self.macd = self.I(MACD(close_series), name='macd')
        
        # Merge higher timeframe data
        self.data_4h = pd.merge_asof(
            self.data.reset_index().sort_values('timestamp'),
            all_data_4h[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']],
            on='timestamp',
            direction='backward',
            suffixes=('', '_4h')
        )
        
        self.data_day = pd.merge_asof(
            self.data_4h.sort_values('timestamp'),
            all_data_day[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']],
            on='timestamp',
            direction='backward',
            suffixes=('', '_day')
        )
        
        self.data_week = pd.merge_asof(
            self.data_day.sort_values('timestamp'),
            all_data_week[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']],
            on='timestamp',
            direction='backward',
            suffixes=('', '_week')
        )
        
        # Set the index back
        self.data_week.set_index('timestamp', inplace=True)
        
    def next(self):
        # Get current index
        current_idx = len(self.data) - 1
        
        # Check RSI and MACD conditions for all timeframes
        rsi_cross_1h = (self.rsi > self.rsi_threshold) & (self.rsi.shift(1) <= self.rsi_threshold)
        macd_cross_1h = (self.macd.macd() > self.macd.macd_signal()) & (self.macd.macd_hist() > self.macd.macd_hist().shift(1))
        
        rsi_cross_4h = (self.data_4h['rsi_4h'] > self.rsi_threshold) & (self.data_4h['rsi_4h'].shift(1) <= self.rsi_threshold)
        macd_cross_4h = (self.data_4h['macd_4h'] > self.data_4h['macd_signal_4h']) & (self.data_4h['macd_hist_4h'] > self.data_4h['macd_hist_4h'].shift(1))
        
        rsi_cross_day = (self.data_day['rsi_day'] > self.rsi_threshold) & (self.data_day['rsi_day'].shift(1) <= self.rsi_threshold)
        macd_cross_day = (self.data_day['macd_day'] > self.data_day['macd_signal_day']) & (self.data_day['macd_hist_day'] > self.data_day['macd_hist_day'].shift(1))
        
        rsi_cross_week = (self.data_week['rsi_week'] > self.rsi_threshold) & (self.data_week['rsi_week'].shift(1) <= self.rsi_threshold)
        macd_cross_week = (self.data_week['macd_week'] > self.data_week['macd_signal_week']) & (self.data_week['macd_hist_week'] > self.data_week['macd_hist_week'].shift(1))
        
        # Buy signal: All timeframes confirm
        if (rsi_cross_1h | macd_cross_1h) & (rsi_cross_4h | macd_cross_4h) & (rsi_cross_day | macd_cross_day) & (rsi_cross_week | macd_cross_week):
            if not self.position:
                self.buy()
        
        # Sell signal: Any timeframe shows weakness
        elif self.position and ((self.rsi < (100 - self.rsi_threshold)) | (self.macd.macd() < self.macd.macd_signal())):
            self.sell()

# Prepare data for backtesting
data_1h = all_data_1h.copy()
data_1h.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data_1h.set_index('timestamp', inplace=True)

# Run backtest
bt = Backtest(data_1h, MultiTimeframeRSIMACDStrategy, cash=initial_cash, commission=.002)
results = bt.run()
print(results)

# Plot results
bt.plot()