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
start_date = datetime.datetime(2017, 1, 1)
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

timeframes = [all_data_1h, all_data_4h, all_data_day, all_data_week]
all_data = [compute_indicators(df) for df in timeframes]
all_data_1h, all_data_4h, all_data_day, all_data_week = all_data

class MultiTimeframeRSIMACDStrategy(Strategy):
    rsi_threshold = 30
    allow_short = False  # Set to True if you want to allow short trades later
    
    def init(self):
        # Convert the backtesting data to a DataFrame for merging
        df_1h = pd.DataFrame({
            'timestamp': self.data.index,
            'close': self.data.Close,
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'volume': self.data.Volume
        })
        
        # Initialize indicators with pandas Series
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x, index=self.data.index), window=10).rsi().values, self.data.Close, name='rsi')
        self.macd = self.I(lambda x: MACD(pd.Series(x, index=self.data.index)).macd().values, self.data.Close, name='macd')
        self.macd_signal = self.I(lambda x: MACD(pd.Series(x, index=self.data.index)).macd_signal().values, self.data.Close, name='macd_signal')
        self.macd_hist = self.I(lambda x: MACD(pd.Series(x, index=self.data.index)).macd_diff().values, self.data.Close, name='macd_hist')
        
        # Prepare higher timeframe data
        data_4h = all_data_4h.rename(columns={
            'rsi': 'rsi_4h', 'macd': 'macd_4h', 'macd_signal': 'macd_signal_4h', 'macd_hist': 'macd_hist_4h'
        })

        data_day = all_data_day.rename(columns={
            'rsi': 'rsi_day', 'macd': 'macd_day', 'macd_signal': 'macd_signal_day', 'macd_hist': 'macd_hist_day'
        })

        data_week = all_data_week.rename(columns={
            'rsi': 'rsi_week', 'macd': 'macd_week', 'macd_signal': 'macd_signal_week', 'macd_hist': 'macd_hist_week'
        })
        
        # Merge higher timeframes into one DataFrame
        self.multi_tf_data = pd.merge_asof(
            pd.merge_asof(
                pd.merge_asof(
                    df_1h.sort_values('timestamp'),
                    data_4h.sort_values('timestamp'),
                    on='timestamp',
                    direction='backward',
                    suffixes=('_1h', '_4h')
                ),
                data_day.sort_values('timestamp'),
                on='timestamp',
                direction='backward',
                suffixes=('', '_day')
            ),
            data_week.sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            suffixes=('', '_week')
        )
        
        # Set timestamp as index again
        self.multi_tf_data.set_index('timestamp', inplace=True)
        
        # Print column names for debugging
        print("Available columns in multi_tf_data:")
        print(self.multi_tf_data.columns.tolist())

    def signal_confirmation(self, tf_data, rsi_key, macd_key, macd_signal_key, macd_hist_key, rsi_threshold):
        """Returns a tuple: (long_signal, short_signal) for a given timeframe."""
        if tf_data is None:
            return False, False

        rsi = tf_data[rsi_key]
        macd = tf_data[macd_key]
        macd_signal = tf_data[macd_signal_key]
        macd_hist = tf_data[macd_hist_key]

        long_signal = (rsi > rsi_threshold) and (macd > macd_signal)
        short_signal = (rsi < (100 - rsi_threshold)) and (macd < macd_signal)

        return long_signal, short_signal
    
    def signal_confirmation_v2(self, tf_data, rsi_key, macd_key, macd_signal_key, macd_hist_key, rsi_threshold):
        """Returns a tuple: (long_signal, short_signal) for a given timeframe."""
        if tf_data is None:
            return False, False

        rsi = tf_data[rsi_key]
        macd = tf_data[macd_key]
        macd_signal = tf_data[macd_signal_key]
        macd_hist = tf_data[macd_hist_key]

        rsi_cross = (rsi > rsi_threshold & rsi.shift(1) <= (100 - rsi_threshold)) and (macd > macd_signal & macd_hist > macd_hist.shift(1))
        macd_cross = (rsi < (100 - rsi_threshold) & rsi.shift(1) >= (100 - rsi_threshold)) and (macd < macd_signal & macd_hist < macd_hist.shift(1))

        return rsi, macd, macd_signal, macd_hist, rsi_cross, macd_cross
    
    def next(self):
        idx = len(self.data) - 1
        
        # Get current row from merged data
        row = self.multi_tf_data.iloc[idx]
        
        # Get current and previous values for all timeframes
        # 1H timeframe (using the indicators we initialized)
        rsi_1h = self.rsi[-1]
        macd_1h = self.macd[-1]
        macd_signal_1h = self.macd_signal[-1]
        macd_hist_1h = self.macd_hist[-1]
        
        # 4H timeframe
        rsi_4h = row['rsi_4h']
        macd_4h = row['macd_4h']
        macd_signal_4h = row['macd_signal_4h']
        macd_hist_4h = row['macd_hist_4h']
        
        # Day timeframe
        rsi_day = row['rsi_day']
        macd_day = row['macd_day']
        macd_signal_day = row['macd_signal_day']
        macd_hist_day = row['macd_hist_day']
        
        # Week timeframe
        rsi_week = row['rsi_week']
        macd_week = row['macd_week']
        macd_signal_week = row['macd_signal_week']
        macd_hist_week = row['macd_hist_week']
        
        # Get previous values for all timeframes
        if idx > 0:
            prev_row = self.multi_tf_data.iloc[idx-1]
            
            # 1H timeframe (using the indicators we initialized)
            prev_rsi_1h = self.rsi[-2]
            prev_macd_hist_1h = self.macd_hist[-2]
            
            # 4H timeframe
            prev_rsi_4h = prev_row['rsi_4h']
            prev_macd_hist_4h = prev_row['macd_hist_4h']
            
            # Day timeframe
            prev_rsi_day = prev_row['rsi_day']
            prev_macd_hist_day = prev_row['macd_hist_day']
            
            # Week timeframe
            prev_rsi_week = prev_row['rsi_week']
            prev_macd_hist_week = prev_row['macd_hist_week']
        else:
            # If this is the first row, use current values as previous
            prev_rsi_1h = rsi_1h
            prev_macd_hist_1h = macd_hist_1h
            prev_rsi_4h = rsi_4h
            prev_macd_hist_4h = macd_hist_4h
            prev_rsi_day = rsi_day
            prev_macd_hist_day = macd_hist_day
            prev_rsi_week = rsi_week
            prev_macd_hist_week = macd_hist_week
        
        # 1H timeframe signals
        rsi_cross_1h = (rsi_1h > self.rsi_threshold) and (prev_rsi_1h <= self.rsi_threshold)
        macd_cross_1h = (macd_1h > macd_signal_1h) and (macd_hist_1h > prev_macd_hist_1h)
        macd_hist_rising_1h = (macd_hist_1h > prev_macd_hist_1h)
        
        rsi_drop_1h = (rsi_1h < (100 - self.rsi_threshold)) and (prev_rsi_1h >= (100 - self.rsi_threshold))
        macd_drop_1h = (macd_1h < macd_signal_1h) and (macd_hist_1h < prev_macd_hist_1h)
        macd_hist_falling_1h = (macd_hist_1h < prev_macd_hist_1h)
        
        # 4H timeframe signals
        rsi_cross_4h = (rsi_4h > self.rsi_threshold) and (prev_rsi_4h <= self.rsi_threshold)
        macd_cross_4h = (macd_4h > macd_signal_4h) and (macd_hist_4h > prev_macd_hist_4h)
        macd_hist_rising_4h = (macd_hist_4h > prev_macd_hist_4h)
        
        rsi_drop_4h = (rsi_4h < (100 - self.rsi_threshold)) and (prev_rsi_4h >= (100 - self.rsi_threshold))
        macd_drop_4h = (macd_4h < macd_signal_4h) and (macd_hist_4h < prev_macd_hist_4h)
        macd_hist_falling_4h = (macd_hist_4h < prev_macd_hist_4h)
        
        # Day timeframe signals
        rsi_cross_day = (rsi_day > self.rsi_threshold) and (prev_rsi_day <= self.rsi_threshold)
        macd_cross_day = (macd_day > macd_signal_day) and (macd_hist_day > prev_macd_hist_day)
        macd_hist_rising_day = (macd_hist_day > prev_macd_hist_day)
        
        rsi_drop_day = (rsi_day < (100 - self.rsi_threshold)) and (prev_rsi_day >= (100 - self.rsi_threshold))
        macd_drop_day = (macd_day < macd_signal_day) and (macd_hist_day < prev_macd_hist_day)
        macd_hist_falling_day = (macd_hist_day < prev_macd_hist_day)
        
        # Week timeframe signals
        rsi_cross_week = (rsi_week > self.rsi_threshold) and (prev_rsi_week <= self.rsi_threshold)
        macd_cross_week = (macd_week > macd_signal_week) and (macd_hist_week > prev_macd_hist_week)
        macd_hist_rising_week = (macd_hist_week > prev_macd_hist_week)
        
        rsi_drop_week = (rsi_week < (100 - self.rsi_threshold)) and (prev_rsi_week >= (100 - self.rsi_threshold))
        macd_drop_week = (macd_week < macd_signal_week) and (macd_hist_week < prev_macd_hist_week)
        macd_hist_falling_week = (macd_hist_week < prev_macd_hist_week)
        
        # Consolidated signals
        long_confirmed = (
            (rsi_cross_1h and macd_cross_1h)
            or (rsi_cross_4h and macd_cross_4h)
            or (rsi_cross_day and macd_cross_day)
            or (rsi_cross_week and macd_cross_week)
        )
        short_confirmed = (
            (rsi_drop_1h and macd_drop_1h)
            or (rsi_drop_4h and macd_drop_4h)
            or (rsi_drop_day and macd_drop_day)
            or (rsi_drop_week and macd_drop_week)
        )
        
        if long_confirmed:
            if not self.position:
                self.buy()
            elif self.allow_short and self.position.is_short:
                self.buy()  # close short and go long
                
        elif short_confirmed:
            if not self.position and self.allow_short:
                self.sell()
            elif self.position.is_long:
                self.sell()  # close long, possibly open short if allow_short is True
                
        elif self.position:
            # Exit on signal invalidation (if you want)
            if not long_confirmed and self.position.is_long:
                self.position.close()
            elif not short_confirmed and self.position.is_short:
                self.position.close()
    




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
bt = Backtest(data_1h, MultiTimeframeRSIMACDStrategy, cash=initial_cash, commission=0.0035, finalize_trades=True)
results = bt.run()
print(results)

# Plot results
bt.plot(plot_equity=True, plot_return=True, plot_volume=True, plot_pl=True, plot_trades=True, show_legend=True, resample=True)