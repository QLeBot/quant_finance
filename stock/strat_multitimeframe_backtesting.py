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

# Compute indicators for all timeframes
all_data_1h = compute_indicators(all_data_1h)
all_data_4h = compute_indicators(all_data_4h)
all_data_day = compute_indicators(all_data_day)
all_data_week = compute_indicators(all_data_week)

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
        
        # Convert Close prices to pandas Series for indicators
        close_series = pd.Series(self.data.Close, index=self.data.index)
        
        # Initialize indicators with pandas Series
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x), window=10).rsi(), close_series, name='rsi')
        self.macd = self.I(lambda x: MACD(pd.Series(x)).macd(), close_series, name='macd')
        self.macd_signal = self.I(lambda x: MACD(pd.Series(x)).macd_signal(), close_series, name='macd_signal')
        self.macd_hist = self.I(lambda x: MACD(pd.Series(x)).macd_diff(), close_series, name='macd_hist')
        
        # Prepare higher timeframe data with proper column names
        data_4h = all_data_4h[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']].copy()
        data_4h.columns = ['timestamp', 'rsi_4h', 'macd_4h', 'macd_signal_4h', 'macd_hist_4h']
        
        data_day = all_data_day[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']].copy()
        data_day.columns = ['timestamp', 'rsi_day', 'macd_day', 'macd_signal_day', 'macd_hist_day']
        
        data_week = all_data_week[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']].copy()
        data_week.columns = ['timestamp', 'rsi_week', 'macd_week', 'macd_signal_week', 'macd_hist_week']
        
        # Merge higher timeframe data
        self.data_4h = pd.merge_asof(
            df_1h.sort_values('timestamp'),
            data_4h,
            on='timestamp',
            direction='backward'
        )
        
        self.data_day = pd.merge_asof(
            self.data_4h.sort_values('timestamp'),
            data_day,
            on='timestamp',
            direction='backward'
        )
        
        self.data_week = pd.merge_asof(
            self.data_day.sort_values('timestamp'),
            data_week,
            on='timestamp',
            direction='backward'
        )
        
        # Set the index back
        self.data_week.set_index('timestamp', inplace=True)

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
    """    
    def next(self):
        # Get current index
        current_idx = len(self.data) - 1
        
        # Get current and previous RSI values
        current_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2] if len(self.rsi) > 1 else current_rsi
        
        # Get current and previous MACD values
        current_macd = self.macd[-1]
        current_macd_signal = self.macd_signal[-1]
        current_macd_hist = self.macd_hist[-1]
        prev_macd_hist = self.macd_hist[-2] if len(self.macd_hist) > 1 else current_macd_hist
        
        # Check RSI and MACD conditions for 1h timeframe
        rsi_cross_1h = (current_rsi > self.rsi_threshold) and (prev_rsi <= self.rsi_threshold)
        macd_cross_1h = (current_macd > current_macd_signal) and (current_macd_hist > prev_macd_hist)
        macd_hist_rising_1h = (current_macd_hist > prev_macd_hist)

        rsi_drop_1h = (current_rsi < (100 - self.rsi_threshold)) and (prev_rsi >= (100 - self.rsi_threshold))
        macd_drop_1h = (current_macd < current_macd_signal) and (current_macd_hist < prev_macd_hist)
        macd_hist_falling_1h = (current_macd_hist < prev_macd_hist)
        
        # Get current and previous values for higher timeframes
        current_4h = self.data_4h.iloc[current_idx] if current_idx < len(self.data_4h) else None
        prev_4h = self.data_4h.iloc[current_idx-1] if current_idx > 0 and current_idx < len(self.data_4h) else None
        
        current_day = self.data_day.iloc[current_idx] if current_idx < len(self.data_day) else None
        prev_day = self.data_day.iloc[current_idx-1] if current_idx > 0 and current_idx < len(self.data_day) else None
        
        current_week = self.data_week.iloc[current_idx] if current_idx < len(self.data_week) else None
        prev_week = self.data_week.iloc[current_idx-1] if current_idx > 0 and current_idx < len(self.data_week) else None
        
        # Check conditions for higher timeframes
        rsi_cross_4h = (current_4h['rsi_4h'] > self.rsi_threshold) if current_4h is not None else False
        macd_cross_4h = (current_4h['macd_4h'] > current_4h['macd_signal_4h']) if current_4h is not None else False
        macd_hist_rising_4h = (current_4h['macd_hist_4h'] > prev_4h['macd_hist_4h']) if current_4h is not None and prev_4h is not None else False

        rsi_drop_4h = (current_4h['rsi_4h'] < (100 - self.rsi_threshold)) if current_4h is not None else False
        macd_drop_4h = (current_4h['macd_4h'] < current_4h['macd_signal_4h']) if current_4h is not None else False
        macd_hist_falling_4h = (current_4h['macd_hist_4h'] < prev_4h['macd_hist_4h']) if current_4h is not None and prev_4h is not None else False
        
        rsi_cross_day = (current_day['rsi_day'] > self.rsi_threshold) if current_day is not None else False
        macd_cross_day = (current_day['macd_day'] > current_day['macd_signal_day']) if current_day is not None else False
        macd_hist_rising_day = (current_day['macd_hist_day'] > prev_day['macd_hist_day']) if current_day is not None and prev_day is not None else False

        rsi_drop_day = (current_day['rsi_day'] < (100 - self.rsi_threshold)) if current_day is not None else False
        macd_drop_day = (current_day['macd_day'] < current_day['macd_signal_day']) if current_day is not None else False
        macd_hist_falling_day = (current_day['macd_hist_day'] < prev_day['macd_hist_day']) if current_day is not None and prev_day is not None else False
        
        rsi_cross_week = (current_week['rsi_week'] > self.rsi_threshold) if current_week is not None else False
        macd_cross_week = (current_week['macd_week'] > current_week['macd_signal_week']) if current_week is not None else False
        macd_hist_rising_week = (current_week['macd_hist_week'] > prev_week['macd_hist_week']) if current_week is not None and prev_week is not None else False

        rsi_drop_week = (current_week['rsi_week'] < (100 - self.rsi_threshold)) if current_week is not None else False
        macd_drop_week = (current_week['macd_week'] < current_week['macd_signal_week']) if current_week is not None else False
        macd_hist_falling_week = (current_week['macd_hist_week'] < prev_week['macd_hist_week']) if current_week is not None and prev_week is not None else False
        
        # Buy / Sell signals
        # V1 better than Buy&Hold if hedging and exclusive orders are set to default
        #buy_signal = (rsi_cross_1h and macd_cross_1h) or (rsi_cross_4h and macd_cross_4h) or (rsi_cross_day and macd_cross_day) or (rsi_cross_week and macd_cross_week)
        #sell_signal = ((rsi_drop_1h and macd_drop_1h) and (rsi_drop_4h and macd_drop_4h) and (rsi_drop_day and macd_drop_day) and (rsi_drop_week and macd_drop_week))

        # V2 shit
        #buy_signal = (rsi_cross_1h or macd_cross_1h) and (rsi_cross_4h or macd_cross_4h) and (rsi_cross_day or macd_cross_day) and (rsi_cross_week or macd_cross_week)
        #sell_signal = ((rsi_drop_1h or macd_drop_1h) and (rsi_drop_4h or macd_drop_4h) and (rsi_drop_day or macd_drop_day) and (rsi_drop_week or macd_drop_week))

        # V3 meh 1 trade basically follows buy and hold with hedging
        #buy_signal = (rsi_cross_1h and macd_cross_1h) or (rsi_cross_4h and macd_cross_4h) or (rsi_cross_day and macd_cross_day) or (rsi_cross_week and macd_cross_week)
        #sell_signal = ((rsi_drop_1h or macd_drop_1h) or (rsi_drop_4h or macd_drop_4h) and (rsi_drop_day and macd_drop_day) and (rsi_drop_week and macd_drop_week))

        # V4 
        buy_signal = (rsi_cross_1h and macd_cross_1h) or (rsi_cross_4h and macd_cross_4h) or (rsi_cross_day and macd_cross_day) or (rsi_cross_week and macd_cross_week)
        sell_signal = ((rsi_drop_1h and macd_drop_1h) or (rsi_drop_4h and macd_drop_4h) and (rsi_drop_day and macd_drop_day))

        # Buy signal: All timeframes confirm
        if buy_signal:
            if not self.position:
                self.buy()
        
        # Sell signal: Any timeframe shows weakness
        elif sell_signal:
            self.sell()
    """
    def next(self):
        idx = len(self.data) - 1

        # 1H timeframe (from primary data)
        rsi_1h, macd_1h, macd_signal_1h, macd_hist_1h = self.rsi[-1], self.macd[-1], self.macd_signal[-1], self.macd_hist[-1]
        long_1h = (rsi_1h > self.rsi_threshold) and (macd_1h > macd_signal_1h)
        short_1h = (rsi_1h < (100 - self.rsi_threshold)) and (macd_1h < macd_signal_1h)

        # Higher timeframes
        long_4h, short_4h = self.signal_confirmation(self.data_4h.iloc[idx] if idx < len(self.data_4h) else None,
                                                     'rsi_4h', 'macd_4h', 'macd_signal_4h', 'macd_hist_4h', self.rsi_threshold)
        long_day, short_day = self.signal_confirmation(self.data_day.iloc[idx] if idx < len(self.data_day) else None,
                                                       'rsi_day', 'macd_day', 'macd_signal_day', 'macd_hist_day', self.rsi_threshold)
        long_week, short_week = self.signal_confirmation(self.data_week.iloc[idx] if idx < len(self.data_week) else None,
                                                         'rsi_week', 'macd_week', 'macd_signal_week', 'macd_hist_week', self.rsi_threshold)

        # Consolidated signals
        # Consistent curve that minimizes drawdowns but underperforms buy and hold
        long_confirmed = long_1h and long_4h and long_day and long_week
        short_confirmed = short_1h and short_4h and short_day and short_week

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
bt = Backtest(data_1h, MultiTimeframeRSIMACDStrategy, cash=initial_cash, commission=.002, finalize_trades=True)
results = bt.run()
print(results)

# Plot results
bt.plot(plot_equity=True, plot_return=True, plot_volume=True, plot_pl=True, plot_trades=True, show_legend=True, resample=True)