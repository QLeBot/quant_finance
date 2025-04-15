import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from ta.momentum import RSIIndicator
from ta.trend import MACD
# Load historical stock data
def fetch_data(ticker, start="2022-01-01"):
    df = yf.download(ticker, start=start)
    df.dropna(inplace=True)
    return df

# Calculate On Balance Volume (OBV)
def calculate_obv(df):
    direction = np.sign(df['Close'].diff())
    direction.iloc[0] = 0  # handle the NaN at first row
    obv = (direction * df['Volume']).cumsum()
    df['OBV'] = obv
    return df


# Check for volume breakout (2x average volume)
def volume_breakout(df, window=20):
    avg_volume = df['Volume'].rolling(window=window).mean()
    df['Vol_Breakout'] = df['Volume'] > 2 * avg_volume
    return df

# Check for OBV divergence (OBV making lower highs while price makes higher highs)
def detect_divergence(df):
    df['Price_Change'] = df['Close'].diff()
    df['OBV_Change'] = df['OBV'].diff()
    df['Divergence'] = (df['Price_Change'] > 0) & (df['OBV_Change'] < 0)
    return df

# Final signal: Volume Breakout + Divergence
def generate_signal(df):
    df['Signal'] = (df['Vol_Breakout']) & (df['Divergence'])
    return df

# Function to calculate RSI and MACD
def calculate_rsi_macd(df):
    close = df['Close'].squeeze()
    #print(type(close))
    # Calculate RSI (Relative Strength Index) with a 14-period window
    rsi = RSIIndicator(close, window=14)
    df['RSI'] = rsi.rsi()
    
    # Calculate MACD with default parameters (12, 26, 9)
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    return df

# Custom Strategy Class for Backtesting
class LeadingIndicatorStrategy(Strategy):
    def init(self):
        self.signal = self.data.Signal  # Boolean Series
        self.rsi = self.data.RSI  # RSI values
        self.macd = self.data.MACD  # MACD line
        self.macd_signal = self.data.MACD_signal  # MACD signal line
    
    def next(self):
        if self.signal[-1] and (self.rsi[-1] < 30 or self.macd[-1] > self.macd_signal[-1]):   # if last value is True
            if not self.position:  # Only enter if no open position
                self.buy(size=1)   # Buy 1 share
        elif self.position:
            self.sell()  # Close the position

# Run Backtest
ticker = "AAPL"
df = fetch_data(ticker)
df = calculate_obv(df)
df = volume_breakout(df)
df = detect_divergence(df)
df = generate_signal(df)

# Calculate RSI and MACD
df = calculate_rsi_macd(df)

# Flatten MultiIndex columns
df.columns = [col[0] for col in df.columns]

# Keep only required columns for Backtest
price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df_bt = df[price_columns].copy()

# Attach custom signals to df_bt separately
df_bt['Signal'] = df['Signal'].values  # ensure no index mismatch
df_bt['Signal'] = df_bt['Close'] > df_bt['Close'].rolling(10).mean()
print(df_bt['Signal'].value_counts())

# Reset to a clean RangeIndex (not MultiIndex)
all_days = pd.date_range(start=df_bt.index.min(), end=df_bt.index.max(), freq='B')
df_bt = df_bt.reindex(all_days).fillna(method='ffill')
df_bt.index.freq = 'B'

# Add RSI and MACD to the dataframe for backtest
df_bt['RSI'] = df['RSI']
df_bt['MACD'] = df['MACD']
df_bt['MACD_signal'] = df['MACD_signal']

bt = Backtest(df_bt, LeadingIndicatorStrategy, cash=10000, commission=.002)
results = bt.run()
bt.plot(superimpose=False)
print(results)
