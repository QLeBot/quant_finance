import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score
import xgboost as xgb

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
start_date = datetime.datetime(2013, 1, 1)
#end_date = datetime.datetime.now() - datetime.timedelta(days=10)
end_date = datetime.datetime(2024, 12, 31)

def get_data(symbol, timeframe):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="split"
    )
    return stock_client.get_stock_bars(request_params).df

df = get_data(symbols, TimeFrame.Hour)

# Convert timestamps to datetime and reset index
df = df.reset_index()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

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

#timeframes = [all_data_1h]
#all_data = [compute_indicators(df) for df in timeframes]
#all_data_1h = all_data

# ======= ML Model =======
# Feature engineering
df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd_diff()

for lag in range(1, 6):
    df[f'Close_lag_{lag}'] = df['close'].shift(lag)

# Future return over 1 hour (percentage)
df['return_1h'] = df['close'].pct_change().shift(-1)

df.dropna(inplace=True)

# Binary classification target: 1 for up, 0 for down
df['up_down'] = (df['return_1h'] > 0).astype(int)

# Split and scale
train_end = '2023-12-31'
train_df = df.loc[df.index < train_end]
test_df = df.loc[df.index >= train_end]

feature_cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD',
                'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4', 'Close_lag_5']
#target_col = 'close'
target_col = 'return_1h'

# Split data into training and test sets
X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Binary classification target: 1 for up, 0 for down
y_train_cls = train_df['up_down']
y_test_cls = test_df['up_down']

# Train XGBoost model
regressor = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
regressor.fit(X_train_scaled, y_train)

classifier = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
classifier.fit(X_train_scaled, y_train_cls)

# Predict returns
predicted_return = regressor.predict(X_test_scaled)
test_df['Predicted_Return'] = predicted_return

# Predict up/down
predicted_up_down = classifier.predict(X_test_scaled)
test_df['Predicted_UpDown'] = predicted_up_down

# Evaluate the model
mse = mean_squared_error(y_test, predicted_return)
mae = mean_absolute_error(y_test, predicted_return)
r2 = r2_score(y_test, predicted_return)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Optional: check accuracy
print(classification_report(y_test_cls, test_df['Predicted_UpDown']))

# Make it compatible with backtesting.py
#bt_df = test_df[['open', 'high', 'low', 'close']].copy()
#bt_df.columns = ['Open', 'High', 'Low', 'Close']

bt_df = test_df.copy()

# Select only required columns and reset index (Backtest needs datetime index)
bt_df = bt_df[['open', 'high', 'low', 'close', 'volume', 'Predicted_Return', 'Predicted_UpDown']]
bt_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Predicted_Return', 'Predicted_UpDown']

bt_df.index = pd.to_datetime(bt_df.index)

# Inject model predictions
bt_df['Predicted_Return'] = test_df['Predicted_Return']

bt_df['Predicted_UpDown'] = test_df['Predicted_UpDown']



class MLBasedStrategy(Strategy):
    def init(self):
        self.predicted_return = self.data.Predicted_Return
        self.predicted_up_down = self.data.Predicted_UpDown
    def next(self):
        price_now = self.data.Close[-1]
        predicted_return = self.predicted_return[-1]
        predicted_direction = self.predicted_up_down[-1]

        threshold_long = 0.02  # 2% expected return
        threshold_short = -0.05  # -5% expected return

        # Open long position
        if not self.position:
            if predicted_direction == 1 and predicted_return > threshold_long:
                self.buy()
            elif predicted_direction == 0 and predicted_return < threshold_short:
                self.sell()

        # Close positions when signal is invalidated
        if self.position:
            if (self.position.is_long and (predicted_direction == 0 or predicted_return < threshold_short)) or \
               (self.position.is_short and (predicted_direction == 1 or predicted_return > threshold_long)):
                self.position.close()
        

# Run backtest
bt = Backtest(bt_df, MLBasedStrategy, cash=initial_cash, commission=0.0035, finalize_trades=True)
results = bt.run()
print(results)

# Plot results
bt.plot(plot_equity=True, plot_return=True, plot_volume=True, plot_pl=True, plot_trades=True, show_legend=True, resample=True)