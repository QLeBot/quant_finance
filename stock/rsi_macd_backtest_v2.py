import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

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
symbols = ["NVDA", "SPY", "FEZ", "EWQ", "MSFT", "META", "PLTR"]
initial_cash = 10000
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

# Request all data once
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date
)
all_data = stock_client.get_stock_bars(request_params).df
all_data = all_data.reset_index()  # bring 'symbol' and 'timestamp' into columns

# Function to compute performance metrics
def compute_metrics(df, initial_cash):
    final_value = df['portfolio_value'].iloc[-1]
    profit = final_value - initial_cash
    roi = (profit / initial_cash) * 100

    returns = df['portfolio_value'].pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else np.nan

    running_max = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    trades = df[df['buy_signal'] | df['sell_signal']]
    num_trades = len(trades) // 2  # Buy+Sell = 1 round trip

    profits = []
    pos = 0
    buy_price = 0

    for i in range(len(df)):
        if df['buy_signal'].iloc[i] and pos == 0:
            buy_price = df['close'].iloc[i]
            pos = 1
        elif df['sell_signal'].iloc[i] and pos == 1:
            sell_price = df['close'].iloc[i]
            profits.append(sell_price - buy_price)
            pos = 0

    win_rate = (np.sum(np.array(profits) > 0) / len(profits) * 100) if profits else np.nan

    # CAGR calculation
    days_held = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    years = days_held / 365.25
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    return {
        'Final Value': round(final_value, 2),
        'Profit': round(profit, 2),
        'ROI (%)': round(roi, 2),
        'CAGR (%)': round(cagr, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Number of Trades': num_trades,
        'Win Rate (%)': round(win_rate, 2) if not np.isnan(win_rate) else 'N/A'
    }

# Backtest
results = {}

for symbol in symbols:
    try:
        df = all_data[all_data['symbol'] == symbol].copy()
        if df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        # Trend filter: 200-day Moving Average
        df['ma200'] = df['close'].rolling(window=200).mean()
        trend_filter = df['close'] > df['ma200']

        # Volume filter: Check if volume is higher than average (20-day)
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        volume_filter = df['volume'] > df['avg_volume']

        # RSI + MACD indicators
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        macd_hist = macd.macd_diff()  # MACD Histogram
        
        # Print recent RSI and MACD values for debugging
        print(f"Last 10 RSI and MACD values for {symbol}:")
        print(df[['timestamp', 'close', 'rsi', 'macd', 'macd_signal']].tail(10))
        
        # Buy signal: RSI > 30, MACD bullish cross, trend filter, volume filter, MACD histogram rising
        recent_rsi_cross = (df['rsi'] > 30) & (df['rsi'].shift(1) <= 30)
        recent_macd_cross = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_hist_rising = macd_hist > macd_hist.shift(1)
        df['buy_signal'] = (
            recent_rsi_cross |  # RSI cross above 30
            recent_macd_cross |                # MACD cross above signal
            macd_hist_rising &                                # MACD histogram rising
            trend_filter & volume_filter                      # Trend & Volume filter
        )

        # Sell signal: RSI < 70, MACD bearish cross, MACD histogram weakening
        recent_rsi_drop = (df['rsi'] < 70) & (df['rsi'].shift(1) >= 70)
        recent_macd_drop = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        macd_hist_falling = macd_hist < macd_hist.shift(1)
        df['sell_signal'] = (
            recent_rsi_drop |  # RSI cross below 70
            recent_macd_drop |                # MACD cross below signal
            macd_hist_falling                                 # MACD histogram falling
        )

        # Strategy simulation
        cash = initial_cash
        shares = 0
        position = 0
        portfolio_values = []

        for i in range(len(df)):
            price = df['close'].iloc[i]
            if df['buy_signal'].iloc[i] and position == 0:
                shares = cash // price
                cash -= shares * price
                position = 1
            elif df['sell_signal'].iloc[i] and position == 1:
                cash += shares * price
                shares = 0
                position = 0
            total_value = cash + shares * price
            portfolio_values.append(total_value)

        df['portfolio_value'] = portfolio_values

        # Benchmark: Buy & Hold
        df['benchmark_value'] = initial_cash * (df['close'] / df['close'].iloc[0])

        # Save metrics
        results[symbol] = compute_metrics(df, initial_cash)

        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(df['timestamp'], df['portfolio_value'], label='Strategy')
        plt.plot(df['timestamp'], df['benchmark_value'], label='Buy & Hold')
        plt.scatter(df['timestamp'][df['buy_signal']], df['close'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(df['timestamp'][df['sell_signal']], df['close'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.title(f"{symbol} Strategy vs Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

# Results Summary
print("\n📊 Backtest Summary:")
summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv("stock/summary_rsi_macd_v2.csv")
