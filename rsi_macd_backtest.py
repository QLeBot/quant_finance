import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from alpaca.data import StockHistoricalDataClient, TimeFrame

# Alpaca API credentials
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Symbols to test (including ETF proxies for indices)
symbols = ["NVDA", "SPY", "FEZ", "EWQ", "MSFT", "META", "PLTR"]

# Backtest parameters
initial_cash = 10000
results = {}

for symbol in symbols:
    try:
        # Fetch data
        bars = stock_client.get_bars(symbol, TimeFrame.Day, limit=300).df
        df = bars[bars['symbol'] == symbol].copy()
        if df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        # Compute indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # Buy/Sell signals
        df['buy_signal'] = (
            (df['rsi'] > 30) & (df['rsi'].shift(1) <= 30) &
            (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        df['sell_signal'] = (
            (df['rsi'] < 70) & (df['rsi'].shift(1) >= 70) &
            (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )

        # Backtesting logic
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

        # Store result
        final_value = df['portfolio_value'].iloc[-1]
        profit = final_value - initial_cash
        results[symbol] = {
            'Final Value': round(final_value, 2),
            'Profit': round(profit, 2),
            'ROI (%)': round((profit / initial_cash) * 100, 2)
        }

        # Plot (optional: one plot per symbol)
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['close'], label=f'{symbol} Price')
        plt.plot(df.index, df['portfolio_value'] / 10, '--', label='Portfolio (scaled)')
        plt.scatter(df.index[df['buy_signal']], df['close'][df['buy_signal']], marker='^', color='green', label='Buy')
        plt.scatter(df.index[df['sell_signal']], df['close'][df['sell_signal']], marker='v', color='red', label='Sell')
        plt.title(f"{symbol} RSI + MACD Strategy Backtest")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

# Show summary
print("\n📊 Backtest Summary:")
summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv('backtest_summary.csv')