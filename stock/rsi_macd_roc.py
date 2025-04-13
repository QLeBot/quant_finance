"""
RSI : Relative Strength Index
MACD : Moving Average Convergence Divergence

RSI is a momentum oscillator that measures the speed and change of price movements. 
It helps to identify overbought or oversold conditions.
Range is between 0 and 100.
Above 70 is overbought (sell signal), below 30 is oversold (buy signal).
Default period is 14 for the lookback period (14 days, 14 hours, 14 minutes, etc.)

MACD is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a security's price.
Default parameters are :
MACD Line = 12-period EMA - 26-period EMA
Signal Line = 9-period EMA of MACD Line
MACD Histogram = MACD Line - Signal Line

When MACD Line crosses above Signal Line, it is a buy signal.
When MACD Line crosses below Signal Line, it is a sell signal.
Histogram visualizes the difference between the MACD Line and the Signal Line.

ROC : Rate of Change
ROC is a momentum oscillator that measures the percentage change in price between the current price and the price a certain number of periods ago.
ROC = ((Current Price - Price n periods ago) / Price n periods ago ) * 100
ROC shows how fast the price is changing and in what direction.
Above 0 price is rising (buy signal), below 0 price is falling (sell signal).
Crossing 0 is often used as a buy or sell signal.
ROC is very sensitive to sharp price changes — good for detecting breakouts.
Best used with a baseline (zero line) or combined with other indicators (like RSI, MACD) to reduce false signals.
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pytz

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

"""
Observations:
- TSLA seems to work in this strategy
- The rest of the stocks are not performing well, either underperforming the buy&hold or loosing money

Tickers:
- ATO for Atmos.
- ATOS for Atos, overall downtrend from 2017 to 2024
- MC.PA for Moet Hennessy Louis Vuitton, overall uptrend to mid 2021 then stable with big swings
- NVDA for Nvidia, overall uptrend, big uptrend mid 2022 to end 2024. End 2024 seems to be the top
- SPY for S&P 500, overall uptrend can be used as a benchmark for overall market
- TSLA for Tesla, slow uptrend 2013 to mid 2020 then big uptrend from mid 2020 to mid/end 2021 then overall stable with big swings

"""

# Parameters
#symbols = ["NVDA", "SPY", "TSLA", "MC.PA", "ATO", "ATOS"]
#symbols = ["NVDA"]
#symbols = ["NVDA", "ATOS"]
symbols = ["NVDA", "ATOS" , "ATO"]
#symbols = ["SPY"]
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
# Convert the 'timestamp' column to datetime if not already
all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])

# Adjust prices and volumes for stock splits for each symbol
for symbol, splits in stock_splits.items():
    for split_date, split_ratio in splits.items():
        # Convert split_date to timezone-aware (UTC)
        split_date = pd.to_datetime(split_date).tz_localize('UTC')
        
        mask = (all_data['symbol'] == symbol) & (all_data['timestamp'] < split_date)
        all_data.loc[mask, 'close'] /= split_ratio
        all_data.loc[mask, 'volume'] *= split_ratio

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

        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values in each column for {symbol}:")
        print(missing_values)

        # Check for outliers in the 'close' prices
        # Using a simple statistical method to identify outliers
        q1 = df['close'].quantile(0.25)
        q3 = df['close'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df['close'] < (q1 - 1.5 * iqr)) | (df['close'] > (q3 + 1.5 * iqr))]
        print(f"Number of outliers in 'close' prices for {symbol}: {len(outliers)}")

        # Check data range
        print(f"Data range from {df['timestamp'].min()} to {df['timestamp'].max()} for {symbol}")

        # Visual inspection of 'close' prices
        """
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['close'], label='Close Price')
        plt.title(f"{symbol} Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid()
        plt.show()
        """

        # Trend filter: 200-day Moving Average
        df['ma200'] = df['close'].rolling(window=200).mean()
        trend_filter = df['close'] > df['ma200']

        # Volume filter: Check if volume is higher than average (20-day)
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        volume_filter = df['volume'] > df['avg_volume']

        # RSI + MACD indicators
        df['rsi'] = RSIIndicator(df['close'], window=10).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        macd_hist = macd.macd_diff()  # MACD Histogram

        # Add ROC (Rate of Change) as a trend strength filter
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100  # 10-day ROC
        
        # Print recent RSI and MACD values for debugging
        print(f"Last 10 RSI and MACD values for {symbol}:")
        print(df[['timestamp', 'close', 'rsi', 'macd', 'macd_signal']].tail(10))

        rsi_threshold = 30
        
        # Buy signal: RSI > 30, MACD bullish cross, trend filter, volume filter, MACD histogram rising
        recent_rsi_cross = (df['rsi'] > rsi_threshold) & (df['rsi'].shift(1) <= rsi_threshold)
        recent_macd_cross = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_hist_rising = macd_hist > macd_hist.shift(1)

        # Strategy 
        #buy_strategy = (recent_rsi_cross & (recent_macd_cross & macd_hist_rising) & trend_filter & volume_filter & df['roc'] > 0)
        #buy_strategy = ((recent_macd_cross | macd_hist_rising) & recent_rsi_cross & df['roc'] > 0) # good for ATOS
        buy_strategy = ((recent_macd_cross | macd_hist_rising) & recent_rsi_cross & df['roc'] > 0 & trend_filter & volume_filter)
        
        df['buy_signal'] = buy_strategy

        # Sell signal: RSI < 70, MACD bearish cross, MACD histogram weakening
        recent_rsi_drop = (df['rsi'] < (100 - rsi_threshold)) & (df['rsi'].shift(1) >= (100 - rsi_threshold))
        recent_macd_drop = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        macd_hist_falling = macd_hist < macd_hist.shift(1)

        """
        ROC < 0 never produces good results on the sell side
        """
        #sell_strategy = ((recent_macd_drop | macd_hist_falling) & recent_rsi_drop) # good for ATOS
        sell_strategy = ((recent_macd_drop | macd_hist_falling) & recent_rsi_drop & trend_filter & volume_filter)
        #sell_strategy = ((recent_macd_drop & macd_hist_falling) & recent_rsi_drop)  # follows curve for ATOS but with higer returns   
        #sell_strategy = ((recent_macd_drop & macd_hist_falling) & recent_rsi_drop & trend_filter & volume_filter)
        
        df['sell_signal'] = sell_strategy

        # Define total capital and position size percentage
        #position_size_percentage = 0.1  # Invest 10% of total capital in each trade

        # Strategy simulation with position sizing
        cash = initial_cash
        shares = 0
        position = 0
        portfolio_values = []

        for i in range(len(df)):
            price = df['close'].iloc[i]
            if df['buy_signal'].iloc[i] and position == 0:
                shares = cash // price
                # Calculate position size
                #position_size = cash * position_size_percentage
                #shares = position_size // price
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

        
        # Plot Portfolio Value, Benchmark and buy/sell signals
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
        

        # Plot RSI
        plt.figure(figsize=(12, 5))
        plt.plot(df['timestamp'], df['rsi'], label='RSI')
        plt.axhline(rsi_threshold, color='red', linestyle='--', label=f'RSI {rsi_threshold}')
        plt.axhline((100 - rsi_threshold), color='green', linestyle='--', label=f'RSI {100 - rsi_threshold}')
        #plt.scatter(df['timestamp'][df['buy_signal']], df['close'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        #plt.scatter(df['timestamp'][df['sell_signal']], df['close'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.scatter(df['timestamp'][df['buy_signal']], df['rsi'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(df['timestamp'][df['sell_signal']], df['rsi'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.title(f"{symbol} RSI")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        

        # Plot MACD
        plt.figure(figsize=(12, 5))
        plt.plot(df['timestamp'], df['macd'], label='MACD Line')
        plt.plot(df['timestamp'], df['macd_signal'], label='Signal Line')
        plt.bar(df['timestamp'], macd_hist, label='MACD Histogram', color='grey', alpha=0.3)
        # Plot on close price
        #plt.scatter(df['timestamp'][df['buy_signal']], df['close'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        #plt.scatter(df['timestamp'][df['sell_signal']], df['close'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        # Plot on MACD lines
        plt.scatter(df['timestamp'][df['buy_signal']], df['macd'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(df['timestamp'][df['sell_signal']], df['macd'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.title(f"{symbol} MACD")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        """
        # Plot Portfolio Value, Benchmark, RSI, and MACD on the same plot
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot Portfolio and Benchmark
        ax1.plot(df['timestamp'], df['portfolio_value'], label='Strategy', color='blue')
        ax1.plot(df['timestamp'], df['benchmark_value'], label='Buy & Hold', color='orange')
        ax1.scatter(df['timestamp'][df['buy_signal']], df['close'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        ax1.scatter(df['timestamp'][df['sell_signal']], df['close'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend(loc='upper left')
        ax1.grid()

        # Create a second y-axis for RSI
        ax2 = ax1.twinx()
        ax2.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', alpha=0.5)
        ax2.axhline(30, color='red', linestyle='--', label='RSI 30')
        ax2.axhline(70, color='green', linestyle='--', label='RSI 70')
        ax2.set_ylabel("RSI")
        ax2.legend(loc='upper right')

        # Create a third y-axis for MACD
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        ax3.plot(df['timestamp'], df['macd'], label='MACD Line', color='cyan', alpha=0.5)
        ax3.plot(df['timestamp'], df['macd_signal'], label='Signal Line', color='magenta', alpha=0.5)
        ax3.bar(df['timestamp'], macd_hist, label='MACD Histogram', color='grey', alpha=0.3)
        ax3.set_ylabel("MACD")
        ax3.legend(loc='lower right')

        plt.title(f"{symbol} Strategy, RSI, and MACD")
        plt.tight_layout()
        plt.show()
        """

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

# Results Summary
print("\n📊 Backtest Summary:")
summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv("stock/csv/summary_rsi_macd_roc.csv")

"""
TO DO:
- Better understand the RSI and MACD indicators
- Keep and save the RSI, MACD and ROC values for each stock

"""
