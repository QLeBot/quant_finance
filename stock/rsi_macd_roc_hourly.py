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
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

# Request 1-hour data for primary analysis and 4-hour data for confirmation
request_params_1h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date
)
all_data_1h = stock_client.get_stock_bars(request_params_1h).df
all_data_1h = all_data_1h.reset_index()  # bring 'symbol' and 'timestamp' into columns

request_params_4h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date
)
all_data_4h = stock_client.get_stock_bars(request_params_4h).df
all_data_4h = all_data_4h.reset_index()  # bring 'symbol' and 'timestamp' into columns

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
all_data_1h['timestamp'] = pd.to_datetime(all_data_1h['timestamp'])
all_data_4h['timestamp'] = pd.to_datetime(all_data_4h['timestamp'])

# Adjust prices and volumes for stock splits for each symbol in both dataframes
for symbol, splits in stock_splits.items():
    for split_date, split_ratio in splits.items():
        # Convert split_date to timezone-aware (UTC)
        split_date = pd.to_datetime(split_date).tz_localize('UTC')
        
        mask_1h = (all_data_1h['symbol'] == symbol) & (all_data_1h['timestamp'] < split_date)
        all_data_1h.loc[mask_1h, 'close'] /= split_ratio
        all_data_1h.loc[mask_1h, 'volume'] *= split_ratio

        mask_4h = (all_data_4h['symbol'] == symbol) & (all_data_4h['timestamp'] < split_date)
        all_data_4h.loc[mask_4h, 'close'] /= split_ratio
        all_data_4h.loc[mask_4h, 'volume'] *= split_ratio

# Function to compute indicators for a given dataframe
def compute_indicators(df):
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = RSIIndicator(df['close'], window=10).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    return df


# Compute indicators for both 1-hour and 4-hour data
#all_data_1h, macd_hist_1h = compute_indicators(all_data_1h)
#all_data_4h, macd_hist_4h = compute_indicators(all_data_4h)

all_data_1h = compute_indicators(all_data_1h)
all_data_4h = compute_indicators(all_data_4h)

macd_hist_1h = all_data_1h['macd_hist'].iloc[-1]
macd_hist_4h = all_data_4h['macd_hist'].iloc[-1]

def compute_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return df

def apply_trailing_stop(df, atr_multiplier=3):
    position = 0
    entry_price = 0
    trailing_stop = 0
    df['trailing_stop'] = np.nan
    df['sell_signal'] = False  # ensure column exists

    for i in range(1, len(df)):
        if df['buy_signal'].iloc[i] and position == 0:
            # Enter trade
            entry_price = df['close'].iloc[i]
            atr = df['ATR'].iloc[i]
            trailing_stop = entry_price - atr_multiplier * atr
            position = 1

        elif position == 1:
            # Update trailing stop if price rises
            atr = df['ATR'].iloc[i]
            new_stop = df['close'].iloc[i] - atr_multiplier * atr
            trailing_stop = max(trailing_stop, new_stop)
            df.loc[df.index[i], 'trailing_stop'] = trailing_stop

            # Check for exit condition
            if df['close'].iloc[i] <= trailing_stop:
                df.loc[df.index[i], 'sell_signal'] = True
                position = 0
                entry_price = 0
                trailing_stop = 0

    return df

#all_data_1h = compute_atr(all_data_1h)
#all_data_1h = apply_trailing_stop(all_data_1h, atr_multiplier=3)

# Function to compute performance metrics
def compute_metrics(df, initial_cash):
    final_value = df['portfolio_value'].iloc[-1]
    profit = final_value - initial_cash
    roi = (profit / initial_cash) * 100

    returns = df['portfolio_value'].pct_change().dropna()

    # Infer timeframe from median time difference
    time_diffs = pd.to_datetime(df['timestamp']).diff().dropna()
    median_diff = time_diffs.median()
    seconds_per_period = median_diff.total_seconds()

    periods_per_year = 365.25 * 24 * 3600 / seconds_per_period
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year) if np.std(returns) > 0 else np.nan

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
    time_diff = pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[0])
    years = time_diff.total_seconds() / (365.25 * 24 * 3600)
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else np.nan


    return {
        'Final Value': round(final_value, 2),
        'Profit': round(profit, 2),
        'ROI (%)': round(roi, 2),
        'CAGR (%)': round(cagr, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Number of Trades': num_trades,
        'Win Rate (%)': round(win_rate, 2) if not np.isnan(win_rate) else 'N/A',
        #'Timeframe (seconds)': int(seconds_per_period),
        #'Periods per year': round(periods_per_year, 2)
    }

# Backtest with MTF confirmation
results = {}

for symbol in symbols:
    try:
        df_1h = all_data_1h[all_data_1h['symbol'] == symbol].copy()
        df_4h = all_data_4h[all_data_4h['symbol'] == symbol].copy()
        if df_1h.empty or df_4h.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        # Check for missing values
        missing_values = df_1h.isnull().sum()
        print(f"Missing values in each column for {symbol}:")
        print(missing_values)

        # Check for outliers in the 'close' prices
        # Using a simple statistical method to identify outliers
        q1 = df_1h['close'].quantile(0.25)
        q3 = df_1h['close'].quantile(0.75)
        iqr = q3 - q1
        outliers = df_1h[(df_1h['close'] < (q1 - 1.5 * iqr)) | (df_1h['close'] > (q3 + 1.5 * iqr))]
        #print(f"Number of outliers in 'close' prices for {symbol}: {len(outliers)}")

        # Check data range
        #print(f"Data range from {df_1h['timestamp'].min()} to {df_1h['timestamp'].max()} for {symbol}")

        # Visual inspection of 'close' prices
        """
        plt.figure(figsize=(10, 5))
        plt.plot(df_1h['timestamp'], df_1h['close'], label='Close Price')
        plt.title(f"{symbol} Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid()
        plt.show()
        """

        # Forward fill daily trend into 1H data
        df_1h = df_1h.reset_index()  # Moves datetime index to a 'timestamp' column
        df_4h = df_4h.reset_index()

        data_merged = pd.merge_asof(
            df_1h.sort_values('timestamp'),
            df_4h[['timestamp', 'rsi', 'macd', 'macd_signal', 'macd_hist']],
            on='timestamp',
            direction='backward',
            suffixes=('', '_4h')
        )

        # Trend filter: 200-day Moving Average
        #data_merged['trend'].fillna(method='ffill', inplace=True)
        trend_filter = data_merged['close'] > data_merged['close'].rolling(window=4800).mean()

        # Volume filter: Check if volume is higher than average (20-day)
        data_merged['avg_volume'] = data_merged['volume'].rolling(window=480).mean()
        volume_filter = data_merged['volume'] > data_merged['avg_volume']
        
        # Add ROC (Rate of Change) as a trend strength filter
        data_merged['roc'] = (data_merged['close'] - data_merged['close'].shift(240)) / data_merged['close'].shift(240) * 100  # 10-day ROC
        
        # Print recent RSI and MACD values for debugging
        #print(f"Last 10 RSI and MACD values for {symbol}:")
        #print(df_1h[['timestamp', 'close', 'rsi', 'macd', 'macd_signal']].tail(10))

        rsi_threshold = 30

        recent_rsi_cross_1h = (data_merged['rsi'] > rsi_threshold) & (data_merged['rsi'].shift(1) <= rsi_threshold)
        recent_macd_cross_1h = (data_merged['macd'] > data_merged['macd_signal']) & (data_merged['macd_hist'] > data_merged['macd_hist'].shift(1))
        macd_hist_rising_1h = (data_merged['macd_hist'] > data_merged['macd_hist'].shift(1))

        recent_rsi_cross_4h = (data_merged['rsi_4h'] > rsi_threshold) & (data_merged['rsi_4h'].shift(1) <= rsi_threshold)
        recent_macd_cross_4h = (data_merged['macd_4h'] > data_merged['macd_signal_4h']) & (data_merged['macd_hist_4h'] > data_merged['macd_hist_4h'].shift(1))
        macd_hist_rising_4h = (data_merged['macd_hist_4h'] > data_merged['macd_hist_4h'].shift(1))

        # Buy signal: 1H signal + 4H confirmation
        buy_signal = (
            # Follows very well the benchmark with better performance in ATOS
            recent_rsi_cross_1h | (recent_macd_cross_1h | macd_hist_rising_1h)
            & (recent_macd_cross_4h | macd_hist_rising_4h)
            
            #(recent_rsi_cross_1h & (recent_macd_cross_1h | macd_hist_rising_1h)) &
            #(recent_rsi_cross_4h & (recent_macd_cross_4h | macd_hist_rising_4h)) #&
            
            #(trend_filter) &
            #(volume_filter) &
            #(data_merged['roc'] > 0)
        )

        recent_rsi_drop_1h = (data_merged['rsi'] < (100 - rsi_threshold)) & (data_merged['rsi'].shift(1) >= (100 - rsi_threshold))
        recent_macd_drop_1h = (data_merged['macd'] < data_merged['macd_signal']) & (data_merged['macd_hist'] < data_merged['macd_hist'].shift(1))
        macd_hist_falling_1h = (data_merged['macd_hist'] < data_merged['macd_hist'].shift(1))

        recent_rsi_drop_4h = (data_merged['rsi_4h'] < (100 - rsi_threshold)) & (data_merged['rsi_4h'].shift(1) >= (100 - rsi_threshold))
        recent_macd_drop_4h = (data_merged['macd_4h'] < data_merged['macd_signal_4h']) & (data_merged['macd_hist_4h'] < data_merged['macd_hist_4h'].shift(1))
        macd_hist_falling_4h = (data_merged['macd_hist_4h'] < data_merged['macd_hist_4h'].shift(1))

        # Sell signal: 1H signal + 4H confirmation
        sell_signal = (
            # Follows very well the benchmark with better performance in ATOS
            recent_rsi_drop_1h & (recent_macd_drop_1h | macd_hist_falling_1h)
            & (recent_macd_drop_4h | macd_hist_falling_4h)
            
            #(recent_rsi_drop_1h & (recent_macd_drop_1h & macd_hist_falling_1h)) &
            #(recent_rsi_drop_4h & (recent_macd_drop_4h & macd_hist_falling_4h)) #&
            
            #(trend_filter) &
            #(volume_filter)
        )  

        data_merged['buy_signal'] = buy_signal
        data_merged['sell_signal'] = sell_signal

        # Define total capital and position size percentage
        #position_size_percentage = 0.1  # Invest 10% of total capital in each trade

        # Strategy simulation with position sizing
        cash = initial_cash
        shares = 0
        position = 0
        portfolio_values = []
        
        for i in range(len(data_merged)):
            price = data_merged['close'].iloc[i]
            if data_merged['buy_signal'].iloc[i] and position == 0:
                shares = cash // price
                # Calculate position size
                #position_size = cash * position_size_percentage
                #shares = position_size // price
                cash -= shares * price
                position = 1
            elif data_merged['sell_signal'].iloc[i] and position == 1:
                cash += shares * price
                shares = 0
                position = 0
            total_value = cash + shares * price
            portfolio_values.append(total_value)
                    
        data_merged['portfolio_value'] = portfolio_values

        # Benchmark: Buy & Hold
        data_merged['benchmark_value'] = initial_cash * (data_merged['close'] / data_merged['close'].iloc[0])

        # Save metrics
        results[symbol] = compute_metrics(data_merged, initial_cash)

        
        # Plot Portfolio Value, Benchmark and buy/sell signals
        plt.figure(figsize=(12, 5))
        plt.plot(data_merged['timestamp'], data_merged['portfolio_value'], label='Strategy')
        plt.plot(data_merged['timestamp'], data_merged['benchmark_value'], label='Buy & Hold')
        plt.scatter(data_merged['timestamp'][data_merged['buy_signal']], data_merged['close'][data_merged['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(data_merged['timestamp'][data_merged['sell_signal']], data_merged['close'][data_merged['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.title(f"{symbol} Strategy vs Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        

        # Plot RSI
        plt.figure(figsize=(12, 5))
        plt.plot(data_merged['timestamp'], data_merged['rsi'], label='RSI')
        plt.axhline(rsi_threshold, color='red', linestyle='--', label=f'RSI {rsi_threshold}')
        plt.axhline((100 - rsi_threshold), color='green', linestyle='--', label=f'RSI {100 - rsi_threshold}')
        #plt.scatter(df_1h['timestamp'][df_1h['buy_signal']], df_1h['close'][df_1h['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        #plt.scatter(df_1h['timestamp'][df_1h['sell_signal']], df_1h['close'][df_1h['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.scatter(data_merged['timestamp'][data_merged['buy_signal']], data_merged['rsi'][data_merged['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(data_merged['timestamp'][data_merged['sell_signal']], data_merged['rsi'][data_merged['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.title(f"{symbol} RSI")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        

        # Plot MACD
        plt.figure(figsize=(12, 5))
        plt.plot(data_merged['timestamp'], data_merged['macd'], label='MACD Line')
        plt.plot(data_merged['timestamp'], data_merged['macd_signal'], label='Signal Line')
        plt.bar(data_merged['timestamp'], macd_hist_1h, label='MACD Histogram', color='grey', alpha=0.3)
        # Plot on close price
        #plt.scatter(df_1h['timestamp'][df_1h['buy_signal']], df_1h['close'][df_1h['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        #plt.scatter(df_1h['timestamp'][df_1h['sell_signal']], df_1h['close'][df_1h['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        # Plot on MACD lines
        plt.scatter(data_merged['timestamp'][data_merged['buy_signal']], data_merged['macd'][data_merged['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(data_merged['timestamp'][data_merged['sell_signal']], data_merged['macd'][data_merged['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
        plt.title(f"{symbol} MACD")
        plt.xlabel("Date")
        plt.ylabel("MACD")
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
summary_df.to_csv("stock/csv/summary_rsi_macd_roc_hourly.csv")
