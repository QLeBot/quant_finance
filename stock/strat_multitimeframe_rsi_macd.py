import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pytz
import yfinance as yf

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
symbols = ["NVDA"]
#symbols = ["NVDA", "ATOS"]
#symbols = ["NVDA", "ATOS" , "ATO"]
#symbols = ["SPY"]
initial_cash = 10000

start_date = datetime.datetime(2022, 1, 1)
#end_date = datetime.datetime(2024, 12, 31)
end_date = datetime.datetime.now() - datetime.timedelta(days=10)

# Request 1-hour data for primary analysis and 4-hour data for confirmation
request_params_1h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_1h = stock_client.get_stock_bars(request_params_1h).df
all_data_1h = all_data_1h.reset_index()  # bring 'symbol' and 'timestamp' into columns

request_params_4h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_4h = stock_client.get_stock_bars(request_params_4h).df
all_data_4h = all_data_4h.reset_index()  # bring 'symbol' and 'timestamp' into columns

request_params_day = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_day = stock_client.get_stock_bars(request_params_day).df
all_data_day = all_data_day.reset_index()  # bring 'symbol' and 'timestamp' into columns

request_params_week = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Week,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_week = stock_client.get_stock_bars(request_params_week).df
all_data_week = all_data_week.reset_index()  # bring 'symbol' and 'timestamp' into columns


# Convert the 'timestamp' column to datetime if not already
all_data_1h['timestamp'] = pd.to_datetime(all_data_1h['timestamp'])
all_data_4h['timestamp'] = pd.to_datetime(all_data_4h['timestamp'])
all_data_day['timestamp'] = pd.to_datetime(all_data_day['timestamp'])
all_data_week['timestamp'] = pd.to_datetime(all_data_week['timestamp'])

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

def compute_indicators_rolling(df):
    """
    Calculate indicators using only past data to avoid look-ahead bias.
    Uses rolling windows to ensure we only use data up to the current point.
    """
    # Initialize empty columns
    df['ma200'] = np.nan
    df['rsi'] = np.nan
    df['macd'] = np.nan
    df['macd_signal'] = np.nan
    df['macd_hist'] = np.nan
    
    # Calculate indicators using only past data
    for i in range(200, len(df)):
        # Use only data up to current point
        window = df['close'].iloc[:i+1]
        df.loc[df.index[i], 'ma200'] = window.rolling(window=200).mean().iloc[-1]
        df.loc[df.index[i], 'rsi'] = RSIIndicator(window, window=10).rsi().iloc[-1]
        macd = MACD(window)
        df.loc[df.index[i], 'macd'] = macd.macd().iloc[-1]
        df.loc[df.index[i], 'macd_signal'] = macd.macd_signal().iloc[-1]
        df.loc[df.index[i], 'macd_hist'] = macd.macd_diff().iloc[-1]
    
    return df

all_data_1h = compute_indicators_rolling(all_data_1h)
all_data_4h = compute_indicators_rolling(all_data_4h)
all_data_day = compute_indicators_rolling(all_data_day)
all_data_week = compute_indicators_rolling(all_data_week)

macd_hist_1h = all_data_1h['macd_hist'].iloc[-1]
macd_hist_4h = all_data_4h['macd_hist'].iloc[-1]
macd_hist_day = all_data_day['macd_hist'].iloc[-1]
macd_hist_week = all_data_week['macd_hist'].iloc[-1]

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
    benchmark_value = df['benchmark_value'].iloc[-1]
    profit = final_value - initial_cash
    benchmark_profit = benchmark_value - initial_cash
    roi = (profit / initial_cash) * 100
    benchmark_roi = (benchmark_profit / initial_cash) * 100

    returns = df['portfolio_value'].pct_change().dropna()

    # Infer timeframe from median time difference
    time_diffs = pd.to_datetime(df['timestamp']).diff().dropna()
    median_diff = time_diffs.median()
    seconds_per_period = median_diff.total_seconds()

    risk_free_rate = get_risk_free_rate_tnx()
    print(f"Risk-free rate: {risk_free_rate}")

    #periods_per_year = 365.25 * 24 * 3600 / seconds_per_period
    # alternative periods per year calculation based on 252 trading days per year and 6.5 hours per day for 1H data
    periods_per_year = 252 * 6.5 * 3600 / seconds_per_period
    
    # Sharpe ratio = (Return - Risk-free rate) / Standard deviation of returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(periods_per_year) if np.std(excess_returns) > 0 else np.nan
    #sharpe_ratio = (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(periods_per_year) if np.std(returns) > 0 else np.nan
    # sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year) if np.std(returns) > 0 else np.nan

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
        'Benchmark Value': round(benchmark_value, 2),
        'Profit': round(profit, 2),
        'Benchmark Profit': round(benchmark_profit, 2),
        'ROI (%)': round(roi, 2),
        'Benchmark ROI (%)': round(benchmark_roi, 2),
        'CAGR (%)': round(cagr, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Number of Trades': num_trades,
        'Win Rate (%)': round(win_rate, 2) if not np.isnan(win_rate) else 'N/A',
        #'Timeframe (seconds)': int(seconds_per_period),
        #'Periods per year': round(periods_per_year, 2)
    }

def merge_timeframes_properly(df_1h, df_4h, df_day, df_week):
    """
    Merge different timeframe data properly to avoid look-ahead bias.
    Only uses data up to the current point in time.
    """
    # Ensure all dataframes have timestamp as index
    df_1h = df_1h.set_index('timestamp')
    df_4h = df_4h.set_index('timestamp')
    df_day = df_day.set_index('timestamp')
    df_week = df_week.set_index('timestamp')
    
    # Initialize merged dataframe with 1h data
    merged = df_1h.copy()
    
    # Add columns for higher timeframe data
    for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
        merged[f'{col}_4h'] = np.nan
        merged[f'{col}_day'] = np.nan
        merged[f'{col}_week'] = np.nan
    
    # Fill in higher timeframe data using only past data
    for i in range(len(merged)):
        current_time = merged.index[i]
        
        # Get the most recent 4h data point before current_time
        past_4h = df_4h[df_4h.index <= current_time].iloc[-1] if not df_4h[df_4h.index <= current_time].empty else None
        if past_4h is not None:
            for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
                merged.loc[current_time, f'{col}_4h'] = past_4h[col]
        
        # Get the most recent daily data point before current_time
        past_day = df_day[df_day.index <= current_time].iloc[-1] if not df_day[df_day.index <= current_time].empty else None
        if past_day is not None:
            for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
                merged.loc[current_time, f'{col}_day'] = past_day[col]
        
        # Get the most recent weekly data point before current_time
        past_week = df_week[df_week.index <= current_time].iloc[-1] if not df_week[df_week.index <= current_time].empty else None
        if past_week is not None:
            for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
                merged.loc[current_time, f'{col}_week'] = past_week[col]
    
    return merged

# Backtest with MTF confirmation
results = {}

# Add a list to store trade details
trade_log = []

for symbol in symbols:
    try:
        df_1h = all_data_1h[all_data_1h['symbol'] == symbol].copy()
        df_4h = all_data_4h[all_data_4h['symbol'] == symbol].copy()
        df_day = all_data_day[all_data_day['symbol'] == symbol].copy()
        df_week = all_data_week[all_data_week['symbol'] == symbol].copy()
        if df_1h.empty or df_4h.empty or df_day.empty or df_week.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        # Handle missing values with backward fill
        df_1h = df_1h.bfill()
        df_4h = df_4h.bfill()
        df_day = df_day.bfill()
        df_week = df_week.bfill()

        # If there are still missing values at the start, fill with the first valid value
        df_1h = df_1h.fillna(method='bfill')
        df_4h = df_4h.fillna(method='bfill')
        df_day = df_day.fillna(method='bfill')
        df_week = df_week.fillna(method='bfill')

        # Check for outliers in the 'close' prices
        # Using a simple statistical method to identify outliers
        """
        q1 = df_1h['close'].quantile(0.25)
        q3 = df_1h['close'].quantile(0.75)
        iqr = q3 - q1
        outliers = df_1h[(df_1h['close'] < (q1 - 1.5 * iqr)) | (df_1h['close'] > (q3 + 1.5 * iqr))]
        print(f"Number of outliers in 'close' prices for {symbol}: {len(outliers)}")
        """

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
        df_day = df_day.reset_index()
        df_week = df_week.reset_index()

        # Use proper time-based merging
        data_merged = merge_timeframes_properly(df_1h, df_4h, df_day, df_week)
        
        # Reset index to make timestamp a column again
        data_merged = data_merged.reset_index()

        # Trend filter: 200-day Moving Average ; 200 day = 4800 hours
        #data_merged['trend'].fillna(method='ffill', inplace=True)
        trend_filter = data_merged['close'] > data_merged['close'].rolling(window=4800).mean()

        # Volume filter: Check if volume is higher than average (20-day) ; 20 day = 480 hours
        data_merged['avg_volume'] = data_merged['volume'].rolling(window=480).mean()
        volume_filter = data_merged['volume'] > data_merged['avg_volume']
        
        # Add ROC (Rate of Change) as a trend strength filter
        data_merged['roc'] = (data_merged['close'] - data_merged['close'].shift(240)) / data_merged['close'].shift(240) * 100  # 10-day ROC
        
        # Print recent RSI and MACD values for debugging
        #print(f"Last 10 RSI and MACD values for {symbol}:")
        #print(df_1h[['timestamp', 'close', 'rsi', 'macd', 'macd_signal']].tail(10))

        rsi_threshold = 30

        print(type(data_merged['rsi']))
        print(type(data_merged['rsi_4h']))
        print(type(data_merged['rsi_day']))
        print(type(data_merged['rsi_week']))

        # Generate signals using only past data
        data_merged['buy_signal'] = False
        data_merged['sell_signal'] = False
        
        for i in range(1, len(data_merged)):
            # Get current and previous values
            current = data_merged.iloc[i]
            prev = data_merged.iloc[i-1]
            
            # 1H signals
            rsi_cross_1h = (current['rsi'] > rsi_threshold) and (prev['rsi'] <= rsi_threshold)
            macd_cross_1h = (current['macd'] > current['macd_signal']) and (current['macd_hist'] > prev['macd_hist'])
            macd_hist_rising_1h = (current['macd_hist'] > prev['macd_hist'])
            
            # 4H signals
            rsi_cross_4h = (current['rsi_4h'] > rsi_threshold) and (prev['rsi_4h'] <= rsi_threshold)
            macd_cross_4h = (current['macd_4h'] > current['macd_signal_4h']) and (current['macd_hist_4h'] > prev['macd_hist_4h'])
            macd_hist_rising_4h = (current['macd_hist_4h'] > prev['macd_hist_4h'])
            
            # Day signals
            rsi_cross_day = (current['rsi_day'] > rsi_threshold) and (prev['rsi_day'] <= rsi_threshold)
            macd_cross_day = (current['macd_day'] > current['macd_signal_day']) and (current['macd_hist_day'] > prev['macd_hist_day'])
            macd_hist_rising_day = (current['macd_hist_day'] > prev['macd_hist_day'])
            
            # Week signals
            rsi_cross_week = (current['rsi_week'] > rsi_threshold) and (prev['rsi_week'] <= rsi_threshold)
            macd_cross_week = (current['macd_week'] > current['macd_signal_week']) and (current['macd_hist_week'] > prev['macd_hist_week'])
            macd_hist_rising_week = (current['macd_hist_week'] > prev['macd_hist_week'])
            
            # Buy signal: 1H signal + confirmation from higher timeframes
            buy_signal = (
                (rsi_cross_1h or (macd_cross_1h or macd_hist_rising_1h))
                and (macd_cross_4h or macd_hist_rising_4h)
                and (macd_cross_day or macd_hist_rising_day)
                and (macd_cross_week or macd_hist_rising_week)
            )
            
            # Sell signals
            rsi_drop_1h = (current['rsi'] < (100 - rsi_threshold)) and (prev['rsi'] >= (100 - rsi_threshold))
            macd_drop_1h = (current['macd'] < current['macd_signal']) and (current['macd_hist'] < prev['macd_hist'])
            macd_hist_falling_1h = (current['macd_hist'] < prev['macd_hist'])
            
            rsi_drop_4h = (current['rsi_4h'] < (100 - rsi_threshold)) and (prev['rsi_4h'] >= (100 - rsi_threshold))
            macd_drop_4h = (current['macd_4h'] < current['macd_signal_4h']) and (current['macd_hist_4h'] < prev['macd_hist_4h'])
            macd_hist_falling_4h = (current['macd_hist_4h'] < prev['macd_hist_4h'])
            
            rsi_drop_day = (current['rsi_day'] < (100 - rsi_threshold)) and (prev['rsi_day'] >= (100 - rsi_threshold))
            macd_drop_day = (current['macd_day'] < current['macd_signal_day']) and (current['macd_hist_day'] < prev['macd_hist_day'])
            macd_hist_falling_day = (current['macd_hist_day'] < prev['macd_hist_day'])
            
            rsi_drop_week = (current['rsi_week'] < (100 - rsi_threshold)) and (prev['rsi_week'] >= (100 - rsi_threshold))
            macd_drop_week = (current['macd_week'] < current['macd_signal_week']) and (current['macd_hist_week'] < prev['macd_hist_week'])
            macd_hist_falling_week = (current['macd_hist_week'] < prev['macd_hist_week'])
            
            # Sell signal: 1H signal + confirmation from higher timeframes
            sell_signal = (
                (rsi_drop_1h and (macd_drop_1h or macd_hist_falling_1h))
                and (macd_drop_4h or macd_hist_falling_4h)
                and (macd_drop_day or macd_hist_falling_day)
                and (macd_drop_week or macd_hist_falling_week)
            )
            
            # Set signals
            data_merged.loc[data_merged.index[i], 'buy_signal'] = buy_signal
            data_merged.loc[data_merged.index[i], 'sell_signal'] = sell_signal

        # Strategy simulation with position sizing
        cash = initial_cash
        shares = 0
        position = 0
        portfolio_values = []
        entry_price = 0

        for i in range(len(data_merged)):
            price = data_merged['close'].iloc[i]
            timestamp = data_merged['timestamp'].iloc[i]
            if data_merged['buy_signal'].iloc[i] and position == 0:
                shares = cash // price
                cash -= shares * price
                position = 1
                entry_price = price
                # Record the buy trade
                trade_log.append({'timestamp': timestamp, 'action': 'buy', 'entry_price': entry_price, 'shares': shares})
            elif data_merged['sell_signal'].iloc[i] and position == 1:
                cash += shares * price
                exit_price = price
                pnl = (exit_price - entry_price) * shares
                # Record the sell trade
                trade_log.append({'timestamp': timestamp, 'action': 'sell', 'entry_price': entry_price, 'exit_price': exit_price, 'shares': shares, 'pnl': pnl})
                shares = 0
                position = 0
            total_value = cash + shares * price
            portfolio_values.append(total_value)
                    
        data_merged['portfolio_value'] = portfolio_values

        # Benchmark: Buy & Hold
        data_merged['benchmark_value'] = initial_cash * (data_merged['close'] / data_merged['close'].iloc[0])

        # Save metrics
        results[symbol] = compute_metrics(data_merged, initial_cash)

        
        # Define a constant value for the y-axis where the markers will be placed
        marker_y_position = data_merged['portfolio_value'].min() - 0.05 * (data_merged['portfolio_value'].max() - data_merged['portfolio_value'].min())

        # Plot the main line
        plt.plot(data_merged['timestamp'], data_merged['portfolio_value'], label='Strategy')

        # Plot the benchmark line
        plt.plot(data_merged['timestamp'], data_merged['benchmark_value'], label='Buy & Hold')

        # Plot buy markers on a single line
        plt.scatter(data_merged['timestamp'][data_merged['buy_signal']], 
                    [marker_y_position] * data_merged['buy_signal'].sum(), 
                    marker='^', color='green', label='Buy', alpha=0.7)

        # Plot sell markers on a single line
        plt.scatter(data_merged['timestamp'][data_merged['sell_signal']], 
                    [marker_y_position] * data_merged['sell_signal'].sum(), 
                    marker='v', color='red', label='Sell', alpha=0.7)

        # Add labels and legend
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.title("Strategy vs Benchmark with Buy/Sell Markers on a Single Line")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        """
        trades_df = pd.DataFrame(trade_log)
        trades_df['equity'] = trades_df['pnl'].cumsum()
        trades_df['peak'] = trades_df['equity'].cummax()
        trades_df['drawdown'] = (trades_df['equity'] - trades_df['peak']) / trades_df['peak']

        # Plot Equity Curve
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 1, 1)
        plt.plot(trades_df['timestamp'], trades_df['equity'], label='Equity Curve')
        plt.title(f"{symbol} Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity Value")
        plt.legend()
        plt.grid()
        #plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.plot(trades_df['timestamp'], trades_df['drawdown'], label='Drawdown')
        plt.title(f"{symbol} Drawdown Curve")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] < 0]

        avg_win = win_trades['pnl'].mean()
        avg_loss = loss_trades['pnl'].mean()

        total_trades = len(trades_df)
        win_rate = len(win_trades) / total_trades * 100
        loss_rate = len(loss_trades) / total_trades * 100

        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        trades_df['risk_per_trade'] = abs(trades_df['entry_price'] - trades_df['exit_price']) * trades_df['shares']
        trades_df['r_multiple'] = trades_df['pnl'].fillna(0) / trades_df['risk_per_trade'].fillna(0)
        avg_r_multiple = trades_df['r_multiple'].mean()

        print(f"{symbol} average win: {avg_win:.2f} average loss: {avg_loss:.2f} win rate: {win_rate:.2f}% loss rate: {loss_rate:.2f}% expectancy: {expectancy:.2f} avg r multiple: {avg_r_multiple:.2f}")
        """
        # Plot RSI
        """
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
        """

        # Plot MACD
        """
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
        """

        # Save trades to CSV
        trades_df = pd.DataFrame(trade_log)
        trades_df.to_csv(f"stock/csv/trades_{symbol}.csv", index=False)
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

# Results Summary
print("\n📊 Backtest Summary:")
summary_df = pd.DataFrame(results).T
print(summary_df)
#summary_df.to_csv("stock/csv/strat_multitimeframe_rsi_macd.csv")
