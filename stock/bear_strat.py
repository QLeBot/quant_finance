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

"""
Observations:
- TSLA seems to work in this strategy
- The rest of the stocks are not performing well, either underperforming the buy&hold or loosing money

Tickers:
- ATO for Atos, overall downtrend from 2017 to 2024
- MC.PA for Moet Hennessy Louis Vuitton, overall uptrend to mid 2021 then stable with big swings
- NVDA for Nvidia, overall uptrend, big uptrend mid 2022 to end 2024. End 2024 seems to be the top
- SPY for S&P 500, overall uptrend can be used as a benchmark for overall market
- TSLA for Tesla, slow uptrend 2013 to mid 2020 then big uptrend from mid 2020 to mid/end 2021 then overall stable with big swings

"""

# Parameters
symbols = ["NVDA", "SPY", "TSLA", "MC.PA", "ATO"]
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


def short_bias_volatility_strategy(data):
    """
    Short Bias / Long Volatility Strategies
    - Short Bias: Focuses on identifying overvalued stocks or those in a confirmed downtrend (e.g., using momentum or fundamental signals).
    - Volatility Long Strategies: Use instruments like VIX ETFs or options to benefit from spikes in volatility.
    - Example: Long VIX when the S&P 500 drops below the 200-day moving average.
    """
    short_signals = (data['rsi'] > 70) & (data['macd'] < 0) & (data['price'] < data['200dma'])
    long_vix = (data['spy_price'] < data['spy_200dma'])

    positions = pd.Series(0, index=data.index)
    positions[short_signals] = -1
    if long_vix:
        positions['VIXY'] = 1  # or a volatility ETF

    return positions

def mean_reversion_strategy(data):
    """
    Mean Reversion (especially in oversold conditions)
    - In bear markets, panic selling often leads to oversold conditions that eventually revert.
    - Use indicators like:
        - RSI < 30 + MACD crossover
        - Z-score of price deviation from mean
    - Pairs trading can also work well when correlations temporarily break down.
    """
    # Calculate RSI and MACD indicators
    data['rsi'] = RSIIndicator(data['close'], window=14).rsi()
    macd = MACD(data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_histogram'] = macd.macd_diff()

    # Define oversold condition
    oversold = (data['rsi'] < 30) & (data['macd_histogram'] > 0)
    positions = pd.Series(0, index=data.index)
    positions[oversold] = 1  # go long oversold stocks

    return positions

def pairs_trading_strategy(stock_a, stock_b):
    """
    Statistical Arbitrage / Market Neutral
    - Market neutral strategies (e.g., beta-neutral pairs trading) try to profit from relative mispricings while hedging out market direction.
    - Especially good during volatile or uncertain periods.
    """
    spread = stock_a['price'] - stock_b['price']
    zscore = (spread - spread.mean()) / spread.std()

    long_a = zscore < -1
    short_a = zscore > 1

    positions = pd.DataFrame(0, index=stock_a.index, columns=['A', 'B'])
    positions['A'][long_a] = 1
    positions['B'][long_a] = -1
    positions['A'][short_a] = -1
    positions['B'][short_a] = 1

    return positions

def short_momentum_strategy(data):
    """
    Momentum — but short side focused
    - In down markets, losers tend to keep losing. You can short weak momentum stocks:
        - Rank stocks by 3-12 month returns, short the bottom decile.
        - Combine with filters like low short interest to avoid squeezes.
    """
    momentum = data['price'].pct_change(252)  # 1-year momentum
    quantile = momentum.rank(pct=True)
    
    positions = pd.Series(0, index=data.index)
    positions[quantile < 0.1] = -1  # short worst 10%
    
    return positions

def trend_following_strategy(data):
    """
    Trend Following (Time-Series Momentum)
    - Trade based on the trend of each asset class:
        - Go short when the price is below long-term moving average (e.g., 200-day).
        - This can apply to stocks, commodities, FX, and fixed income.
    """
    long_term_ma = data['price'].rolling(window=200).mean()
    signal = data['price'] > long_term_ma
    
    positions = pd.Series(0, index=data.index)
    positions[signal] = 1
    positions[~signal] = -1  # optional, could also just go flat
    
    return positions

def cross_asset_rotation_strategy(data):
    """
    Cross-Asset Rotation (Defensive Sectors or Asset Classes)
    - Rotate into traditionally safer assets:
        - Utilities, consumer staples, healthcare
        - Gold, Treasury bonds
        - Use macro signals (e.g., inflation, yield curve) for allocation decisions.
    """
    # Assume `data` is a DataFrame of ETFs or assets
    returns = data.pct_change(63).mean()  # 3-month return
    top_assets = returns.nlargest(3).index
    
    weights = pd.Series(0, index=data.columns)
    weights[top_assets] = 1 / len(top_assets)
    
    return weights

def tail_risk_hedging_strategy(portfolio_value, spy_price, vix_level):
    """
    Tail Risk Hedging
    - Use options strategies like:
        - Long put spreads
        - Protective puts on major indices
        - VIX call options
    """
    hedge_cost = 0.02  # 2% of portfolio
    if vix_level < 20:  # cheap volatility
        put_position = hedge_cost * portfolio_value / (spy_price * 0.05)  # 5% OTM puts
    else:
        put_position = 0

    return {'SPY_PUT': put_position}

# Function to apply a given strategy

def apply_strategy(strategy_func, df):
    positions = strategy_func(df)
    cash = initial_cash
    shares = 0
    position = 0
    portfolio_values = []

    for i in range(len(df)):
        price = df['close'].iloc[i]
        if positions.iloc[i] == 1 and position == 0:  # Buy signal
            shares = cash // price
            cash -= shares * price
            position = 1
        elif positions.iloc[i] == -1 and position == 1:  # Sell signal
            cash += shares * price
            shares = 0
            position = 0
        total_value = cash + shares * price
        portfolio_values.append(total_value)

    df['portfolio_value'] = portfolio_values
    return df

# Backtest with strategy selection

selected_strategies = [short_bias_volatility_strategy, mean_reversion_strategy, pairs_trading_strategy, short_momentum_strategy, trend_following_strategy, cross_asset_rotation_strategy, tail_risk_hedging_strategy]

results = {}

for strategy in selected_strategies:
    print(f"\nTesting strategy: {strategy.__name__}")
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

            # Add ROC (Rate of Change) as a trend strength filter
            df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100  # 10-day ROC

            # Apply the selected strategy
            df = apply_strategy(strategy, df)

            # Benchmark: Buy & Hold
            df['benchmark_value'] = initial_cash * (df['close'] / df['close'].iloc[0])

            # Save metrics
            results[(symbol, strategy.__name__)] = compute_metrics(df, initial_cash)

            # Plot
            plt.figure(figsize=(12, 5))
            plt.plot(df['timestamp'], df['portfolio_value'], label='Strategy')
            plt.plot(df['timestamp'], df['benchmark_value'], label='Buy & Hold')
            plt.scatter(df['timestamp'][df['buy_signal']], df['close'][df['buy_signal']], marker='^', color='green', label='Buy', alpha=0.7)
            plt.scatter(df['timestamp'][df['sell_signal']], df['close'][df['sell_signal']], marker='v', color='red', label='Sell', alpha=0.7)
            plt.title(f"{symbol} {strategy.__name__} Strategy vs Benchmark")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error processing {symbol} with {strategy.__name__}: {e}")

# Results Summary
print("\n📊 Backtest Summary:")
summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv("stock/summary_rsi_macd_v2.csv")
