import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

client = StockHistoricalDataClient(API_KEY, API_SECRET)
request_params = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Minute,
    start="2024-01-01",
    end="2024-12-31"
)

bars = client.get_stock_bars(request_params).df
bars = bars.reset_index()  # Make sure the index is a column

# Create 'Date' column from the timestamp
bars['Date'] = pd.to_datetime(bars['timestamp']).dt.date

# Example: Get 5-minute bars for SPY for the past year
# ticker = 'SPY'
# data = yf.download(ticker, period='1d', interval='5m')
# data = data.reset_index()
# data['Date'] = data['Datetime'].dt.date

def get_intraday_pct_move(df):
    daily_moves = []

    for date, group in df.groupby('Date'):
        group = group.set_index('timestamp')
        open_price = float(group['open'].iloc[0])  # Ensure float
        first_30min_median = float(group['close'].iloc[:6].median())  # Ensure float

        pct_move = ((first_30min_median - open_price) / open_price) * 100
        daily_moves.append({'Date': date, 'PctMove': float(pct_move)})  # Ensure float

    return pd.DataFrame(daily_moves)
    
intraday_moves = get_intraday_pct_move(bars)

# Mock regime columns
np.random.seed(42)
intraday_moves['Gamma'] = np.random.choice(['Low', 'High'], size=len(intraday_moves))
intraday_moves['SpotVol'] = np.random.choice(['Low', 'High'], size=len(intraday_moves))
intraday_moves['Regime'] = intraday_moves['Gamma'] + ' Gamma & ' + intraday_moves['SpotVol'] + ' SpotVol'

# Ensure PctMove is numeric
intraday_moves['PctMove'] = pd.to_numeric(intraday_moves['PctMove'])

plt.figure(figsize=(12, 6))

regimes = intraday_moves['Regime'].unique()
colors = sns.color_palette(n_colors=len(regimes))

for i, regime in enumerate(regimes):
    subset = intraday_moves[intraday_moves['Regime'] == regime]
    # Step histogram
    sns.histplot(
        subset['PctMove'],
        bins=40,
        stat='density',
        element='step',
        fill=False,
        color=colors[i],
        label=f"{regime} (hist)"
    )
    # Smooth KDE
    sns.kdeplot(
        subset['PctMove'],
        color=colors[i],
        label=f"{regime} (kde)",
        linewidth=2
    )

plt.title(f"Distribution of Intraday % Move by Regime (1st 30min vs median, {intraday_moves['Date'].min()} to {intraday_moves['Date'].max()})")
plt.xlabel("% Move from Open")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()

# 1. Calculate % move from open for each 5-min bar
bars['Time'] = pd.to_datetime(bars['timestamp']).dt.time
bars['PctMoveFromOpen'] = 100 * (bars['close'] - bars.groupby('Date')['open'].transform('first')) / bars.groupby('Date')['open'].transform('first')

# 2. Merge regime info
bars = bars.merge(intraday_moves[['Date', 'Regime']], on='Date', how='left')

# 3. Prepare for plotting
regime_list = bars['Regime'].unique()
times = sorted(bars['Time'].unique())
nrows = 2
ncols = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, regime in enumerate(regime_list):
    ax = axes[i]
    regime_bars = bars[bars['Regime'] == regime]
    days = regime_bars['Date'].unique()
    all_paths = []
    for day in days:
        day_bars = regime_bars[regime_bars['Date'] == day].sort_values('Time')
        times_day = day_bars['Time']
        day_path = day_bars['PctMoveFromOpen'].values
        if len(day_path) > 0:
            all_paths.append(day_path)
            # Use a very light color and low alpha for daily lines
            ax.plot([t.strftime('%H:%M') for t in times_day], day_path, color='#4A90E2', alpha=0.03, linewidth=1)
    if len(all_paths) == 0:
        continue
    # To compute mean/std, need to align by time. Use pandas pivot:
    pivot = regime_bars.pivot(index='Date', columns='Time', values='PctMoveFromOpen')
    mean_path = pivot.mean(axis=0)
    std_path = pivot.std(axis=0)
    ax.plot([t.strftime('%H:%M') for t in mean_path.index], mean_path.values, color='black', linewidth=2, zorder=10)
    ax.fill_between([t.strftime('%H:%M') for t in mean_path.index], mean_path - std_path, mean_path + std_path, color='gray', alpha=0.25, zorder=5)
    ax.set_title(f"{regime} ({len(days)} days)")
    ax.set_xlabel("Time of Day (ET)")
    ax.set_ylabel("% Move from Open")

start_date = bars['Date'].min()
end_date = bars['Date'].max()
plt.suptitle(f"SPY Intraday % Move by Regime with Bands ({start_date} to {end_date})")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
