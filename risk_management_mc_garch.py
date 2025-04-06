import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# GARCH model
from arch import arch_model

from alpaca.data import StockHistoricalDataClient, StockHistoricalDataRequest, TimeFrame

# --- Alpaca API Setup ---
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# --- Configuration ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
weights = np.array([0.25, 0.25, 0.25, 0.25])
confidence_level = 0.95

request_params = StockHistoricalDataRequest(
    symbol_or_symbols=tickers,
    timeframe=TimeFrame.Day,
    start=datetime.datetime(2020, 1, 1),
    end=datetime.datetime(2024, 12, 31)
)

# --- Retrieve Historical Data via Alpaca ---
# Note: The Alpaca API v2 allows retrieval of bars with the get_bars() method.
# Here we use a loop to retrieve data for each ticker.
def get_ticker_data(ticker, start, end):
    # Alpaca's get_bars expects ISO format strings
    bars = stock_client.get_bars(
        request_params
    ).df
    # Ensure the timestamp is treated as datetime and sorted
    bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()
    return bars[['close']]

# Download data for each ticker and combine into one DataFrame
price_data = pd.DataFrame()
for ticker in tickers:
    df_ticker = get_ticker_data(ticker, start, end)
    df_ticker.rename(columns={'close': ticker}, inplace=True)
    if price_data.empty:
        price_data = df_ticker.copy()
    else:
        price_data = price_data.join(df_ticker, how='inner')

# --- Calculate Daily Returns ---
returns = price_data.pct_change().dropna()

# --- Portfolio Returns & Basic Metrics ---
portfolio_returns = returns.dot(weights)
mean_return = portfolio_returns.mean()
volatility = portfolio_returns.std()

# --- Historical VaR and CVaR ---
VaR_hist = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
CVaR_hist = portfolio_returns[portfolio_returns <= VaR_hist].mean()

# --- Monte Carlo Simulation for VaR ---
n_simulations = 1000  # number of simulation runs
n_days = 252          # simulate one trading year
simulated_end_returns = []

# Use the latest observed portfolio return as a starting point (or assume 1.0 for normalized returns)
last_value = 1.0
for i in range(n_simulations):
    simulated_value = last_value
    for day in range(n_days):
        # Simulate daily return from a normal distribution using historical mean and volatility
        daily_return = np.random.normal(mean_return, volatility)
        simulated_value *= (1 + daily_return)
    simulated_end_returns.append(simulated_value)

simulated_end_returns = np.array(simulated_end_returns)
VaR_monte = np.percentile(simulated_end_returns, (1 - confidence_level) * 100)

# --- GARCH Model for Volatility Forecasting ---
# We multiply returns by 100 to work with percentages
garch_model = arch_model(portfolio_returns * 100, vol='Garch', p=1, q=1, dist='normal')
garch_res = garch_model.fit(disp='off')
print(garch_res.summary())

# Forecast one day ahead volatility using the fitted GARCH model
forecast_horizon = 1
garch_forecast = garch_res.forecast(horizon=forecast_horizon)
# Predicted variance for the forecast day (note: conversion back to return scale)
predicted_vol = np.sqrt(garch_forecast.variance.values[-1, 0]) / 100
# An approximate GARCH-based VaR (using a z-score for the confidence level, e.g., -1.65 for 95% one-tailed)
from scipy.stats import norm
z_score = norm.ppf(1 - confidence_level)
VaR_garch = mean_return + z_score * predicted_vol

# --- Output Results ---
print("\nPortfolio Risk Summary")
print("-----------------------")
print(f"Mean Daily Return: {mean_return:.5f}")
print(f"Volatility (Std Dev): {volatility:.5f}")
print(f"{int(confidence_level*100)}% Historical VaR: {VaR_hist:.5f}")
print(f"{int(confidence_level*100)}% Historical CVaR: {CVaR_hist:.5f}")
print(f"{int(confidence_level*100)}% Monte Carlo VaR (simulated portfolio value): {VaR_monte:.5f}")
print(f"{int(confidence_level*100)}% GARCH-based VaR: {VaR_garch:.5f}")

# --- Plotting the Distribution of Portfolio Returns ---
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_returns, kde=True, bins=100, color='skyblue')
plt.axvline(VaR_hist, color='red', linestyle='--', label=f'Historical VaR ({VaR_hist:.2%})')
plt.axvline(VaR_garch, color='purple', linestyle='--', label=f'GARCH VaR ({VaR_garch:.2%})')
plt.title("Portfolio Daily Returns with VaR Estimates")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
