# Information about the fund strategy

"""
From https://www.ipe.com/news/cern-pension-fund-posts-negative-returns-while-other-schemes-rebound/10074444.article

Official documentation
https://pensionfund.cern.ch/sites/default/files/2022-09/Statement%20of%20Investment%20Principles_0.pdf
https://pensionfund.cern.ch/sites/default/files/2019-06/Statement%20of%20Investment%20Principles%20-%20EN.pdf


Dive into the fund strategy
https://pensionfund.cern.ch/sites/default/files/2024-09/Annual%20Report%20and%20Financial%20Statements%202023.pdf
"""

# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# Import alpaca data
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest

# --- Alpaca API Setup ---
from dotenv import load_dotenv
import os

load_dotenv()

# Fund data
asset_value = 4.46e9

metals_commodities_allocation = 0.04
cash_allocation = 0.12
fixed_income_allocation = 0.21
equity_allocation = 0.16
real_assets_allocation = 0.22
private_equity_allocation = 0.11
hedge_funds_allocation = 0.14

# Calculate the value of each asset class
metals_commodities_value = asset_value * metals_commodities_allocation
cash_value = asset_value * cash_allocation
fixed_income_value = asset_value * fixed_income_allocation
equity_value = asset_value * equity_allocation
real_assets_value = asset_value * real_assets_allocation
private_equity_value = asset_value * private_equity_allocation
hedge_funds_value = asset_value * hedge_funds_allocation

# Print the value of each asset class
"""
print(f"Metals and Commodities Value: {metals_commodities_value}")
print(f"Cash Value: {cash_value}")
print(f"Fixed Income Value: {fixed_income_value}")
print(f"Equity Value: {equity_value}")
print(f"Real Assets Value: {real_assets_value}")
print(f"Private Equity Value: {private_equity_value}")
print(f"Hedge Funds Value: {hedge_funds_value}")
"""
# 2023 financial statements

# Strategic Asset Allocation (SAA) Target
saa = {
    "fixed_income": 0.265,
    "equity": 0.17,
    "real_assets": {
        "real_estate": 0.19,
        "infrastructure": 0.025,
        "timber_farmland": 0.025,
    },
    "private_equity": 0.06,
    "hedge_funds": 0.11,
    "metals_commodities": 0.055,
    "cash": 0.10,
}

# Current Asset Allocation (CAA) Realized
caa = {
    "fixed_income": 0.2106,
    "equity": 0.1587,
    "real_assets": {
        "real_estate": 0.177,
        "infrastructure": 0.0185,
        "timber_farmland": 0.0234,
    },
    "private_equity": 0.112,
    "hedge_funds": 0.1391,
    "metals_commodities": 0.0355,
    "cash": 0.1245,
}

performance_net_2023 = -0.011

# --- Alpaca API Setup ---

# Load environment variables from .env file
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# --- Configuration ---
tickers = ['AAPL']

start = datetime.datetime(2023, 1, 1)
end = datetime.datetime(2023, 12, 31)

request_params = StockBarsRequest(
    symbol_or_symbols=tickers,
    timeframe=TimeFrame.Day,
    start=start,
    end=end
)

# --- Retrieve Historical Data via Alpaca ---

bars = stock_client.get_stock_bars(request_params).df

print(bars)
