import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

"""
Experimenting with different asset classes and ETF proxies for a Mean Variance optimization.

Assets are from Yahoo Finance so they are 
"""

# --- Fund Asset Classes and ETF Proxies ---
asset_classes = {
    "fixed_income": ["AGG", "BND"],
    #"equity": ["SPY", "VOO"],
    "equity": ["NESN.SW", "NOVN.SW", "UBS", "CFR", "SLHN.SW", "ZURN.SW", "VOW3.DE", "SIE.DE", "SAP", "OR", "TTE", "AIR", "KER.PA", "ALV", "MC.PA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "AMD", "JPM", "BRK-B", "MCD", "KO", "INTC", "NKE", "DIS", "V", "MA", "PYPL", "ADBE", "CRM", "CSCO", "IBM", "PEP", "WMT", "XOM", "LMT", "CAT", "AXP"],
    "real_estate": ["VNQ", "SCHH"],
    "infrastructure": ["IFRA"],
    "timber_farmland": ["WOOD"],
    "private_equity": ["PSP", "PEX"],
    "hedge_funds": ["HDG"],
    "metals_commodities": ["DBB", "DBC"],
    "cash": ["BIL"]
}

# --- Strategic Asset Allocation (SAA) ---
saa = {
    "fixed_income": 0.265,
    "equity": 0.17,
    "real_estate": 0.19,
    "infrastructure": 0.025,
    "timber_farmland": 0.025,
    "private_equity": 0.06,
    "hedge_funds": 0.11,
    "metals_commodities": 0.055,
    "cash": 0.10,
}

# --- Fetch historical data ---
tickers = [ticker for sublist in asset_classes.values() for ticker in sublist]
data = yf.download(tickers, start="2019-01-01", end="2024-01-01")['Close']

# Calculate returns for each asset class by averaging the returns of its ETFs
returns = pd.DataFrame({
    asset_class: data[tickers].pct_change().mean(axis=1)
    for asset_class, tickers in asset_classes.items()
})

# return the tickers that have NaNs
#print(data[tickers].isna().sum())

returns = returns.fillna(method='ffill').fillna(method='bfill')  # Forward and back-fill missing data

# --- Calculate Expected Returns and Covariance Matrix ---
expected_returns = returns.mean() * 252  # Annualize the returns
cov_matrix = returns.cov() * 252  # Annualize the covariance matrix

# --- Define the Optimization Problem ---
def portfolio_performance(weights, expected_returns, cov_matrix):
    returns = np.dot(weights, expected_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_volatility

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- Optimize Portfolio ---
optimal_portfolio = optimize_portfolio(expected_returns, cov_matrix)
optimal_weights = optimal_portfolio.x

# Calculate the optimal weights for each asset class
opt_weights_by_class = {}
start_idx = 0
for asset_class, tickers in asset_classes.items():
    num_tickers = len(tickers)
    class_weight = optimal_weights[start_idx:start_idx + num_tickers].sum()
    opt_weights_by_class[asset_class] = class_weight
    start_idx += num_tickers

print("📊 Optimal Portfolio Weights by Asset Class (Mean-Variance Optimized):\n")
for asset_class, weight in opt_weights_by_class.items():
    print(f"{asset_class}: {weight:.2%}")

# Calculate the performance of the optimal portfolio
optimal_returns, optimal_volatility = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
print("\nExpected Portfolio Return:", optimal_returns)
print("Expected Portfolio Volatility:", optimal_volatility)

