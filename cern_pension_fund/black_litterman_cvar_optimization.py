import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

"""
Experimenting with different asset classes and ETF proxies for a Black-Litterman optimization.

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

mean_returns = returns.mean()
cov_matrix = returns.cov()

# --- Step 1: Black-Litterman implied returns from SAA ---
delta = 2.5  # risk aversion coefficient
w_mkt = np.array([saa[k] for k in asset_classes])  # market cap proxy = SAA
pi = delta * cov_matrix @ w_mkt  # implied returns

# --- Step 2: View: Equity will outperform Fixed Income by 1% ---
P = np.zeros((1, len(asset_classes)))
P[0, list(asset_classes).index("equity")] = 1
P[0, list(asset_classes).index("fixed_income")] = -1
Q = np.array([0.01])
omega = np.array([[0.0001]])

# --- Step 3: Compute posterior returns (Black-Litterman formula) ---
tau = 0.05
tau_cov = tau * cov_matrix
middle = np.linalg.inv(P @ tau_cov @ P.T + omega)
adjustment = tau_cov @ P.T @ middle @ (Q - P @ pi)
posterior_returns = pi + adjustment

# --- Step 4: CVaR Optimization with Return Constraint and Allocation Cap ---
def calculate_cvar(weights, alpha=0.05):
    portfolio_returns = returns @ weights
    if portfolio_returns.isna().any():  # Check if there are any NaNs
        print("NaNs in portfolio returns!")
        return np.nan
    var = np.percentile(portfolio_returns, 100 * alpha)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar  # for minimization

# Constraints
n = len(asset_classes)
bounds = [(0, 0.35)] * n  # Cap maximum weight per asset at 35%
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum of weights = 1

# Add expected return constraint (minimizing CVaR but ensuring a return target)
target_return = posterior_returns.mean()  # Target return: mean of posterior returns
#target_return = 0.06  # 6% annual return target (adjust as needed)
constraints.append({
    'type': 'ineq', 
    'fun': lambda w: w @ posterior_returns - target_return  # Ensure portfolio return >= target
})

# Initial guess (equally weighted portfolio)
initial_weights = np.array([1/n] * n)

# Optimization
result = minimize(calculate_cvar, initial_weights, bounds=bounds, constraints=constraints)
opt_weights = result.x

# --- Output ---
opt_df = pd.Series(opt_weights, index=asset_classes.keys()).sort_values(ascending=False)
print("📊 Optimal Portfolio Weights (CVaR-Optimized Black-Litterman):\n")
print(opt_df.apply(lambda x: f"{x:.2%}"))

# --- Step 5: Check CVaR for Individual Assets (Separate Diagnostic) ---
print("\n📊 Individual Asset CVaR Values:")
for asset, idx in zip(asset_classes.keys(), range(len(asset_classes))):
    # Create portfolio with 100% in one asset
    w = np.zeros(len(asset_classes))
    w[idx] = 1
    
    # Calculate CVaR for this single-asset portfolio
    cvar = calculate_cvar(w)
    
    # Output CVaR for the asset
    print(f"{asset}: CVaR = {-cvar:.4%}")