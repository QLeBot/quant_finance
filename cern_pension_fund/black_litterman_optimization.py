import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# --- Fund Asset Classes and ETF Proxies ---
asset_classes = {
    "fixed_income": "AGG",
    "equity": "VT",
    "real_estate": "VNQ",
    "infrastructure": "IFRA",
    "timber_farmland": "WOOD",
    "private_equity": "PSP",
    "hedge_funds": "HDG",
    "metals_commodities": "DBC",
    "cash": "BIL"
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
tickers = list(asset_classes.values())
data = yf.download(tickers, start="2019-01-01", end="2024-01-01")['Close']

returns = data.pct_change().dropna()
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

# --- Step 4: CVaR Optimization ---
def calculate_cvar(weights, alpha=0.05):
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, 100 * alpha)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar  # for minimization

def objective(weights, lam=0.5):
    cvar = calculate_cvar(weights)
    ret = weights @ posterior_returns
    return lam * cvar - (1 - lam) * ret

# Constraints
n = len(asset_classes)
bounds = [(0, 1)] * n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
initial_weights = np.array([1/n] * n)

# Add a minimum expected return constraint using posterior returns
target_return = posterior_returns.mean()  # or use a % like 6%/year
constraints.append({'type': 'ineq', 'fun': lambda w: w @ posterior_returns - target_return})

bl_df = pd.DataFrame({
    "Implied": pi,
    "Posterior": posterior_returns
}, index=asset_classes.keys())
print(bl_df.sort_values("Posterior", ascending=False))

for asset, idx in zip(asset_classes.keys(), range(len(asset_classes))):
    w = np.zeros(len(asset_classes))
    w[idx] = 1
    cvar = calculate_cvar(w)
    print(f"{asset}: CVaR = {-cvar:.4%}")

# Optimization
result = minimize(calculate_cvar, initial_weights, bounds=bounds, constraints=constraints)
opt_weights = result.x

# --- Output ---
opt_df = pd.Series(opt_weights, index=asset_classes.keys()).sort_values(ascending=False)
print("📊 Optimal Portfolio Weights (CVaR-Optimized Black-Litterman):\n")
print(opt_df.apply(lambda x: f"{x:.2%}"))
