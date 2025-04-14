import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# =============== Parameters ================
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'V', 'MA',
    'UNH', 'HD', 'PG', 'DIS', 'NFLX',
    'XOM', 'CVX', 'BA', 'KO', 'PFE'
]

start_date = "2022-01-01"
end_date = "2024-01-01"
delta = 0.5  # risk aversion
tau = 0.05   # uncertainty in prior
view_strength = 0.03  # expected return difference in views
max_weight = 0.15  # 15% max per asset
lambda_reg = 0.1

# =============== Download Data ===============
data = yf.download(tickers, start=start_date, end=end_date)['Close']
returns = data.pct_change().dropna()

# =============== Covariance Matrix & Market Implied Returns ===============
cov_matrix = returns.cov()
market_weights = np.repeat(1/len(tickers), len(tickers))  # Equal weights
pi = delta * cov_matrix @ market_weights

# =============== Define Views ===============
# Example: AAPL will outperform TSLA by 3%, NVDA will outperform META by 3%
P = np.array([
    [1, 0, -1, 1, 1, -1, 0, 1, -2, 0, 0, 1, 1, -1, 2, 5, 0, -1, 7, 3],
    [-1, 1, 1, 0, -1, 3, 1, 2, 2, -1, -1, 0, 0, 1, 1, 1, 0, 0, -1, 2]
])

Q = np.array([view_strength, view_strength])

# Omega: diagonal variance of the views (simplified here)
omega = np.diag(np.dot(P, tau * cov_matrix.values @ P.T))
omega = np.diag(omega)

# =============== Black-Litterman Posterior Returns ===============
middle = np.linalg.inv(np.linalg.inv(tau * cov_matrix.values) + P.T @ np.linalg.inv(omega) @ P)
posterior_mean = middle @ (np.linalg.inv(tau * cov_matrix.values) @ pi + P.T @ np.linalg.inv(omega) @ Q)

# =============== Mean-Variance Optimization ===============
n = len(tickers)
w = cp.Variable(n)
risk = cp.quad_form(w, cov_matrix.values)
expected_return = posterior_mean @ w

# Objective: maximize return - risk_aversion * risk
problem = cp.Problem(cp.Maximize(expected_return - delta * risk),
                     [cp.sum(w) == 1, w >= 0, w <= max_weight])

problem.solve()

optimal_weights = w.value

# =============== Results ===============
portfolio = pd.DataFrame({
    'Ticker': tickers,
    'Weight': optimal_weights
})

print("\nOptimal Black-Litterman Portfolio Weights:")
print(portfolio.sort_values(by='Weight', ascending=False).reset_index(drop=True).round(4))

# =============== Plotting ===============
plt.figure(figsize=(10, 6))
plt.bar(portfolio['Ticker'], portfolio['Weight'], color='skyblue')
plt.xlabel('Ticker')
plt.ylabel('Weight')
plt.title('Optimal Black-Litterman Portfolio Weights')
plt.show()