"""
data from : https://data.mendeley.com/datasets/v43sbd5wpy/1

Sheets :
- Returns : contains a 360x11 matrix, where the rows represent 360 months and the columns represent 11 sectors so that each entry gives the return of the sector in the given month.
- MarketCap : contains a 360x11 matrix, where the rows represent 360 months and the columns represent 11 sectors so that each entry gives the percentage market capitalization of the sector in the given month.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Load the data
file = pd.read_excel('external_data/input_data_black_litterman_portfolio_construction.xlsx', sheet_name=[1, 2], header=None)
print(file.keys())
# transform returns dict into two dataframes
returns_df = file[1]
print(returns_df.head())
market_cap_df = file[2]
print(market_cap_df.head())


# Calculate the market capitalization weights
market_weights = market_cap_df.div(market_cap_df.sum(axis=1), axis=0)

# Calculate the implied excess returns (Pi)
# Assume risk-free rate is 0 for simplicity
risk_free_rate = 0.02
risk_aversion = 2.5  # A typical value for risk aversion
implied_excess_returns = risk_aversion * returns_df.cov().dot(market_weights.mean())

# Define investor views
# Example: View that the first sector will outperform the second by 2%
P = np.array([[1, -1] + [0]*9])  # View matrix
Q = np.array([0.02])  # View returns

# Uncertainty in the views
omega = np.diag([0.0001])  # Small uncertainty

# Regularization term
regularization = 1e-5

# Black-Litterman formula with regularization
# Calculate the adjusted expected returns
covariance_matrix = returns_df.cov()
M_inverse = np.linalg.inv(covariance_matrix)

# Add regularization to P_omega_Pt
P_omega_Pt = np.linalg.inv(P.T @ np.linalg.inv(omega) @ P + np.eye(P.shape[1]) * regularization)

adjusted_returns = np.linalg.inv(M_inverse + P.T @ np.linalg.inv(omega) @ P) @ (M_inverse @ implied_excess_returns + P.T @ np.linalg.inv(omega) @ Q)

# Optimize the portfolio
# For simplicity, use equal weights as a starting point
initial_weights = np.array([1/11] * 11)

# Define the objective function for optimization
def objective(weights):
    return -weights.dot(adjusted_returns) + risk_aversion * weights.T.dot(covariance_matrix).dot(weights)

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Bounds: weights between 0 and 1
bounds = tuple((0, 1) for _ in range(11))

# Optimize
result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

# Optimal weights in percentage
optimal_weights = result.x * 100

for i, weight in enumerate(optimal_weights):
    print(f"Sector {i+1}: {round(weight, 2)}%")





