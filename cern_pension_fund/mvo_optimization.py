import numpy as np
from scipy.optimize import minimize

# Simulate portfolio returns from historical data
n_simulations = 10000
simulated_weights = np.random.dirichlet(np.ones(len(tickers)), size=n_simulations)
simulated_returns = returns @ simulated_weights.T

# Compute CVaR
def calculate_cvar(returns_series, alpha=0.05):
    var = np.percentile(returns_series, 100 * alpha)
    cvar = returns_series[returns_series <= var].mean()
    return -cvar  # positive number for loss

# Portfolio return simulation
def portfolio_cvar(weights, alpha=0.05):
    portfolio_returns = returns @ weights
    return calculate_cvar(portfolio_returns, alpha)

# Optimization: minimize CVaR for a minimum expected return
target_return = 0.07  # adjust for your pension fund needs

def objective(weights):
    return portfolio_cvar(weights)

constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'ineq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
]

bounds = tuple((0, 1) for _ in tickers)
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
opt_weights_cvar = result.x

pd.Series(opt_weights_cvar, index=tickers).plot.bar(title="CVaR-Optimized Portfolio")
