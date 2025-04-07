import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Configuration
# -----------------------------
swiss_tickers = ['UBSG.SW', 'NESN.SW', 'ROG.SW']
europe_etfs = ['EUNL.DE', 'VWRL.AS', 'EXX6.DE', 'IEGA.L']
global_mix = ['SPY', 'VEA', 'BND', 'GLD']  # Keep some global exposure
tickers = swiss_tickers + europe_etfs + global_mix
weights = [0.3, 0.3, 0.15, 0.1, 0.1, 0.05]  # Target asset allocation
start_date = '2013-01-01'
end_date = '2023-12-31'

# -----------------------------
# Download historical data
# -----------------------------
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
data.dropna(inplace=True)

# -----------------------------
# Rebalanced portfolio simulation
# -----------------------------
def simulate_rebalanced_portfolio(data, weights, rebalance_freq='A'):
    """Simulate portfolio value with periodic rebalancing (default: yearly)."""
    portfolio_values = []
    current_weights = np.array(weights)
    rebalance_dates = data.resample(rebalance_freq).first().index

    value = 1.0  # Start with 1 unit of value
    shares = (value * current_weights) / data.iloc[0].values

    for i in range(1, len(data)):
        date = data.index[i]
        prices = data.iloc[i].values
        value = np.dot(shares, prices)
        portfolio_values.append(value)

        if date in rebalance_dates:
            shares = (value * current_weights) / prices  # Rebalance shares

    return pd.Series(portfolio_values, index=data.index[1:])

portfolio = simulate_rebalanced_portfolio(data, weights)

# -----------------------------
# Calculate performance metrics
# -----------------------------
returns = portfolio.pct_change().dropna()
cumulative_return = (portfolio[-1] / portfolio[0]) - 1
annualized_return = (1 + cumulative_return) ** (1 / (len(returns) / 252)) - 1
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / volatility

print("Performance Metrics")
print("-------------------")
print(f"Annualized Return:  {annualized_return:.2%}")
print(f"Annualized Volatility: {volatility:.2%}")
print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")

# -----------------------------
# Monte Carlo simulation
# -----------------------------
def monte_carlo_simulation(S0, mu, sigma, T=10, N=1000, steps_per_year=252):
    """Simulate future portfolio values using geometric Brownian motion."""
    dt = 1 / steps_per_year
    simulations = np.zeros((N, T * steps_per_year))
    for i in range(N):
        rand_walk = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), T * steps_per_year)
        path = S0 * np.exp(np.cumsum(rand_walk))
        simulations[i] = path
    return simulations

simulations = monte_carlo_simulation(S0=portfolio.iloc[-1], mu=annualized_return, sigma=volatility)

# Plot simulations
plt.figure(figsize=(12, 6))
plt.plot(simulations.T, alpha=0.01, color='blue')
plt.title("Monte Carlo Simulated Future Portfolio Values (10 years)")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()

# -----------------------------
# Liability modeling
# -----------------------------
def present_value_liabilities(payment_per_year=40000, years=20, retirees=100, rate=0.02):
    """Calculate the present value of all future liabilities."""
    liabilities = []
    for t in range(1, years + 1):
        pv = (payment_per_year * retirees) / ((1 + rate) ** t)
        liabilities.append(pv)
    return sum(liabilities)

pv_liabilities = present_value_liabilities()
print(f"\nPresent Value of Pension Liabilities (CHF): {pv_liabilities:,.2f}")

# -----------------------------
# Funding ratio
# -----------------------------
funding_ratio = (portfolio.iloc[-1] * 1_000_000) / pv_liabilities  # Scale up the portfolio
print(f"Estimated Funding Ratio: {funding_ratio:.2f}")

# -----------------------------
# Save results as CSV
# -----------------------------
output_dir = "simulation_results"
os.makedirs(output_dir, exist_ok=True)

portfolio.to_csv(os.path.join(output_dir, "portfolio_values.csv"))
returns.to_csv(os.path.join(output_dir, "portfolio_returns.csv"))

sim_df = pd.DataFrame(simulations.T)
sim_df.to_csv(os.path.join(output_dir, "monte_carlo_simulations.csv"), index=False)

with open(os.path.join(output_dir, "summary.txt"), "w") as f:
    f.write(f"Annualized Return: {annualized_return:.4f}\n")
    f.write(f"Annualized Volatility: {volatility:.4f}\n")
    f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
    f.write(f"Present Value of Liabilities: {pv_liabilities:,.2f} CHF\n")
    f.write(f"Estimated Funding Ratio: {funding_ratio:.2f}\n")

print("\n✅ Simulation complete. Results saved to /simulation_results/")
