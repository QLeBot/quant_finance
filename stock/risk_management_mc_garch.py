"""
This code is a risk management script that uses historical data and statistical models to estimate the risk of a stock portfolio. It performs several key tasks:

Data Acquisition:
- The script downloads historical price data for a list of specified stock tickers (AAPL, MSFT, GOOGL, AMZN) from a data source using the alpaca_data function. The data spans from January 1, 2020, to December 31, 2024.
- It processes this data to calculate daily returns for each stock.

Portfolio Metrics Calculation:
- It calculates the mean return and volatility (standard deviation) of the portfolio using the specified weights for each stock.

Risk Measures Calculation:
- Historical VaR and CVaR: The script calculates the Value at Risk (VaR) and Conditional Value at Risk (CVaR) using historical returns. VaR is a measure of the potential loss in value of the portfolio over a defined period for a given confidence interval. CVaR is the expected loss exceeding the VaR.
- Monte Carlo Simulation: It simulates future returns using a Monte Carlo method, assuming normal distribution of returns, to estimate the VaR.
- GARCH Model: The script fits a GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model to the portfolio returns to estimate future volatility and calculate a GARCH-based VaR.

Results and Plotting:
- The script prints a summary of the portfolio's risk metrics, including mean return, volatility, and the calculated VaR and CVaR values.
- It plots the distribution of portfolio returns with lines indicating the historical and GARCH-based VaR estimates. The plot includes:
  - A histogram of the portfolio's daily returns.
  - A kernel density estimate (KDE) of the returns distribution.
  - Vertical lines representing the historical VaR (in red) and GARCH-based VaR (in purple).

The plot visually represents the risk of the portfolio by showing the distribution of returns and highlighting the points where significant losses (as defined by VaR) are expected to occur. This helps in understanding the potential downside risk of the portfolio under different modeling assumptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from typing import Tuple
from arch import arch_model
from scipy.stats import norm

from get_data import alpaca_data

def calculate_portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray) -> Tuple[float, float]:
    """Calculate portfolio mean return and volatility."""
    portfolio_returns = returns.dot(weights)
    return portfolio_returns.mean(), portfolio_returns.std()

def calculate_historical_var_cvar(returns: pd.Series, confidence_level: float) -> Tuple[float, float]:
    """Calculate historical VaR and CVaR."""
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def monte_carlo_simulation(mean_return: float, volatility: float, 
                         n_simulations: int = 1000, n_days: int = 252) -> np.ndarray:
    """Perform Monte Carlo simulation with GARCH dynamics."""
    simulated_returns = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        # Generate random shocks
        shocks = np.random.normal(0, 1, n_days)
        # Simulate returns with GARCH dynamics
        for t in range(1, n_days):
            simulated_returns[i, t] = mean_return + volatility * shocks[t]
    return simulated_returns

def fit_garch_model(returns: pd.Series) -> Tuple[arch_model, float]:
    """Fit GARCH model and return the model and predicted volatility."""
    garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
    garch_res = garch_model.fit(disp='off')
    forecast = garch_res.forecast(horizon=1)
    predicted_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100
    return garch_res, predicted_vol

def plot_results(portfolio_returns: pd.Series, var_hist: float, var_garch: float, 
                confidence_level: float) -> None:
    """Plot the distribution of portfolio returns with VaR estimates."""
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy array to avoid pandas options issue
    returns_array = portfolio_returns.values
    
    # Plot histogram using matplotlib directly
    plt.hist(returns_array, bins=100, density=True, alpha=0.6, color='skyblue')
    
    # Add KDE using scipy
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(returns_array)
    x = np.linspace(returns_array.min(), returns_array.max(), 1000)
    plt.plot(x, kde(x), color='blue', linewidth=2)
    
    # Add VaR lines
    plt.axvline(var_hist, color='red', linestyle='--', 
                label=f'Historical VaR ({var_hist:.2%})')
    plt.axvline(var_garch, color='purple', linestyle='--', 
                label=f'GARCH VaR ({var_garch:.2%})')
    
    plt.title(f"Portfolio Daily Returns with {confidence_level*100:.0f}% VaR Estimates")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # --- Configuration ---
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    confidence_level = 0.95
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    n_simulations = 1000
    n_days = 252

    try:
        # Download and process data
        print("Downloading data...")
        price_data = pd.DataFrame()
        
        for ticker in tickers:
            print(f"Downloading {ticker}...")
            df_ticker = alpaca_data(ticker, start_date, end_date, "day", "split")
            if df_ticker.empty:
                print(f"Warning: No data returned for {ticker}")
                continue
                
            # Reset index to get timestamp as a column
            df_ticker = df_ticker.reset_index()
            
            # Use only the close price and timestamp
            if price_data.empty:
                price_data = pd.DataFrame({
                    'timestamp': df_ticker['timestamp'],
                    ticker: df_ticker['close']
                })
            else:
                # Merge on timestamp to ensure proper alignment
                temp_df = pd.DataFrame({
                    'timestamp': df_ticker['timestamp'],
                    ticker: df_ticker['close']
                })
                price_data = pd.merge(price_data, temp_df, on='timestamp', how='outer')

        if price_data.empty:
            raise ValueError("No data was successfully downloaded for any ticker")

        # Sort by timestamp and set it as index
        price_data = price_data.sort_values('timestamp').set_index('timestamp')

        print("\nData Summary:")
        print(f"Number of rows: {len(price_data)}")
        print(f"Columns: {price_data.columns.tolist()}")
        print(f"Missing values:\n{price_data.isnull().sum()}")

        # Forward fill missing values
        price_data = price_data.ffill()
        # Backward fill any remaining missing values
        price_data = price_data.bfill()

        # Calculate returns
        returns = price_data.pct_change().dropna()
        if returns.empty:
            raise ValueError("No valid returns data after calculating percentage changes")

        print("\nReturns Summary:")
        print(f"Number of rows: {len(returns)}")
        print(f"Columns: {returns.columns.tolist()}")
        print(f"Missing values:\n{returns.isnull().sum()}")

        # Verify shapes match
        if len(returns.columns) != len(weights):
            raise ValueError(f"Shape mismatch: returns has {len(returns.columns)} columns but weights has {len(weights)} elements")

        portfolio_returns = returns.dot(weights)

        # Calculate basic metrics
        mean_return, volatility = calculate_portfolio_metrics(returns, weights)
        
        # Calculate historical VaR and CVaR
        var_hist, cvar_hist = calculate_historical_var_cvar(portfolio_returns, confidence_level)

        # Monte Carlo simulation
        simulated_returns = monte_carlo_simulation(mean_return, volatility, n_simulations, n_days)
        var_monte = np.percentile(simulated_returns[:, -1], (1 - confidence_level) * 100)

        # GARCH model
        garch_res, predicted_vol = fit_garch_model(portfolio_returns)
        z_score = norm.ppf(1 - confidence_level)
        var_garch = mean_return + z_score * predicted_vol

        # Print results
        print("\nPortfolio Risk Summary")
        print("-----------------------")
        print(f"Mean Daily Return: {mean_return:.5f}")
        print(f"Volatility (Std Dev): {volatility:.5f}")
        print(f"{int(confidence_level*100)}% Historical VaR: {var_hist:.5f}")
        print(f"{int(confidence_level*100)}% Historical CVaR: {cvar_hist:.5f}")
        print(f"{int(confidence_level*100)}% Monte Carlo VaR: {var_monte:.5f}")
        print(f"{int(confidence_level*100)}% GARCH-based VaR: {var_garch:.5f}")

        # Plot results
        plot_results(portfolio_returns, var_hist, var_garch, confidence_level)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
