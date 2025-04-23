import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import yfinance as yf
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D

def get_risk_free_rate_tnx():
    """Fetches the latest U.S. 10-Year Treasury yield using yfinance."""
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100
        return latest_yield
    else:
        raise ValueError("No data found for ^TNX.")

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price
    S: current stock price
    K: strike price
    T: time to maturity (in years)
    r: risk-free rate
    sigma: volatility
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility using Brent's method
    """
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - price
    
    try:
        return brentq(objective, 0.0001, 10.0)
    except ValueError:
        return np.nan

def plot_implied_volatility_surface_3d(ticker, max_years=3):
    """
    Plot 3D implied volatility surface for a given stock using moneyness (S/K)
    max_years: maximum time to expiration in years
    """
    # Get stock data
    stock = yf.Ticker(ticker)
    current_price = stock.history(period='1d')['Close'].iloc[-1]
    
    # Get all expiration dates
    all_expirations = stock.options
    
    # Filter expiration dates within max_years
    current_date = datetime.now()
    valid_expirations = []
    for exp_date in all_expirations:
        exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
        time_to_exp = (exp_datetime - current_date).days / 365
        if time_to_exp <= max_years:
            valid_expirations.append(exp_date)
    
    # Initialize arrays for 3D plot
    all_moneyness = []
    all_times = []
    all_ivs = []
    
    r = get_risk_free_rate_tnx()
    
    for expiration_date in valid_expirations:
        options = stock.option_chain(expiration_date)
        calls = options.calls
        puts = options.puts
        
        # Calculate time to expiration in years
        T = (datetime.strptime(expiration_date, '%Y-%m-%d') - current_date).days / 365
        
        # Process calls
        for _, row in calls.iterrows():
            if not np.isnan(row['lastPrice']):
                iv = implied_volatility(row['lastPrice'], current_price, row['strike'], T, r, 'call')
                if not np.isnan(iv):
                    moneyness = current_price / row['strike']
                    all_moneyness.append(moneyness)
                    all_times.append(T)
                    all_ivs.append(iv)
        
        # Process puts
        for _, row in puts.iterrows():
            if not np.isnan(row['lastPrice']):
                iv = implied_volatility(row['lastPrice'], current_price, row['strike'], T, r, 'put')
                if not np.isnan(iv):
                    moneyness = current_price / row['strike']
                    all_moneyness.append(moneyness)
                    all_times.append(T)
                    all_ivs.append(iv)
    
    # Convert to numpy arrays
    moneyness = np.array(all_moneyness)
    times = np.array(all_times)
    ivs = np.array(all_ivs)
    
    # Create grid for surface plot
    moneyness_min, moneyness_max = min(moneyness), max(moneyness)
    time_min, time_max = 0, max_years  # Set time range from 0 to max_years
    
    # Create a grid of points
    grid_moneyness = np.linspace(moneyness_min, moneyness_max, 100)
    grid_times = np.linspace(time_min, time_max, 100)
    grid_moneyness, grid_times = np.meshgrid(grid_moneyness, grid_times)
    
    # Interpolate the data onto the grid
    grid_ivs = griddata(
        (moneyness, times), 
        ivs, 
        (grid_moneyness, grid_times), 
        method='cubic'
    )
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(
        grid_moneyness, 
        grid_times, 
        grid_ivs, 
        cmap='viridis',
        alpha=0.8,
        linewidth=0,
        antialiased=True
    )
    
    # Add scatter plot of original points
    ax.scatter(moneyness, times, ivs, color='black', alpha=0.3, s=10)
    
    # Add colorbar
    cbar = plt.colorbar(surf)
    cbar.set_label('Implied Volatility')
    
    # Add labels
    ax.set_xlabel('Moneyness (M = S/K)')
    ax.set_ylabel('Time to Expiration (T)')
    ax.set_zlabel('Implied Volatility $\sigma$(T, M)')
    
    # Add title
    plt.title(f'Implied Volatility Surface for {ticker} (up to {max_years} years)')
    
    # Add at-the-money line (moneyness = 1)
    x_line = np.array([1.0, 1.0])
    y_line = np.array([time_min, time_max])
    X, Y = np.meshgrid(x_line, y_line)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.3, color='red')
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.show()

if __name__ == "__main__":
    # Example usage with 3 years of expiration dates
    plot_implied_volatility_surface_3d('AAPL', max_years=3)
