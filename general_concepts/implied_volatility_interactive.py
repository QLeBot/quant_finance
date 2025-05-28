import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import yfinance as yf
from datetime import datetime
import streamlit as st

st.title("Implied Volatility Calculator")

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
    # Input validation
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
        
    try:
        d1 = (np.log(S/K) + (r + ((sigma**2)/2)) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    except (ValueError, ZeroDivisionError):
        return np.nan
    
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

def get_stock_data(ticker):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice', None)
        if current_price is None:
            st.error(f"Could not fetch current price for {ticker}")
            return None
        return current_price
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def create_iv_surface(S, K_range, T_range, r, q, option_type='call'):
    """Create implied volatility surface data"""
    iv_surface = np.zeros((len(K_range), len(T_range)))
    
    for i, K in enumerate(K_range):
        for j, T in enumerate(T_range):
            # Use a reasonable market price estimate for demonstration
            # In practice, you would use real market prices
            market_price = black_scholes(S, K, T, r, 0.3, option_type)  # Using 30% vol as initial guess
            iv = implied_volatility(market_price, S, K, T, r, option_type)
            iv_surface[i, j] = iv
            
    return iv_surface

# Streamlit interface
st.sidebar.header("Parameters")

st.sidebar.subheader("Parameters for Black-Scholes Model")
r = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0) / 100
q = st.sidebar.number_input("Dividend Yield (%)", min_value=0.0, max_value=10.0, value=0.0) / 100

# Visualization parameters
st.sidebar.subheader("Visualization Parameters")
y_axis_label = st.sidebar.selectbox("Y-Axis Label", ["Strike Price", "Moneyness"])

# Ticker parameters
st.sidebar.subheader("Ticker Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", "SPY")

# Strike Price Filter parameters
st.sidebar.subheader("Strike Price Filter")
min_strike = st.sidebar.number_input("Min Strike Price (% of Spot Price)", min_value=0.0, value=70)
max_strike = st.sidebar.number_input("Max Strike Price (% of Spot Price)", min_value=0.0, value=130)




# Get current stock price
current_price = get_stock_data(ticker)
if current_price:
    st.sidebar.write(f"Current Price: ${current_price:.2f}")
    
    # Strike price range
    min_strike = st.sidebar.number_input("Min Strike Price", min_value=0.0, value=current_price * 0.7)
    max_strike = st.sidebar.number_input("Max Strike Price", min_value=0.0, value=current_price * 1.3)
    
    # Time to maturity range
    min_T = st.sidebar.number_input("Min Time to Maturity (years)", min_value=0.01, value=0.1)
    max_T = st.sidebar.number_input("Max Time to Maturity (years)", min_value=0.01, value=1.0)
    
    # Option type selection
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    
    # Create strike and time ranges
    K_range = np.linspace(min_strike, max_strike, 20)
    T_range = np.linspace(min_T, max_T, 20)
    
    # Create the IV surface
    iv_surface = create_iv_surface(current_price, K_range, T_range, r, q, option_type)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plotting
    K_mesh, T_mesh = np.meshgrid(K_range, T_range)
    
    # Plot the surface
    surf = ax.plot_surface(K_mesh, T_mesh, iv_surface.T, cmap='viridis', edgecolor='none')
    
    # Customize the plot
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Implied Volatility Surface - {ticker} {option_type.upper()}')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.write("Implied Volatility Surface Data:")
        st.dataframe(iv_surface)






