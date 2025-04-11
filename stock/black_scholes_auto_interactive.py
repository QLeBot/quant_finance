"""
This file takes the code from black_scholes.py and add a streamlit app to it.
In this version the current/drop price and risk-free rate are automatically fetched from alpaca API and calculated.

The risk-free rate is calculated using the 3-month T-bill yield or 10-Year Treasury yield.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import datetime
import yfinance as yf
import streamlit as st

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca client
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Parameters
symbols = ["AAPL", "GOOGL", "AMZN", "TSLA"]
time_to_expiration = 1  # 1 year

def get_risk_free_rate_irx():
    """
    Fetches the latest U.S. 3-month T-bill yield using yfinance.
    """
    index = yf.Ticker("^IRX")
    data = index.history(period="1d")
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100  # convert from percentage
        return latest_yield
    else:
        raise ValueError("No data found for ^IRX.")

def get_risk_free_rate_tnx():
    """
    Fetches the latest U.S. 10-Year Treasury yield using yfinance.
    This is often used as a proxy for the risk-free rate in USD.
    """
    # '^TNX' is the Yahoo Finance symbol for the 10-Year Treasury Note yield (multiplied by 100)
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100  # convert from percentage
        return latest_yield
    else:
        raise ValueError("No data found for ^TNX.")

risk_free_rate_irx = get_risk_free_rate_irx()
print(f"Risk-free rate (3-month T-bill): {risk_free_rate_irx:.4%}")

risk_free_rate_tnx = get_risk_free_rate_tnx()
print(f"Risk-free rate (10Y Treasury): {risk_free_rate_tnx:.4%}")

# set risk-free rate to the desired calculation method
#risk_free_rate = 0.05  # 5% annual risk-free rate
risk_free_rate = risk_free_rate_irx

# Fetch historical stock prices using Alpaca
historical_prices = {}
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=datetime.datetime.now() - datetime.timedelta(days=365),  # 1 year of data
    end=datetime.datetime.now() - datetime.timedelta(days=5)
)
all_data = stock_client.get_stock_bars(request_params).df

# Calculate historical volatility
volatilities = {}
for symbol in symbols:
    df = all_data.loc[symbol].copy()
    df['returns'] = df['close'].pct_change()
    
    volatility = df['returns'].std() * np.sqrt(252)  # Annualize the volatility
    volatilities[symbol] = volatility

# Fetch current stock prices
current_prices = {}
for symbol in symbols:
    current_prices[symbol] = all_data.loc[symbol]['close'].iloc[-1]

# Two method for strike price, 1. fixed strike price, 2. 5% above current price
#strike_price = 100  # Example strike price
strike_prices = {symbol: current_prices[symbol] * 1.05 for symbol in symbols}  # Example: 5% above current price

def black_scholes_call(S, X, T, r, sigma):
    """Calculate the Black-Scholes price for a European call option."""
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def black_scholes_put(S, X, T, r, sigma):
    """Calculate the Black-Scholes price for a European put option."""
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

# Calculate option prices
results = []

for symbol in symbols:
    S = current_prices[symbol]
    X = strike_prices[symbol]
    sigma = volatilities[symbol]
    call_price = black_scholes_call(S, X, time_to_expiration, risk_free_rate, sigma)
    put_price = black_scholes_put(S, X, time_to_expiration, risk_free_rate, sigma)
    results.append({
        "Symbol": symbol,
        "Current Price": round(S, 2),
        "Strike Price": round(X, 2),
        "Call Price": round(call_price, 2),
        "Put Price": round(put_price, 2),
        "Volatility": round(sigma, 4)
    })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print("\n📊 Black-Scholes Option Prices:")
print(results_df)

# Streamlit app
st.title("Black-Scholes Option Pricing Model")

# Input fields for parameters
st.sidebar.header("Input Parameters")

# Input fields for parameters
symbol = st.sidebar.selectbox("Select Stock Symbol", symbols)
#current_price = st.sidebar.number_input("Current Stock Price", value=current_prices[symbol])
strike_price = st.sidebar.number_input("Strike Price", value=strike_prices[symbol])
time_to_expiration = st.sidebar.number_input("Time to Expiration (years)", value=time_to_expiration)
#volatility = st.sidebar.number_input("Volatility (annual)", value=volatilities[symbol])
#risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual)", value=risk_free_rate)

# Calculate option prices
call_price = black_scholes_call(current_prices[symbol], strike_price, time_to_expiration, risk_free_rate, volatility)
put_price = black_scholes_put(current_prices[symbol], strike_price, time_to_expiration, risk_free_rate, volatility)

st.header("Considerations")
st.write("This is an automatic pricer for options pricing based on the Black-Scholes model. The results are for educational purposes only and should not be used for trading or investment decisions.")
st.write("The results are based on the spot price automatically fetched from the Alpaca API. The strike price is 5% above the spot price but can be adjusted by the user.")
st.write("The time to expiration is 1 year but can be adjusted by the user.")
st.write("The volatility is calculated using the annualized volatility of the stock's returns over the past year.")
st.write("The risk-free rate is calculated using the 3-month T-bill yield.")

# Display results
st.header("Parameters")
st.write(f"Current Price: ${current_prices[symbol]:.2f}")
st.write(f"Strike Price: ${strike_price:.2f}")
st.write(f"Time to Expiration: {time_to_expiration:.2f} years")
st.write(f"Volatility: {volatility:.4f}")
st.write(f"Risk-Free Rate: {risk_free_rate:.4%}")

st.header("Call & Put Prices")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div style="background-color:rgb(77, 255, 0); padding: 10px; border-radius: 5px; font-size: 20px; text-align: center;">Call Price: ${call_price:.2f}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div style="background-color:rgb(252, 0, 0); padding: 10px; border-radius: 5px; font-size: 20px; text-align: center;">Put Price: ${put_price:.2f}</div>', unsafe_allow_html=True)

