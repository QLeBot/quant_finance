"""
This file takes the code from black_scholes.py and add a streamlit app to it.
In this version the parameters have all to be set by the user.

It also includes a heatmap of spot price relative to volatility for call and put prices.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import yfinance as yf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from  matplotlib.colors import LinearSegmentedColormap
c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
v = [0,.15,.4,.5,0.6,.9,1.]
l = list(zip(v,c))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

# Parameters
current_price = 100.00
strike_price = 100.00
time_to_expiration = 1.00  # 1 year
risk_free_rate = 0.05
volatility = 0.2

min_spot_price = 0.01
max_spot_price = 200.00
min_volatility = 0.01
max_volatility = 1.00

call_purchase_price = 0.00
put_purchase_price = 0.00
num_contracts = 1

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



# ===== Streamlit App =====
st.set_page_config(layout="wide")

# Streamlit app
st.title("Black-Scholes Option Pricing Model")

# Input fields for parameters
st.sidebar.header("Purchase Price")
call_purchase_price = st.sidebar.number_input("Call Purchase Price", value=0.00, step=0.01)
put_purchase_price = st.sidebar.number_input("Put Purchase Price", value=0.00, step=0.01)
num_contracts = st.sidebar.number_input("Number of Contracts", value=1, step=1)

# Input fields for parameters
st.sidebar.header("Input Parameters")
current_price = st.sidebar.number_input("Current Stock Price", value=current_price, step=0.01)
strike_price = st.sidebar.number_input("Strike Price", value=strike_price, step=0.01)
time_to_expiration = st.sidebar.number_input("Time to Expiration (years)", value=time_to_expiration, step=0.01)
volatility = st.sidebar.number_input("Volatility (annual)", value=volatility, step=0.01)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual)", value=risk_free_rate, step=0.01)

# Input fields for parameters for the heatmap
st.sidebar.header("Input Parameters for Heatmap")
min_spot_price = st.sidebar.number_input("Min Spot Price", value=min_spot_price, step=0.01)
max_spot_price = st.sidebar.number_input("Max Spot Price", value=max_spot_price, step=0.01)
min_volatility = st.sidebar.slider("Min Volatility", value=min_volatility, min_value=0.01, max_value=1.00, step=0.01)
max_volatility = st.sidebar.slider("Max Volatility", value=max_volatility, min_value=0.01, max_value=1.00, step=0.01)

st.header("Call & Put Prices")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div style="background-color:rgb(77, 255, 0); color:black; padding: 10px; border-radius: 10px; font-size: 20px; text-align: center;">Call Price <br>${black_scholes_call(current_price, strike_price, time_to_expiration, risk_free_rate, volatility):.2f}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div style="background-color:rgb(252, 0, 0); color:black; padding: 10px; border-radius: 10px; font-size: 20px; text-align: center;">Put Price <br>${black_scholes_put(current_price, strike_price, time_to_expiration, risk_free_rate, volatility):.2f}</div>', unsafe_allow_html=True)

#call_pnl_price = (black_scholes_call(current_price, strike_price, time_to_expiration, risk_free_rate, volatility) - call_purchase_price) * num_contracts
#put_pnl_price = (black_scholes_put(current_price, strike_price, time_to_expiration, risk_free_rate, volatility) - put_purchase_price) * num_contracts

# Display PnL
st.header("Profit and Loss (PnL)")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div style="background-color:rgb(77, 255, 0); color:black; padding: 10px; border-radius: 10px; font-size: 20px; text-align: center;">Call PnL <br>${(black_scholes_call(current_price, strike_price, time_to_expiration, risk_free_rate, volatility) - call_purchase_price) * num_contracts:.2f}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div style="background-color:rgb(252, 0, 0); color:black; padding: 10px; border-radius: 10px; font-size: 20px; text-align: center;">Put PnL <br>${(black_scholes_put(current_price, strike_price, time_to_expiration, risk_free_rate, volatility) - put_purchase_price) * num_contracts:.2f}</div>', unsafe_allow_html=True)


# ===== Call and Put Prices =====
spot_price_range = np.linspace(min_spot_price, max_spot_price, 10)  # Example range
volatility_range = np.linspace(min_volatility, max_volatility, 10)

call_price_matrix = np.zeros((len(volatility_range), len(spot_price_range)))
put_price_matrix = np.zeros((len(volatility_range), len(spot_price_range)))

for i, vol in enumerate(volatility_range):
    for j, sp in enumerate(spot_price_range):
        call_price_matrix[i, j] = black_scholes_call(sp, strike_price, time_to_expiration, risk_free_rate, vol)
        put_price_matrix[i, j] = black_scholes_put(sp, strike_price, time_to_expiration, risk_free_rate, vol)

fig_call, ax_call = plt.subplots(figsize=(12, 8))
sns.heatmap(call_price_matrix, ax=ax_call, cmap=cmap, xticklabels=np.round(spot_price_range, 2), yticklabels=np.round(volatility_range, 2), cbar_kws={'label': 'Call Price'}, annot=True, fmt='.2f', square=True)
ax_call.set_title('CALL')
ax_call.set_xlabel('Spot Price')
ax_call.set_ylabel('Volatility')

fig_put, ax_put = plt.subplots(figsize=(12, 8))
sns.heatmap(put_price_matrix, ax=ax_put, cmap=cmap, xticklabels=np.round(spot_price_range, 2), yticklabels=np.round(volatility_range, 2), cbar_kws={'label': 'Put Price'}, annot=True, fmt='.2f', square=True)
ax_put.set_title('PUT')
ax_put.set_xlabel('Spot Price')
ax_put.set_ylabel('Volatility')

# ===== PnL =====
#call_pnl_price = (black_scholes_call(current_price, strike_price, time_to_expiration, risk_free_rate, volatility) - call_purchase_price) * num_contracts
#put_pnl_price = (black_scholes_put(current_price, strike_price, time_to_expiration, risk_free_rate, volatility) - put_purchase_price) * num_contracts

call_pnl_matrix = np.zeros((len(volatility_range), len(spot_price_range)))
put_pnl_matrix = np.zeros((len(volatility_range), len(spot_price_range)))

for i, vol in enumerate(volatility_range):
    for j, sp in enumerate(spot_price_range):
        call_pnl_matrix[i, j] = (black_scholes_call(sp, strike_price, time_to_expiration, risk_free_rate, vol) - call_purchase_price) * num_contracts
        put_pnl_matrix[i, j] = (black_scholes_put(sp, strike_price, time_to_expiration, risk_free_rate, vol) - put_purchase_price) * num_contracts 

# Heatmap of call and put PnL
fig_call_pnl, ax_call_pnl = plt.subplots(figsize=(12, 8))
sns.heatmap(call_pnl_matrix, ax=ax_call_pnl, cmap=cmap, xticklabels=np.round(spot_price_range, 2), yticklabels=np.round(volatility_range, 2), cbar_kws={'label': 'Call PnL'}, annot=True, fmt='.2f', square=True)
ax_call_pnl.set_title('CALL PnL')
ax_call_pnl.set_xlabel('Spot Price')
ax_call_pnl.set_ylabel('Volatility')

fig_put_pnl, ax_put_pnl = plt.subplots(figsize=(12, 8))
sns.heatmap(put_pnl_matrix, ax=ax_put_pnl, cmap=cmap, xticklabels=np.round(spot_price_range, 2), yticklabels=np.round(volatility_range, 2), cbar_kws={'label': 'Put PnL'}, annot=True, fmt='.2f', square=True)
ax_put_pnl.set_title('PUT PnL')
ax_put_pnl.set_xlabel('Spot Price')
ax_put_pnl.set_ylabel('Volatility')


# Display heatmap of call and put prices
st.header("Heatmap of Call and Put Prices")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_call)
with col2:
    st.pyplot(fig_put)

# Display heatmap of call and put PnL
st.header("Heatmap of Call and Put PnL")
col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig_call_pnl)
with col4:
    st.pyplot(fig_put_pnl)