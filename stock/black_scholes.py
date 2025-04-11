"""
Black-Scholes Option Pricing Model

The Black-Scholes model is a mathematical model used for pricing call and put options. 
It provides a theoretical estimate of the price of options based on various parameters.

Key Components:
- Call Option: Gives the holder the right, but not the obligation, to buy an asset at a specified price (strike price) on or before a specified date.
- Put Option: Gives the holder the right, but not the obligation, to sell an asset at a specified price on or before a specified date.

Parameters:
- S (Spot Price / Current Stock Price): The current price of the underlying stock.
- X (Strike Price): The price at which the option can be exercised.
- T (Time to Expiration): The time remaining until the option's expiration date, expressed in years.
- r (Risk-Free Rate): The annualized risk-free interest rate, typically the yield on government bonds.
- sigma (Volatility): The annualized volatility of the stock's returns, representing the degree of variation in the stock price.

The strike price (X) in options trading is typically a predefined value set by the options exchange or the parties involved in the contract. 
It represents the price at which the option holder can buy (call option) or sell (put option) the underlying asset. 
The strike price is not calculated based on the current price; rather, it is chosen based on the trader's strategy and market expectations.

For simulation purposes, we can define the strike price based on the current price:
1. At-the-Money (ATM): Set the strike price equal to the current stock price.
2. In-the-Money (ITM): Set the strike price below the current stock price for call options or above for put options.
3. Out-of-the-Money (OTM): Set the strike price above the current stock price for call options or below for put options.

The risk-free rate is a theoretical interest rate that represents the return on an investment with zero risk. It is not about the percentage of risk a trader is willing to take; rather, it is linked to the global interest rate environment.
Key Points about the Risk-Free Rate:
1. Definition: The risk-free rate is the rate of return on an investment that is considered free of risk. In practice, it is often represented by the yield on government bonds, such as U.S. Treasury bills, which are considered to have negligible default risk.
2. Purpose in Models: In financial models like the Black-Scholes option pricing model, the risk-free rate is used to discount future cash flows to their present value. It reflects the time value of money, which is the idea that a dollar today is worth more than a dollar in the future due to its potential earning capacity.
3. Global Interest Rates: The risk-free rate is influenced by the central bank's monetary policy and the overall economic environment. For example, when central banks set low interest rates to stimulate the economy, the risk-free rate will also be low.
4. Practical Use: In practice, the risk-free rate is often approximated using the yield on short-term government securities, such as the 3-month U.S. Treasury bill, because they are highly liquid and have a short maturity, minimizing interest rate risk.
5. Role in Option Pricing: In the Black-Scholes model, the risk-free rate is used to calculate the present value of the option's strike price and to model the expected growth rate of the underlying asset in a risk-neutral world.
Example:
If the current yield on a 3-month U.S. Treasury bill is 0.5%, this would be used as the risk-free rate in financial models. It represents the return an investor would expect from an absolutely risk-free investment over that period.
In summary, the risk-free rate is a fundamental component in financial modeling, reflecting the cost of capital and the time value of money in a risk-neutral context.

Formulas:
- d1 = (ln(S / X) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
- d2 = d1 - sigma * sqrt(T)
- Call Price = S * N(d1) - X * exp(-r * T) * N(d2)
- Put Price = X * exp(-r * T) * N(-d2) - S * N(-d1)

Where N(d) is the cumulative distribution function of the standard normal distribution.

In this implementation, we use the Alpaca API to fetch the current stock prices for a list of symbols. 
The option prices are then calculated using the Black-Scholes formula for each stock.

Note: The Alpaca API has a limit on historical data, so we fetch data from 365 days ago to 5 days ago.
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

