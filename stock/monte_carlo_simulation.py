import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca client
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Parameters
symbols = ["NVDA", "SPY", "TSLA", "MC.PA", "ATO", "ATOS"]
#symbols = ["NVDA"]
#symbols = ["NVDA", "ATOS"]
#symbols = ["NVDA", "ATOS" , "ATO"]
#symbols = ["SPY"]
initial_cash = 10000

end_date = datetime.datetime.now() - datetime.timedelta(days=10)
start_date = end_date - datetime.timedelta(days=300)


request_params_day = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_day = stock_client.get_stock_bars(request_params_day).df
all_data_day = all_data_day.reset_index()  # bring 'symbol' and 'timestamp' into columns


# Convert the 'timestamp' column to datetime if not already
all_data_day['timestamp'] = pd.to_datetime(all_data_day['timestamp'])

# Calculate daily returns for each stock
returns = all_data_day.pivot(index='timestamp', columns='symbol', values='close').pct_change()
returns = returns.dropna()

mean_return = returns.mean()
#print(mean_return)

std_return = returns.std()
#print(std_return)

# Calculate the covariance matrix of the daily returns
cov_matrix = returns.cov()

# Calculate the number of trading days based on the returns DataFrame
num_trading_days = len(returns)

# Monte Carlo Simulation
num_simulations = 100
num_days = 100

# Initialize an array to store the final portfolio values
final_values = np.zeros(num_simulations)

weights = np.random.random(len(mean_return))
weights /= weights.sum()

meanM = np.full(shape=(num_trading_days, len(weights)), fill_value=mean_return)
meanM = meanM.T
stdM = np.full(shape=(num_trading_days, len(weights)), fill_value=std_return)
stdM = stdM.T

#print(meanM)
#print(stdM)

portfolio_sims = np.full(shape=(num_trading_days, num_simulations), fill_value=initial_cash)

# Run the Monte Carlo Simulation
for i in range(num_simulations):
    # Initialize the portfolio value
    portfolio_value = initial_cash

    Z = np.random.normal(size=(num_trading_days, len(weights)))
    L = np.linalg.cholesky(cov_matrix)
    daily_returns = meanM + np.inner(L, Z)
    #print(daily_returns)

    portfolio_sims[:, i] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_cash

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.show()
