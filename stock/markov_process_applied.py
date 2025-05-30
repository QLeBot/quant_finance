import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from backtesting import Backtest, Strategy

# import get_data from get_data.py
from get_data import alpaca_data

# Parameters
symbol = "SPY"
initial_cash = 10000
start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

def determine_state(df):
    """Determine the state of the market based on the data."""
    # Calculate the daily returns
    df['daily_return'] = df['close'].pct_change()

    # drop the first row where the daily return is NaN
    df = df.iloc[1:].copy()  # Create a copy to avoid SettingWithCopyWarning

    df.loc[:, 'state'] = np.where(df['daily_return'] >= 0, 'up', 'down')

    return df

def get_transition_matrix(df):
    """Get the transition matrix from the data."""
    # Get the states
    up_count = len(df[df['state'] == 'up'])
    down_count = len(df[df['state'] == 'down'])

    up_to_up = len(df[(df['state'] == 'up') & (df['state'].shift(1) == 'up')]) / up_count
    up_to_down = len(df[(df['state'] == 'up') & (df['state'].shift(1) == 'down')]) / up_count
    down_to_up = len(df[(df['state'] == 'down') & (df['state'].shift(1) == 'up')]) / down_count
    down_to_down = len(df[(df['state'] == 'down') & (df['state'].shift(1) == 'down')]) / down_count

    # Create a transition matrix
    transition_matrix = pd.DataFrame({
        'up': [up_to_up, up_to_down],
        'down': [down_to_up, down_to_down]
    }, index=['up', 'down'])

    return transition_matrix

def prepare_data_for_backtesting(df):
    """Prepare data for backtesting with proper column names and index."""
    # Reset index to get timestamp as a column
    df = df.reset_index()
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Ensure we have the required columns with correct names
    return pd.DataFrame({
        'Open': df['open'],
        'High': df['high'],
        'Low': df['low'],
        'Close': df['close'],
        'Volume': df['volume']
    })

class MarkovProcess(Strategy):
    def init(self):
        df = alpaca_data(symbol, start_date, end_date, "day", "split")
        state_df = determine_state(df)
        self.transition_matrix = get_transition_matrix(state_df)
    
    def next(self):
        if (self.data.Close < self.data.Close[-1] and 
            self.data.Close[-1] < self.data.Close[-2] and 
            self.data.Close[-2] < self.data.Close[-3] and 
            self.data.Close[-3] < self.data.Close[-4] and 
            self.data.Close[-4] < self.data.Close[-5]):
            self.buy()
        elif self.data.Close > self.data.Close[-1]:
            self.sell()

# get data
df = alpaca_data(symbol, start_date, end_date, "day", "split")
# prepare data for backtesting
prepared_df = prepare_data_for_backtesting(df)
print(prepared_df)

transition_matrix = get_transition_matrix(df)
print(transition_matrix)

# probability of up after 5 consecutive down days
print(len(df[(df['state'] == 'up') & (df['state'].shift(-1) == 'down') & (df['state'].shift(-2) == 'down') & (df['state'].shift(-3) == 'down') & (df['state'].shift(-4) == 'down') & (df['state'].shift(-5) == 'down')]) / len(df[(df["state"].shift(1) == "down") & (df["state"].shift(2) == "down") & (df["state"].shift(3) == "down") & (df["state"].shift(4) == "down") & (df["state"].shift(5) == "down")]))

# run backtest
bt = Backtest(prepared_df, MarkovProcess, cash=initial_cash)
output = bt.run()
print(output)

# plot the results
bt.plot()