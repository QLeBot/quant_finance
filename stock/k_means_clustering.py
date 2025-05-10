import numpy as np 
import pandas as pd
import pandas_datareader as dr
import yfinance as yf
import datetime

from pylab import plot,show
from matplotlib import pyplot as plt
import plotly.express as px

from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans 
from sklearn import preprocessing

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)
end_date = datetime.datetime.now() - datetime.timedelta(days=10)
start_date = end_date - datetime.timedelta(days=300)

def get_sp500_tickers():
    # Define the url
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Read in the url and scrape ticker data
    data_table = pd.read_html(sp500_url)
    tickers = data_table[0]['Symbol'].values.tolist()
    tickers = [s.replace('\n', '') for s in tickers]
    tickers = [s.replace('.', '-') for s in tickers]
    tickers = [s.replace(' ', '') for s in tickers]
    return tickers

#tickers = get_sp500_tickers()

def get_stock_data(tickers, start_date, end_date):
    all_data_day = pd.DataFrame()

    for ticker in tickers:
        # download the data from alpaca
        request_params_day = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            adjustment="split"
        )
        # modify the get_stock_bars to handle the case where the ticker is not found
        try:
            ticker_data_day = stock_client.get_stock_bars(request_params_day).df
            ticker_data_day = ticker_data_day.reset_index()  # bring 'symbol' and 'timestamp' into columns
        except Exception as e:
            print(f"Error: {e}")
            print(f"Ticker not found: {ticker}")
        # merge the data into a single dataframe
        all_data_day = pd.concat([all_data_day, ticker_data_day])
    
    all_data_day['timestamp'] = pd.to_datetime(all_data_day['timestamp'])
    return all_data_day

#all_data_day_df = get_stock_data(tickers, start_date, end_date)

# export the data to a csv file
#all_data_day.to_csv('stock/csv/all_data_day.csv', index=False)

# read the data from the csv file
all_data_day_df = pd.read_csv('stock/csv/all_data_day.csv')

# Create an empity dataframe
returns = pd.DataFrame()

price_df = all_data_day_df.groupby(['timestamp', 'symbol'])['close'].mean().unstack()
daily_returns = price_df.pct_change()
returns['Returns'] = daily_returns.mean() * 252
returns['Volatility'] = daily_returns.std() * np.sqrt(252)
returns = returns.dropna()

# Format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
fig = plt.figure(figsize=(15, 5))

plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()

# Computing K-Means with K = 4 (4 clusters)
centroids,_ = kmeans(data,4)

# Assign each sample to a cluster
idx,_ = vq(data,centroids)

# Create a dataframe with the tickers and the clusters that's belong to
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
details_df = pd.DataFrame(details)

# Rename columns
details_df.columns = ['Ticker','Cluster']

# Create another dataframe with the tickers and data from each stock
clusters_df = returns.reset_index()

# Bring the clusters information from the dataframe 'details_df'
clusters_df['Cluster'] = details_df['Cluster']

# Rename columns
clusters_df.columns = ['Ticker', 'Returns', 'Volatility', 'Cluster']

# Plot the clusters created using Plotly
fig = px.scatter(clusters_df, x="Returns", y="Volatility", color="Cluster", hover_data=["Ticker"])
fig.update(layout_coloraxis_showscale=False)
fig.show()

def get_outliers():
    # Find the cluster centers
    cluster_centers = k_means.cluster_centers_
    # Calculate the distance from each point to its assigned cluster center
    distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(X, idx)]

    # Define a threshold for anomaly detection (e.g., based on the distance percentile)
    percentile_threshold = 98
    threshold_distance = np.percentile(distances, percentile_threshold)

    # Identify anomalies
    anomalies = [X[i] for i, distance in enumerate(distances) if distance > threshold_distance]
    anomalies = np.asarray(anomalies, dtype=np.float32)

    # Get the tickers that are anomalies
    anomalies_tickers = [returns.index[i] for i, distance in enumerate(distances) if distance > threshold_distance]
    return anomalies_tickers

outliers = get_outliers()
# show outliers
#print(outliers)

for ticker in outliers:
    returns.drop(ticker,inplace=True)

#returns.drop('MRNA',inplace=True)
#returns.drop('ENPH',inplace=True)
#returns.drop('TSLA',inplace=True)
#returns.drop('CEG',inplace=True)

# Recreate data to feed into the algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

# Computing K-Means with K = 4 (4 clusters)
centroids,_ = kmeans(data,4)

# Assign each sample to a cluster
idx,_ = vq(data,centroids)

# Create a dataframe with the tickers and the clusters that's belong to
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
details_df = pd.DataFrame(details)

# Rename columns
details_df.columns = ['Ticker','Cluster']

# Create another dataframe with the tickers and data from each stock
clusters_df = returns.reset_index()

# Bring the clusters information from the dataframe 'details_df'
clusters_df['Cluster'] = details_df['Cluster']

# Rename columns
clusters_df.columns = ['Ticker', 'Returns', 'Volatility', 'Cluster']

# Plot the clusters created using Plotly
fig = px.scatter(clusters_df, x="Returns", y="Volatility", color="Cluster", hover_data=["Ticker"])
#plt.scatter(anomalies[:, 0], anomalies[:, 1], color='purple', marker='.', s=50, label='Anomalies')
fig.update(layout_coloraxis_showscale=False)
fig.show()