import truststore
truststore.inject_into_ssl()

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pytz

from ta.momentum import RSIIndicator
from ta.trend import MACD
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
#symbols = ["NVDA", "SPY", "TSLA", "MC.PA", "ATO", "ATOS"]
#symbols = ["NVDA"]
#symbols = ["NVDA", "ATOS"]
#symbols = ["NVDA", "ATOS" , "ATO"]
symbols = ["SPY"]
initial_cash = 10000

start_date = datetime.datetime(2000, 1, 1)
#end_date = datetime.datetime(2024, 12, 31)
end_date = datetime.datetime.now() - datetime.timedelta(days=2)

# Request 1-hour data for primary analysis and 4-hour data for confirmation
request_params_1h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_1h = stock_client.get_stock_bars(request_params_1h).df
all_data_1h = all_data_1h.reset_index()  # bring 'symbol' and 'timestamp' into columns

# plot the data for each symbol
for symbol in symbols:
    plt.figure(figsize=(15, 10))
    plt.plot(all_data_1h[all_data_1h['symbol'] == symbol]['timestamp'], all_data_1h[all_data_1h['symbol'] == symbol]['close'], label='Close')
    plt.legend()
    plt.show()