import truststore
truststore.inject_into_ssl()

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

import yfinance as yf

tickers = ['SPY']
for ticker in tickers:
    ticker_yahoo = yf.Ticker(ticker)
    data = ticker_yahoo.history()
    last_quote = data['Close'].iloc[-1]
    print(ticker, last_quote)