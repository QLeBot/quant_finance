#import yahoo_fin.stock_info as si
#import yahoo_fin.options as op

# import from local folder for debugging
import yahoo_fin_master.yahoo_fin.stock_info as si

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests_html
from datetime import datetime
import feedparser
import ftplib
import io
import json
import requests

#st.title("Fundamental Stock Data")

ticker = 'aapl'

# get balance sheets, cash flow statements, and income statements in a single call
#financials = si.get_financials(ticker, yearly = False, quarterly = True)
# or
balance_sheet = si.get_balance_sheet(ticker, yearly = False)
#income_statement = si.get_income_statement(ticker, yearly = False)
#cash_flow = si.get_cash_flow(ticker, yearly = False)

# get earnings data
#earnings = si.get_earnings(ticker)

print(balance_sheet)




