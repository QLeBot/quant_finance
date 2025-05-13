import pandas as pd
import ta

def run_strategy(data):
    data["rsi"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["macd"] = ta.trend.MACD(data["Close"]).macd()
    
    buy = (data["rsi"] < 30) & (data["macd"] > 0)
    sell = (data["rsi"] > 70) & (data["macd"] < 0)
    
    data["position"] = 0
    data.loc[buy, "position"] = 1
    data.loc[sell, "position"] = -1
    data["returns"] = data["Close"].pct_change() * data["position"].shift(1)
    return data
