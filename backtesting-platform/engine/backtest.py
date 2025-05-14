import pandas as pd
from engine.trade_logger import log_trades
from engine.metrics import calculate_metrics

from backtesting import Backtest
from strategies.rsi_macd_bt import RSIMACDStrategy

def run_backtest(ticker):
    df = pd.read_csv(f"data/prices/{ticker}.csv", index_col=0, parse_dates=True)
    df.columns = [col.lower() for col in df.columns]  # ensure lowercase
    bt = Backtest(df, RSIMACDStrategy, cash=10_000, commission=0.002)
    stats = bt.run()
    bt.plot()
    return stats

def run_custom_backtest(ticker, strategy_fn):
    df = pd.read_csv(f"data/prices/{ticker}.csv", index_col=0, parse_dates=True)
    df = strategy_fn(df)

    trades = df[df["position"].diff() != 0]
    metrics = calculate_metrics(df)
    log_trades(ticker, trades)
    return metrics

