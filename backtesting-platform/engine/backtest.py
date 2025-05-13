import pandas as pd
from engine.trade_logger import log_trades
from engine.metrics import calculate_metrics

def run_backtest(ticker, strategy_fn):
    df = pd.read_csv(f"data/prices/{ticker}.csv", index_col=0, parse_dates=True)
    df = strategy_fn(df)

    trades = df[df["position"].diff() != 0]
    metrics = calculate_metrics(df)
    log_trades(ticker, trades)
    return metrics
