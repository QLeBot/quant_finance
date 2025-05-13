import sqlite3

def log_trades(ticker, df_trades):
    conn = sqlite3.connect("data/backtest.db")
    df_trades["ticker"] = ticker
    df_trades.to_sql("trades", conn, if_exists="append", index=False)
    conn.close()
