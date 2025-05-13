import sqlite3
import pandas as pd
import streamlit as st

conn = sqlite3.connect("data/backtest.db")
df = pd.read_sql("SELECT * FROM trades", conn)
st.title("Backtest Dashboard")
st.line_chart(df.set_index("Date")["position"])
st.write(df.tail())
