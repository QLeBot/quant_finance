#from  sqlite4  import  SQLite4
import sqlite3
import pandas as pd
import streamlit as st
import os

DB_PATH = os.path.join("data", "backtest.db")
print(DB_PATH)

con = sqlite3.connect(DB_PATH)

df = pd.read_sql("SELECT * FROM trades", con)
st.title("Backtest Dashboard")
st.line_chart(df.set_index("Date")["position"])
st.write(df.tail())
