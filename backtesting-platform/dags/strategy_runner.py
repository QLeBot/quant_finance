from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from strategies import rsi_macd
from engine.backtest import run_backtest

def run():
    run_backtest("AAPL", rsi_macd.run_strategy)

with DAG("run_strategy", start_date=datetime(2023,1,1), schedule_interval="@daily", catchup=False) as dag:
    run_task = PythonOperator(
        task_id="backtest_strategy",
        python_callable=run
    )
