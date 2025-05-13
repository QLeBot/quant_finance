# Backtesting Platform

A modular backtesting system for financial strategies using Airflow, dbt, and Streamlit.

Tech Stack

- Python: Write your backtesting engine (or use Backtrader).

- Airflow: Orchestrate runs (different tickers, strategies, timeframes).

- Kafka: Stream trade events or logs.

- FastAPI: Optional registry to store & serve strategy metadata (name, params, creation date, status).

- dbt: Aggregate trade logs into performance summaries.

- DuckDB or Postgres: Final data warehouse.

- Streamlit: Visualize strategy returns vs. SPY, Sharpe ratio, drawdowns.

backtesting-platform/
│
├── dags/                      # Airflow DAGs
│   └── strategy_runner.py
│
├── strategies/                # User-defined strategy logic
│   ├── __init__.py
│   ├── rsi_macd.py
│   └── moving_average.py
│
├── engine/                    # Core backtesting engine
│   ├── __init__.py
│   ├── backtest.py
│   ├── trade_logger.py
│   └── metrics.py
│
├── api/                       # FastAPI registry (optional)
│   └── app.py
│
├── dbt_project/               # dbt transformation models
│   ├── models/
│   │   ├── staging/
│   │   └── marts/
│   └── dbt_project.yml
│
├── data/                      # Data storage or DuckDB files
│   └── prices/
│
├── airflow/                   # Airflow configs
│   └── docker-compose.yml
│
├── dashboard/                 # Streamlit app
│   └── app.py
│
├── requirements.txt
└── README.md


docker-compose.yml
This file includes:
- PostgreSQL as the Airflow metadata backend
- Airflow Webserver (UI on localhost:8080)
- Airflow Scheduler
- Volume mapping for ./dags/ to load your DAGs

cd backtesting-platform
docker-compose up airflow-webserver airflow-scheduler
