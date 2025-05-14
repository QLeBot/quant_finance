# 📈 Quant Finance Lab – Strategy Backtesting & Portfolio Optimization

This repository contains exploratory projects in **quantitative finance using Python**. The goal is to apply core data science and programming skills to financial problems: algorithmic trading, option pricing and portfolio optimization.

> :warning: **Warning:** These projects and the code associated are for research and educational purposes only. It is still an ongoing WIP and therefore should not be used elseways.

## 🧠 Projects Overview

### 1. 📊 Interactive Option Pricer – Black-Scholes
Built an UI option pricer based on Black-Scholes model

#### Key Features:
- Simple Streamlit visualization dashboard
- Fully user input for spot price, volatility, risk free rate, time to expiration, etc
- Returns CALL and PUT prices and both P&L
- Adjustments for generating heatmaps of both prices and P&L

📁 Project Files: `stock/black_scholes_interactive.py`

Run the script and open the link in your browser:
```bash
streamlit run stock/black_scholes_interactive.py
```

#### Notes:
- Created an alternate version with calculated parameters such as risk free rate based on treasury bonds yield, stock price, volatility. It can be found in `black_scholes_auto_interactive.py`

---

### 2. 📊 Interactive Fund Dashboard
Built a dashboard to visualize the state and simulating scenarios of the fund using Streamlit. The dashboard includes various metrics and visualizations to analyze the fund's cash flow, asset allocation, and simulation of scenarios for the fund's performance.

#### Key Features:
- Streamlit visualization dashboard
- User input for Expected Returns, Volatility, Inflation and Simulation Years
- Projections based on Investment Strategy (Aggressive, Conservative, Balanced) with different asset allocations

📁 Project Files: `cern_pension_fund/fund_dashboard.py`

Run the script and open the link in your browser:
```bash
streamlit run cern_pension_fund/fund_dashboard.py
```

#### Notes:
- The dashboard, data and simulations are split into different files that all begin with `fund_`. The data are based on the CERN Pension Fund Annual Report and Financial Statements of 2023. 

---

### 3. 🧪 RSI and MACD with Multi Timeframe Strategy – Backtesting
Test using RSI and MACD indicators on different stocks and implement a Multi Timeframe Confirmation strategy. Backtesting the strategy and the influence of different parameters using the backtesting python lib.

#### Key Features:
- Historical data from Alpaca API and Yahoo Finance on multitimeframe (1h, 4h, day, week)
- Signal generation using moving average crossovers
- Backtesting engine to simulate portfolio performance
- Visualization of trades, returns, and performance vs. Buy & Hold
- Metrics: cumulative return, Sharpe ratio, max drawdown

📁 Project Files: `stocks/strat_multitimeframe_backtesting.py`

#### Notes:
- First trying to use a custom framework for backtesting, due to suspiciously good returns, I transitioned to using backtesting python lib for backtesting. 
- Custom framework and strategy can be found in `strat_multitimeframe_rsi_macd.py`

---

### 4. 📊 Portfolio Optimization – Black-Litterman
Introduction to Black-Litterman model to generate optimal weights based on market views and confidence level.
Used actual CERN Pension Fund Asset Allocation and returns from the Annual Report and Financial Statements of 2023.

#### Key Features:
- Portfolio simulation across selected assets
- Theorical portfolio in different assets classes with returns calculated from yfinance data.
- views extracted from the 2023 returns, the view begin the returns will continue similarly. The confidence omega is randomized in a defined range.

📁 Project Files: `cern_pension_fund/portfolio_black_litterman.py`

---

## 📦 Dependencies

All dependencies are listed in the `environment.yml` and `requirements.txt`files.

To install:

```bash
conda env create -f environment.yml
```

## Backtesting Platform

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

```bash
cd backtesting-platform
# Step 1: Initialize the DB
docker-compose run --rm airflow-webserver airflow db init

# Step 2: Start webserver and scheduler
docker-compose up airflow-webserver airflow-scheduler
```
