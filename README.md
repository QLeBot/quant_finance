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

#### Notes:
- Created an alternate version with calculated parameters such as risk free rate based on treasury bonds yield, stock price, volatility. It can be found in `black_scholes_auto_interactive.py`

---

### 2. 🧪 RSI and MACD with Multi Timeframe Strategy – Backtesting
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

### 3. 📊 Portfolio Optimization – Black-Litterman
Introduction to Black-Litterman model to generate optimal weights based on market views and confidence level.
Used actual CERN Pension Fund Asset Allocation and returns from the Annual Report and Financial Statements of 2023.

#### Key Features:
- Portfolio simulation across selected assets
- Theorical portfolio in different assets classes with returns calculated from yfinance data.
- views extracted from the 2023 returns, the view begin the returns will continue similarly. The confidence omega is randomized in a defined range.

📁 Project Files: `cern_pension_fund/portfolio_black_litterman.py`

---

## 📦 Dependencies

All dependencies are listed in the `requirements.txt` file.

To install:

```bash
pip install -r requirements.txt