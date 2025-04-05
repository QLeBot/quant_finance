# 📈 Quant Finance Lab – Strategy Backtesting & Portfolio Optimization

This repository contains two beginner-to-intermediate level projects in **quantitative finance using Python**. The goal is to apply core data science and programming skills to real-world financial problems: algorithmic trading and portfolio optimization.

## 🧠 Projects Overview

### 1. 🧪 Moving Average Crossover Strategy – Backtesting
A simple momentum-based trading strategy using moving averages (SMA 50 and SMA 200) to generate buy/sell signals.

#### Key Features:
- Historical data from Yahoo Finance via `yfinance`
- Signal generation using moving average crossovers
- Backtesting engine to simulate portfolio performance
- Visualization of trades, returns, and performance vs. Buy & Hold
- Metrics: cumulative return, Sharpe ratio, max drawdown

📁 Project Folder: `moving_average_strategy/`

---

### 2. 📊 Portfolio Optimization – Modern Portfolio Theory
Simulation of thousands of portfolio combinations to visualize the efficient frontier and find optimal allocations based on risk-return profiles.

#### Key Features:
- Portfolio simulation across selected assets
- Computation of annualized return, volatility, and Sharpe ratio
- Efficient frontier visualization with `matplotlib`
- Optional: convex optimization to find max Sharpe ratio portfolio

📁 Project Folder: `portfolio_optimizer/`

---

## 📦 Dependencies

All dependencies are listed in the `requirements.txt` file. Key libraries include:

- `yfinance`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scipy` or `cvxpy` (optional for optimization)
- `streamlit` (optional for UI)

To install:

```bash
pip install -r requirements.txt