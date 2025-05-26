import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from ta.momentum import RSIIndicator
from backtesting import Backtest, Strategy

# 1. Load Data
ticker = "SPY"
data = yf.download(ticker, start="2024-01-01", end="2024-05-25", interval="1h")
data.dropna(inplace=True)
data['Return'] = np.log(data['Close'] / data['Close'].shift(1))

# 2. Hidden Markov Model (Regime Detection)
log_returns = data['Return'].dropna().values.reshape(-1, 1)
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
hmm_model.fit(log_returns)
data = data.iloc[1:]  # Align with returns
hidden_states = hmm_model.predict(log_returns)
data['Regime'] = hidden_states

# 3. Technical Indicators (RSI & Z-Score)
data['RSI'] = RSIIndicator(close=data['Close']).rsi()
data['ZScore'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).std()
data.dropna(inplace=True)

# 4. Backtesting Strategy
class MultiFactorStrategy(Strategy):
    def init(self):
        self.rsi = self.I(RSIIndicator, self.data.Close, 14).rsi()
        self.z = self.data.Close.rolling(20).mean()  # Not directly used
        self.zscore = (self.data.Close - self.data.Close.rolling(20).mean()) / self.data.Close.rolling(20).std()

    def next(self):
        if self.rsi[-1] < 30 and self.zscore[-1] < -1:
            self.buy()
        elif self.rsi[-1] > 70 and self.zscore[-1] > 1:
            self.sell()

bt = Backtest(data, MultiFactorStrategy, cash=10_000, commission=0.002)
results = bt.run()
print(results)
bt.plot()

# 5. Visualization with Regimes
fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
regimes = data['Regime'].unique()
colors = ['#ffcccc', '#ccffcc', '#ccccff']

for i, regime in enumerate(regimes):
    regime_mask = data['Regime'] == regime
    ax[0].plot(data.index[regime_mask], data['Close'][regime_mask], '.', label=f'Regime {regime}', color=colors[i])

ax[0].set_title(f"{ticker} Price with Regimes")
ax[0].legend()

ax[1].plot(data['RSI'], label='RSI')
ax[1].axhline(30, linestyle='--', color='red')
ax[1].axhline(70, linestyle='--', color='green')
ax[1].set_title("RSI")
ax[1].legend()

ax[2].plot(data['ZScore'], label='Z-Score')
ax[2].axhline(0, linestyle='--', color='black')
ax[2].axhline(-1, linestyle='--', color='red')
ax[2].axhline(1, linestyle='--', color='green')
ax[2].set_title("Z-Score")
ax[2].legend()

plt.tight_layout()
plt.show()
