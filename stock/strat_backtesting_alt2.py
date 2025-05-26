import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
from hurst import compute_Hc
from ta.momentum import RSIIndicator
from scipy.fft import fft
from backtesting import Strategy
from backtesting import Backtest

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os

load_dotenv()

client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))

request_params = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Hour,
    start="2024-01-01",
    end="2024-05-25"
)

data = client.get_stock_bars(request_params).df

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# If using only one symbol, drop the symbol level and set index to timestamp
if isinstance(data.index, pd.MultiIndex):
    data = data.xs("SPY", level="symbol")  # or your symbol
    data.index = pd.to_datetime(data.index)

def kaufman_er(close, window=10):
    direction = abs(close.diff(window))
    volatility = close.diff().abs().rolling(window=window).sum()
    return direction / volatility

def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns)

def sortino_ratio(returns):
    downside = returns[returns < 0]
    return np.mean(returns) / np.std(downside)

# Calculate indicators
log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna().values.reshape(-1, 1)
log_returns = log_returns[np.isfinite(log_returns)]
log_returns = log_returns.reshape(-1, 1)  # Ensure 2D shape

if len(log_returns) > 100:  # Require at least 100 points for stability
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
    model.fit(log_returns)
    hidden_states = model.predict(log_returns)
    data['regime'] = np.nan
    data.loc[data.index[1:], 'regime'] = hidden_states  # align with returns
else:
    data['regime'] = np.nan

hurst, _, _ = compute_Hc(data['Close'])
data['hurst'] = data['Close'].rolling(100).apply(lambda x: compute_Hc(x)[0])

data['rsi'] = RSIIndicator(data['Close'], window=14).rsi()
data['zscore'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).std()
data['kaufman_er'] = kaufman_er(data['Close'])
data['fft_energy'] = data['Close'].rolling(100).apply(lambda x: np.sum(np.abs(fft(x))))

returns = data['Close'].pct_change()
window = 20
rolling_mean = returns.rolling(window).mean()
rolling_std = returns.rolling(window).std()
downside_std = returns[returns < 0].rolling(window).std()

data['sharpe'] = rolling_mean / rolling_std
data['sortino'] = rolling_mean / downside_std

# Placeholder for session breakout/candlestick pattern detection
def detect_session_breakout(df):
    # TODO: Implement real logic
    return pd.Series([0]*len(df), index=df.index)

def detect_candlestick_pattern(df):
    # TODO: Implement real logic
    return pd.Series([0]*len(df), index=df.index)

data['session_breakout'] = detect_session_breakout(data)
data['candlestick_pattern'] = detect_candlestick_pattern(data)

class MultiIndicatorStrategy(Strategy):
    def init(self):
        self.rsi = self.I(lambda x: x, self.data.rsi, name='RSI')
        self.z = self.I(lambda x: x, self.data.zscore, name='Z-Score')
        self.er = self.I(lambda x: x, self.data.kaufman_er, name='Kaufman ER')
        self.hurst = self.I(lambda x: x, self.data.hurst, name='Hurst')
        self.fft_energy = self.I(lambda x: x, self.data.fft_energy, name='FFT Energy')
        self.regime = self.I(lambda x: x, self.data.regime, name='Regime')
        self.sharpe = self.I(lambda x: x, self.data.sharpe, name='Sharpe')
        self.sortino = self.I(lambda x: x, self.data.sortino, name='Sortino')
        self.session_breakout = self.I(lambda x: x, self.data.session_breakout, name='Session Breakout')
        self.candle = self.I(lambda x: x, self.data.candlestick_pattern, name='Candlestick Pattern')

    def next(self):
        bullish = 0
        bearish = 0
        # Simple default logic for each indicator
        if self.rsi[-1] < 35: bullish += 1
        if self.rsi[-1] > 65: bearish += 1
        if self.z[-1] < -1: bullish += 1
        if self.z[-1] > 1: bearish += 1
        if self.er[-1] > 0.5: bullish += 1
        if self.er[-1] < 0.2: bearish += 1
        if self.hurst[-1] < 0.5: bullish += 1  # mean reverting
        if self.hurst[-1] > 0.7: bearish += 1  # trending
        if self.fft_energy[-1] > np.nanpercentile(self.fft_energy, 80): bullish += 1
        if self.fft_energy[-1] < np.nanpercentile(self.fft_energy, 20): bearish += 1
        if self.regime[-1] == 1: bullish += 1
        if self.regime[-1] == 2: bearish += 1
        if self.sharpe[-1] > 0.5: bullish += 1
        if self.sharpe[-1] < 0: bearish += 1
        if self.sortino[-1] > 0.5: bullish += 1
        if self.sortino[-1] < 0: bearish += 1
        if self.session_breakout[-1] == 1: bullish += 1
        if self.session_breakout[-1] == -1: bearish += 1
        if self.candle[-1] == 1: bullish += 1
        if self.candle[-1] == -1: bearish += 1
        # Default: buy if more bullish, sell if more bearish
        if bullish > bearish and not self.position:
            self.buy()
        elif bearish > bullish and self.position:
            self.sell()

bt = Backtest(data, MultiIndicatorStrategy, cash=10_000, commission=0.002)
results = bt.run()
bt.plot()
