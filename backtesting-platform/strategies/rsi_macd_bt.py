# strategies/rsi_macd_bt.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta

class RSIMACDStrategy(Strategy):
    def init(self):
        close = self.data.Close
        self.rsi = self.I(ta.momentum.RSIIndicator, close, window=14).rsi()
        self.macd, _, _ = self.I(ta.trend.MACD, close)

    def next(self):
        if crossover(self.rsi, 30) and self.macd[-1] > 0:
            self.buy()
        elif crossover(self.rsi, 70) and self.macd[-1] < 0:
            self.sell()
