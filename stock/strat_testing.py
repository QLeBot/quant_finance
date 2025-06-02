import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import ta
from backtesting import Backtest, Strategy
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

client = StockHistoricalDataClient(API_KEY, API_SECRET)

class DeltaTrendTrading(Strategy):
    lookback_period = 20
    z_score_threshold = 1.0
    atr_multiplier = 2.0
    atr_period = 14
    
    def init(self):
        """Initialize the strategy's indicators."""
        # Convert data to pandas Series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Calculate z-scores
        self.rolling_mean = close.rolling(window=self.lookback_period).mean()
        self.rolling_std = close.rolling(window=self.lookback_period).std()
        self.z_scores = (close - self.rolling_mean) / self.rolling_std
        
        # Calculate ATR
        atr_indicator = ta.volatility.AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=self.atr_period
        )
        self.atr = atr_indicator.average_true_range()
        
        # Calculate stop levels
        self.long_stops = close - (self.atr * self.atr_multiplier)
        self.short_stops = close + (self.atr * self.atr_multiplier)
    
    def next(self):
        """Define the strategy's trading logic."""
        if len(self.data) < self.lookback_period:
            return
            
        # Get current values using proper indexing
        current_z = self.z_scores.iloc[-1]
        current_price = self.data.Close[-1]
        
        # Check for stop loss hits
        if self.position.is_long and self.data.Low[-1] <= self.long_stops.iloc[-2]:
            self.position.close()
        elif self.position.is_short and self.data.High[-1] >= self.short_stops.iloc[-2]:
            self.position.close()
            
        # Check for mean reversion exits
        elif self.position.is_long and current_z > 0:
            self.position.close()
        elif self.position.is_short and current_z < 0:
            self.position.close()
            
        # Check for new entries
        elif not self.position:
            if current_z < -self.z_score_threshold:
                self.buy()
            elif current_z > self.z_score_threshold:
                self.sell()

# Example usage
if __name__ == "__main__":
    # Load data
    request_params = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start="2025-01-01",
        end="2025-06-01"
    )
    bars = client.get_stock_bars(request_params).df

    # Prepare data for backtesting
    bt_df = bars[['open', 'high', 'low', 'close', 'volume']]
    bt_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    bt_df.index = pd.to_datetime(bars.index.get_level_values(1))  # Convert to datetime index

    # Run backtest
    bt = Backtest(bt_df, DeltaTrendTrading, cash=100000, commission=0.0035, finalize_trades=True)
    results = bt.run()
    print(results)

    # Plot results
    bt.plot(plot_equity=True, plot_return=True, plot_volume=True, plot_pl=True, plot_trades=True, show_legend=True, resample=True)

    # Calculate overlays
    rolling_mean = bt_df['Close'].rolling(window=20).mean()
    rolling_std = bt_df['Close'].rolling(window=20).std()
    upper_band = rolling_mean + rolling_std
    lower_band = rolling_mean - rolling_std

    # Candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=bt_df.index,
            open=bt_df['Open'],
            high=bt_df['High'],
            low=bt_df['Low'],
            close=bt_df['Close'],
            name='Candles'
        ),
        go.Scatter(
            x=bt_df.index,
            y=rolling_mean,
            line=dict(color='blue', width=1, dash='dash'),
            name='Rolling Mean'
        ),
        go.Scatter(
            x=bt_df.index,
            y=upper_band,
            line=dict(color='green', width=1),
            name='+1std'
        ),
        go.Scatter(
            x=bt_df.index,
            y=lower_band,
            line=dict(color='red', width=1),
            name='-1std'
        )
    ])

    fig.update_layout(
        title='Candlestick with Rolling Mean and Bands',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()
