import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from hurst import compute_Hc
from scipy.fft import fft
from scipy.stats import entropy
from multiprocessing import Pool, cpu_count
from functools import partial
from dotenv import load_dotenv

from backtesting import Backtest, Strategy
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, CCIIndicator
from ta.volume import OnBalanceVolumeIndicator

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca client
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Parameters
symbol = "SPY"
initial_cash = 1000000
start_date = datetime.datetime(2013, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

def get_data(symbol, timeframe):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="split"
    )
    return stock_client.get_stock_bars(request_params).df

def prepare_data_for_backtesting(df):
    """Prepare data for backtesting with proper column names and index."""
    # Reset index to get timestamp as a column
    df = df.reset_index()
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Ensure we have the required columns with correct names
    return pd.DataFrame({
        'Open': df['open'],
        'High': df['high'],
        'Low': df['low'],
        'Close': df['close'],
        'Volume': df['volume']
    })

def resample_daily_to_hourly(daily_df, hourly_index):
    """Resample daily data to match hourly index by forward filling."""
    hourly_df = pd.DataFrame(index=hourly_index)
    for col in daily_df.columns:
        hourly_df[col] = daily_df[col].reindex(hourly_index, method='ffill')
    return hourly_df

def kaufman_er(close, window=10):
    """Calculate Kaufman's Efficiency Ratio."""
    direction = abs(close.diff(window))
    volatility = close.diff().abs().rolling(window=window).sum()
    return direction / volatility

def calculate_price_entropy(prices, window=20):
    """Calculate price entropy using Shannon entropy."""
    prices = np.array(prices)
    entropy_values = np.zeros(len(prices))
    
    for i in range(window, len(prices)):
        price_changes = np.diff(prices[i-window:i])
        hist, _ = np.histogram(price_changes, bins=10, density=True)
        ent = entropy(hist)
        entropy_values[i] = ent
    
    entropy_values[:window] = entropy_values[window]
    return entropy_values

class EnhancedStrategy(Strategy):
    def init(self):
        # Clear existing signals file if it exists
        if os.path.exists('stock/csv/strat_backtesting_signals.csv'):
            os.remove('stock/csv/strat_backtesting_signals.csv')
            
        # Get daily data for multi-timeframe analysis
        daily_data = get_data(symbol, TimeFrame.Day)
        daily_df = prepare_data_for_backtesting(daily_data)
        daily_df = resample_daily_to_hourly(daily_df, self.data.index)
        
        # Calculate log returns for HMM
        close_series = pd.Series(self.data.Close, index=self.data.index)
        log_returns = np.log(close_series / close_series.shift(1)).dropna()
        log_returns = log_returns[np.isfinite(log_returns)]
        log_returns = log_returns.values.reshape(-1, 1)  # Reshape to 2D array
        
        # Fit HMM model if enough data points
        if len(log_returns) > 100:
            self.hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
            self.hmm_model.fit(log_returns)
            self.hidden_states = self.hmm_model.predict(log_returns)
            # Create a series with NaN for the first value to match data length
            regime_series = pd.Series(index=self.data.index)
            regime_series.iloc[1:] = self.hidden_states
            self.regime = self.I(lambda x: x, regime_series, name='Regime')
        
        # Calculate Hurst exponent
        self.hurst = self.I(lambda x: x, pd.Series(self.data.Close).rolling(100).apply(lambda x: compute_Hc(x)[0]), name='Hurst', color='blue')
        
        # Kaufman's Efficiency Ratio
        self.kaufman_er = self.I(lambda x: kaufman_er(pd.Series(x)), self.data.Close, name='Kaufman ER', color='red')
        
        # Combine Hurst and Kaufman ER for plotting
        def combine_indicators(hurst, kaufman):
            # Normalize both indicators to 0-1 range for better visualization
            hurst_norm = (hurst - np.nanmin(hurst)) / (np.nanmax(hurst) - np.nanmin(hurst))
            kaufman_norm = (kaufman - np.nanmin(kaufman)) / (np.nanmax(kaufman) - np.nanmin(kaufman))
            return hurst_norm, kaufman_norm
        
        self.combined_indicators = self.I(
            lambda x, y: combine_indicators(x, y),
            self.hurst,
            self.kaufman_er,
            name='Combined_Indicators',
            overlay=False
        )
        
        # FFT Energy
        self.fft_energy = self.I(lambda x: pd.Series(x).rolling(100).apply(lambda x: np.sum(np.abs(fft(x)))), self.data.Close, name='FFT Energy')
        
        # Price Entropy
        self.price_entropy = self.I(lambda x: calculate_price_entropy(x, window=20), self.data.Close, name='Price_Entropy')
        self.entropy_sma = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.price_entropy, name='Entropy_SMA')
        self.entropy_std = self.I(lambda x: pd.Series(x).rolling(window=20).std(), self.price_entropy, name='Entropy_Std')
        
        # Traditional Technical Indicators
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x), window=14).rsi(), self.data.Close, name='RSI')
        self.macd = self.I(lambda x: MACD(pd.Series(x)).macd(), self.data.Close, name='MACD')
        self.macd_signal = self.I(lambda x: MACD(pd.Series(x)).macd_signal(), self.data.Close, name='MACD_Signal')
        self.stoch = self.I(lambda x: StochasticOscillator(
            high=pd.Series(self.data.High),
            low=pd.Series(self.data.Low),
            close=pd.Series(x),
            window=14,
            smooth_window=3
        ).stoch(), self.data.Close, name='Stoch')
        
        # Moving Averages
        self.ma20 = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.data.Close, name='MA20')
        self.ma50 = self.I(lambda x: pd.Series(x).rolling(window=50).mean(), self.data.Close, name='MA50')
        self.ma200 = self.I(lambda x: pd.Series(x).rolling(window=200).mean(), self.data.Close, name='MA200')
        
        # Volatility
        self.atr = self.I(lambda x: pd.Series(x).rolling(window=14).apply(lambda x: max(x) - min(x)), self.data.Close, name='ATR')
        self.atr_sma = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.atr, name='ATR_SMA')
        
        # Risk Management Parameters
        self.risk_per_trade = 0.05  # 5% risk per trade
        self.max_drawdown = 0.15    # Maximum 15% drawdown
        self.reward_ratio = 2.5     # Target 2.5:1 reward-to-risk ratio
        self.trailing_stop_pct = 0.03 # 3% trailing stop
        self.initial_stop_pct = 0.03 # 3% initial stop loss
        
        # Trade tracking
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.peak_equity = initial_cash
        self.last_trade_date = None
        self.min_days_between_trades = 2
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
        # Signal line for visualization
        self.signal_line = self.I(lambda x: np.zeros_like(x), self.data.Close, name='Signal_Line', color='red')

    def detect_market_regime(self):
        """Enhanced market regime detection using multiple indicators."""
        # HMM Regime
        hmm_regime = self.regime[-1] if hasattr(self, 'regime') else None
        
        # Trend Detection
        price_above_ma20 = self.data.Close[-1] > self.ma20[-1]
        ma20_above_ma50 = self.ma20[-1] > self.ma50[-1]
        price_above_ma200 = self.data.Close[-1] > self.ma200[-1]
        
        # Volatility
        atr_ratio = self.atr[-1] / self.atr_sma[-1] if self.atr_sma[-1] != 0 else 1.0
        
        # Market Efficiency
        er_value = self.kaufman_er[-1]
        
        # Price Entropy
        entropy_zscore = (self.price_entropy[-1] - self.entropy_sma[-1]) / self.entropy_std[-1] if self.entropy_std[-1] != 0 else 0
        
        # Determine regime
        if (price_above_ma20 and ma20_above_ma50 and price_above_ma200 and
            atr_ratio < 1.5 and er_value > 0.5 and entropy_zscore < 1):
            regime = "strong_uptrend"
        elif (not price_above_ma20 and not ma20_above_ma50 and not price_above_ma200 and
              atr_ratio < 1.5 and er_value > 0.5 and entropy_zscore < 1):
            regime = "strong_downtrend"
        elif (price_above_ma20 and ma20_above_ma50 and
              atr_ratio < 2.0 and er_value > 0.3):
            regime = "uptrend"
        elif (not price_above_ma20 and not ma20_above_ma50 and
              atr_ratio < 2.0 and er_value > 0.3):
            regime = "downtrend"
        else:
            regime = "range"
            
        return regime, atr_ratio, hmm_regime

    def check_entry_conditions(self):
        """Enhanced entry conditions using multiple indicators."""
        if self.last_trade_date is not None:
            days_since_last_trade = (self.data.index[-1] - self.last_trade_date).days
            if days_since_last_trade < self.min_days_between_trades:
                return None
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            return None
        
        regime, atr_ratio, hmm_regime = self.detect_market_regime()
        
        # Technical Indicators
        rsi_value = self.rsi[-1]
        macd_value = self.macd[-1]
        macd_signal_value = self.macd_signal[-1]
        stoch_value = self.stoch[-1]
        
        # Advanced Indicators
        er_value = self.kaufman_er[-1]
        entropy_zscore = (self.price_entropy[-1] - self.entropy_sma[-1]) / self.entropy_std[-1] if self.entropy_std[-1] != 0 else 0
        fft_energy = self.fft_energy[-1]
        fft_energy_percentile = np.nanpercentile(self.fft_energy, 80)
        
        # MACD Crossover
        macd_cross_above = (self.macd[-2] <= self.macd_signal[-2] and 
                          self.macd[-1] > self.macd_signal[-1])
        macd_cross_below = (self.macd[-2] >= self.macd_signal[-2] and 
                          self.macd[-1] < self.macd_signal[-1])
        
        # Long Entry Conditions
        if (rsi_value < 70 and  # RSI not overbought
            macd_cross_above and  # MACD crosses above signal
            atr_ratio < 2.0 and  # Volatility check
            er_value > 0.3 and  # Market efficiency
            entropy_zscore < 1 and  # Low entropy (trending)
            fft_energy > fft_energy_percentile and  # High energy
            (hmm_regime == 0 or hmm_regime is None)):  # HMM regime check
            return "long"
            
        # Short Entry Conditions
        elif (rsi_value > 30 and  # RSI not oversold
              macd_cross_below and  # MACD crosses below signal
              atr_ratio < 2.0 and  # Volatility check
              er_value > 0.3 and  # Market efficiency
              entropy_zscore < 1 and  # Low entropy (trending)
              fft_energy > fft_energy_percentile and  # High energy
              (hmm_regime == 2 or hmm_regime is None)):  # HMM regime check
            return "short"
            
        return None

    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk and volatility."""
        adjusted_risk = self.risk_per_trade * (1 - (self.consecutive_losses * 0.2))
        risk_amount = self.equity * adjusted_risk
        risk_per_share = abs(price - stop_loss)
        
        atr_ratio = self.atr[-1] / self.atr_sma[-1] if self.atr_sma[-1] != 0 else 1.0
        volatility_adjustment = 1.0 / max(1.0, atr_ratio)
        
        raw_position_size = (risk_amount / risk_per_share) * volatility_adjustment
        max_position_size = self.equity / price
        
        if raw_position_size < 1:
            position_size = min(raw_position_size, 0.99)
        else:
            position_size = min(int(raw_position_size), int(max_position_size))
            position_size = max(1, position_size)
        
        return position_size

    def update_trailing_stop(self, price):
        """Update trailing stop based on price movement and volatility."""
        if self.trailing_stop is None:
            self.trailing_stop = price * (1 - self.trailing_stop_pct)
        else:
            atr_multiplier = self.atr[-1] / self.data.Close[-1]
            new_trailing_stop = price * (1 - max(self.trailing_stop_pct, atr_multiplier * 2))
            self.trailing_stop = max(self.trailing_stop, new_trailing_stop)

    def track_signals(self):
        """Track and save indicator signals to CSV file."""
        regime, atr_ratio, hmm_regime = self.detect_market_regime()
        
        signals = {
            'timestamp': self.data.index[-1],
            'price': self.data.Close[-1],
            'regime': regime,
            'hmm_regime': hmm_regime,
            'rsi': self.rsi[-1],
            'macd': self.macd[-1],
            'macd_signal': self.macd_signal[-1],
            'stoch': self.stoch[-1],
            'kaufman_er': self.kaufman_er[-1],
            'entropy_zscore': (self.price_entropy[-1] - self.entropy_sma[-1]) / self.entropy_std[-1] if self.entropy_std[-1] != 0 else 0,
            'fft_energy': self.fft_energy[-1],
            'atr_ratio': atr_ratio,
            'hurst': self.hurst[-1] if not np.isnan(self.hurst[-1]) else None
        }
        
        # Convert to DataFrame and save
        df = pd.DataFrame([signals])
        file_path = 'stock/csv/strat_backtesting_signals.csv'
        
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)
        
        return signals

    def next(self):
        current_price = self.data.Close[-1]
        
        # Update peak equity for drawdown protection
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # Track signals
        self.track_signals()
        
        # Update signal line
        if not self.position:
            entry_signal = self.check_entry_conditions()
            if entry_signal == "long":
                self.signal_line[-1] = 1
            elif entry_signal == "short":
                self.signal_line[-1] = -1
            else:
                self.signal_line[-1] = 0
        else:
            self.signal_line[-1] = 0
        
        # Check for entry conditions
        if not self.position:
            entry_signal = self.check_entry_conditions()
            
            if entry_signal == "long":
                stop_loss = current_price * (1 - self.initial_stop_pct)
                position_size = self.calculate_position_size(current_price, stop_loss)
                take_profit = current_price + (current_price - stop_loss) * self.reward_ratio
                
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.trailing_stop = None
                
                self.buy(size=position_size)
                
            elif entry_signal == "short":
                stop_loss = current_price * (1 + self.initial_stop_pct)
                position_size = self.calculate_position_size(current_price, stop_loss)
                take_profit = current_price - (stop_loss - current_price) * self.reward_ratio
                
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.trailing_stop = None
                
                self.sell(size=position_size)
        
        # Manage open positions
        if self.position:
            self.update_trailing_stop(current_price)
            
            if self.position.is_long:
                if current_price <= self.stop_loss or current_price <= self.trailing_stop:
                    self.position.close()
                    self.consecutive_losses += 1
                    self.reset_trade_info()
                elif current_price >= self.take_profit:
                    self.position.close()
                    self.consecutive_losses = 0
                    self.reset_trade_info()
            else:  # Short position
                if current_price >= self.stop_loss or current_price >= self.trailing_stop:
                    self.position.close()
                    self.consecutive_losses += 1
                    self.reset_trade_info()
                elif current_price <= self.take_profit:
                    self.position.close()
                    self.consecutive_losses = 0
                    self.reset_trade_info()

    def reset_trade_info(self):
        """Reset trade tracking information."""
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.last_trade_date = self.data.index[-1]

def main():
    # Get data
    df = get_data(symbol, TimeFrame.Hour)
    if df.empty:
        print(f"No data available for {symbol}")
        return
        
    # Prepare data
    bt_df = prepare_data_for_backtesting(df)
    
    # Run backtest
    bt = Backtest(bt_df, EnhancedStrategy, cash=initial_cash, commission=0.0035, finalize_trades=True)
    results = bt.run()
    
    # Print results
    print(f"\nBacktest Results for {symbol}:")
    print(results)
    
    # Plot results
    bt.plot(filename=f'backtest_{symbol}.html')
    
    # Save results to CSV
    #results_df = pd.DataFrame([results])
    #results_df.to_csv('backtest_results.csv', index=False)
    #print("\nResults saved to backtest_results.csv")

if __name__ == "__main__":
    main() 