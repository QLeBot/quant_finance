import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.stats import entropy

from backtesting import Backtest, Strategy
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, CCIIndicator
from ta.volume import OnBalanceVolumeIndicator
from dotenv import load_dotenv

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

def get_risk_free_rate_tnx():
    """Fetches the latest U.S. 10-Year Treasury yield using yfinance."""
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100
        return latest_yield
    else:
        raise ValueError("No data found for ^TNX.")

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
    # Create a DataFrame with the hourly index
    hourly_df = pd.DataFrame(index=hourly_index)
    
    # Forward fill daily data to match hourly frequency
    for col in daily_df.columns:
        hourly_df[col] = daily_df[col].reindex(hourly_index, method='ffill')
    
    return hourly_df

class SimpleStrategy(Strategy):
    def init(self):
        # Clear existing signals file if it exists
        import os
        if os.path.exists('stock/csv/strat_backtesting_signals.csv'):
            os.remove('stock/csv/strat_backtesting_signals.csv')
            
        # Get daily data for multi-timeframe analysis
        daily_data = get_data(symbol, TimeFrame.Day)
        daily_df = prepare_data_for_backtesting(daily_data)
        
        # Resample daily data to match hourly frequency
        daily_df = resample_daily_to_hourly(daily_df, self.data.index)
        
        # Daily timeframe indicators (not plotted)
        self.daily_ma50 = self.I(lambda x: pd.Series(x).rolling(window=50).mean(), daily_df['Close'], name='Daily_MA50', plot=False)
        self.daily_ma200 = self.I(lambda x: pd.Series(x).rolling(window=200).mean(), daily_df['Close'], name='Daily_MA200', plot=False)
        self.daily_rsi = self.I(lambda x: RSIIndicator(pd.Series(x), window=14).rsi(), daily_df['Close'], name='Daily_RSI', plot=False)
        self.daily_atr = self.I(lambda x: pd.Series(x).rolling(window=14).apply(lambda x: max(x) - min(x)), daily_df['Close'], name='Daily_ATR', plot=False)
        self.daily_atr_sma = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.daily_atr, name='Daily_ATR_SMA', plot=False)
        self.daily_stoch = self.I(lambda x: StochasticOscillator(
            high=pd.Series(daily_df['High']),
            low=pd.Series(daily_df['Low']),
            close=pd.Series(x),
            window=14,
            smooth_window=3
        ).stoch(), daily_df['Close'], name='Daily_Stoch', plot=False)
        self.daily_williams = self.I(lambda x: WilliamsRIndicator(
            high=pd.Series(daily_df['High']),
            low=pd.Series(daily_df['Low']),
            close=pd.Series(x),
            lbp=14
        ).williams_r(), daily_df['Close'], name='Daily_Williams', plot=False)
        self.daily_cci = self.I(lambda x: CCIIndicator(
            high=pd.Series(daily_df['High']),
            low=pd.Series(daily_df['Low']),
            close=pd.Series(x),
            window=20
        ).cci(), daily_df['Close'], name='Daily_CCI', plot=False)
        self.daily_roc = self.I(lambda x: ROCIndicator(close=pd.Series(x), window=12).roc(), daily_df['Close'], name='Daily_ROC', plot=False)
        self.daily_obv = self.I(lambda x: OnBalanceVolumeIndicator(
            close=pd.Series(x),
            volume=pd.Series(daily_df['Volume'])
        ).on_balance_volume(), daily_df['Close'], name='Daily_OBV', plot=False)
        
        # Hourly timeframe indicators
        close_series = pd.Series(self.data.Close, index=self.data.index)
        
        # Price Entropy Indicators
        self.price_entropy = self.I(lambda x: self.calculate_price_entropy(x, window=20), self.data.Close, name='Price_Entropy')
        self.entropy_sma = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.price_entropy, name='Entropy_SMA')
        self.entropy_std = self.I(lambda x: pd.Series(x).rolling(window=20).std(), self.price_entropy, name='Entropy_Std')
        
        # Trend Indicators
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x, index=self.data.index), window=14).rsi(), self.data.Close, name='RSI')
        self.macd = self.I(lambda x: MACD(
            pd.Series(x, index=self.data.index),
            window_slow=26,
            window_fast=12,
            window_sign=9
        ).macd(), self.data.Close, name='MACD')
        self.macd_signal = self.I(lambda x: MACD(
            pd.Series(x, index=self.data.index),
            window_slow=26,
            window_fast=12,
            window_sign=9
        ).macd_signal(), self.data.Close, name='MACD_Signal')
        self.macd_hist = self.I(lambda x: MACD(
            pd.Series(x, index=self.data.index),
            window_slow=26,
            window_fast=12,
            window_sign=9
        ).macd_diff(), self.data.Close, name='MACD_Hist')
        
        # Leading Indicators
        self.stoch = self.I(lambda x: StochasticOscillator(
            high=pd.Series(self.data.High, index=self.data.index),
            low=pd.Series(self.data.Low, index=self.data.index),
            close=pd.Series(x, index=self.data.index),
            window=14,
            smooth_window=3
        ).stoch(), self.data.Close, name='Stoch')
        self.williams = self.I(lambda x: WilliamsRIndicator(
            high=pd.Series(self.data.High, index=self.data.index),
            low=pd.Series(self.data.Low, index=self.data.index),
            close=pd.Series(x, index=self.data.index),
            lbp=14
        ).williams_r(), self.data.Close, name='Williams')
        self.cci = self.I(lambda x: CCIIndicator(
            high=pd.Series(self.data.High, index=self.data.index),
            low=pd.Series(self.data.Low, index=self.data.index),
            close=pd.Series(x, index=self.data.index),
            window=20
        ).cci(), self.data.Close, name='CCI')
        self.roc = self.I(lambda x: ROCIndicator(close=pd.Series(x, index=self.data.index), window=12).roc(), self.data.Close, name='ROC')
        self.obv = self.I(lambda x: OnBalanceVolumeIndicator(
            close=pd.Series(x, index=self.data.index),
            volume=pd.Series(self.data.Volume, index=self.data.index)
        ).on_balance_volume(), self.data.Close, name='OBV')
        
        # Volume Indicators
        self.volume_sma = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.data.Volume, name='Volume_SMA')
        self.volume_ratio = self.I(lambda x: pd.Series(x) / pd.Series(x).rolling(window=20).mean(), self.data.Volume, name='Volume_Ratio')
        
        # Volatility Indicators
        self.atr = self.I(lambda x: pd.Series(x).rolling(window=14).apply(lambda x: max(x) - min(x)), self.data.Close, name='ATR')
        self.atr_sma = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.atr, name='ATR_SMA')
        
        # Trend Indicators
        self.ma20 = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), self.data.Close, name='MA20')
        self.ma50 = self.I(lambda x: pd.Series(x).rolling(window=50).mean(), self.data.Close, name='MA50')
        self.ma200 = self.I(lambda x: pd.Series(x).rolling(window=200).mean(), self.data.Close, name='MA200')
        
        # Signal Line (shows when trades are taken)
        self.signal_line = self.I(lambda x: np.zeros_like(x), self.data.Close, name='Signal_Line', color='red')
        
        # Risk Management Parameters
        self.risk_per_trade = 0.05  # 5% risk per trade
        self.max_drawdown = 0.15    # Maximum 15% drawdown
        self.reward_ratio = 2.5     # Target 2.5:1 reward-to-risk ratio
        self.trailing_stop_pct = 0.03 # 3% trailing stop
        self.initial_stop_pct = 0.03 # 3% initial stop loss
        
        # Track trade information
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.peak_equity = initial_cash
        self.last_trade_date = None
        self.min_days_between_trades = 2
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
    def detect_market_regime(self):
        """Detect current market regime with enhanced trend detection."""
        # Daily timeframe trend
        daily_price_above_ma50 = self.data.Close[-1] > self.daily_ma50[-1]
        daily_price_above_ma200 = self.data.Close[-1] > self.daily_ma200[-1]
        daily_ma50_above_ma200 = self.daily_ma50[-1] > self.daily_ma200[-1]
        
        # Hourly timeframe trend
        price_above_ma20 = self.data.Close[-1] > self.ma20[-1]
        ma20_above_ma50 = self.ma20[-1] > self.ma50[-1]
        price_above_ma200 = self.data.Close[-1] > self.ma200[-1]
        ma50_above_ma200 = self.ma50[-1] > self.ma200[-1]
        
        # Volatility regime
        atr_ratio = self.atr[-1] / self.atr_sma[-1] if self.atr_sma[-1] != 0 else 1.0
        daily_atr_ratio = self.daily_atr[-1] / self.daily_atr_sma[-1] if self.daily_atr_sma[-1] != 0 else 1.0
        
        # Enhanced trend detection with volatility consideration
        if (daily_price_above_ma50 and daily_price_above_ma200 and daily_ma50_above_ma200 and
            price_above_ma20 and ma20_above_ma50 and price_above_ma200 and ma50_above_ma200 and
            atr_ratio < 1.5 and daily_atr_ratio < 1.5):
            regime = "strong_uptrend"
        elif (not daily_price_above_ma50 and not daily_price_above_ma200 and not daily_ma50_above_ma200 and
              not price_above_ma20 and not ma20_above_ma50 and not price_above_ma200 and not ma50_above_ma200 and
              atr_ratio < 1.5 and daily_atr_ratio < 1.5):
            regime = "strong_downtrend"
        elif (daily_price_above_ma50 and daily_ma50_above_ma200 and
              price_above_ma20 and ma20_above_ma50 and
              atr_ratio < 2.0 and daily_atr_ratio < 2.0):
            regime = "uptrend"
        elif (not daily_price_above_ma50 and not daily_ma50_above_ma200 and
              not price_above_ma20 and not ma20_above_ma50 and
              atr_ratio < 2.0 and daily_atr_ratio < 2.0):
            regime = "downtrend"
        else:
            regime = "range"
            
        return regime, atr_ratio
        
    def check_entry_conditions(self):
        """Check for entry conditions with enhanced filtering."""
        # Check minimum days between trades
        if self.last_trade_date is not None:
            days_since_last_trade = (self.data.index[-1] - self.last_trade_date).days
            if days_since_last_trade < self.min_days_between_trades:
                return None
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return None
        
        #regime, atr_ratio = self.detect_market_regime()

        roc_value = self.roc[-1]
        rsi_value = self.rsi[-1]
        macd_value = self.macd[-1]
        macd_signal_value = self.macd_signal[-1]
        
        # Leading Indicators (Primary signals)
        stoch_oversold = self.stoch[-1] < 20
        stoch_overbought = self.stoch[-1] > 80
        williams_oversold = self.williams[-1] < -80
        williams_overbought = self.williams[-1] > -20
        cci_oversold = self.cci[-1] < -100
        cci_overbought = self.cci[-1] > 100
        roc_bullish = self.roc[-1] > 0
        roc_bearish = self.roc[-1] < 0
        
        # Momentum Indicators (Secondary signals)
        rsi_oversold = self.rsi[-1] < 30
        rsi_overbought = self.rsi[-1] > 70
        
        # MA50 Crossover
        price_cross_above_ma50 = (self.data.Close[-2] <= self.ma50[-2] and 
                                self.data.Close[-1] > self.ma50[-1])
        price_cross_below_ma50 = (self.data.Close[-2] >= self.ma50[-2] and 
                                self.data.Close[-1] < self.ma50[-1])

        # Check MACD crossover
        macd_cross_above = (self.macd[-2] <= self.macd_signal[-2] and 
                          self.macd[-1] > self.macd_signal[-1])
        macd_cross_below = (self.macd[-2] >= self.macd_signal[-2] and 
                          self.macd[-1] < self.macd_signal[-1])
        
        # Volatility check
        atr_ratio = self.atr[-1] / self.atr_sma[-1] if self.atr_sma[-1] != 0 else 1.0
        volatility_ok = atr_ratio < 2.0
        
        # Entry conditions with enhanced filtering
        # Long entry conditions
        if (roc_value > 0 and  # ROC bullish
            rsi_value < 70 and  # RSI not overbought
            macd_cross_above and  # MACD crosses above signal
            volatility_ok):  # Volatility check
            return "long"
            
        # Short entry conditions
        elif (roc_value < 0 and  # ROC bearish
              rsi_value > 30 and  # RSI not oversold
              macd_cross_below and  # MACD crosses below signal
              volatility_ok):  # Volatility check
            return "short"
        return None
        
    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk per trade and volatility."""
        # Adjust risk based on consecutive losses
        adjusted_risk = self.risk_per_trade * (1 - (self.consecutive_losses * 0.2))  # Reduce risk by 20% per loss
        
        # Calculate risk amount in dollars
        risk_amount = self.equity * adjusted_risk
        
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss)
        
        # Adjust position size based on volatility
        atr_ratio = self.atr[-1] / self.atr_sma[-1] if self.atr_sma[-1] != 0 else 1.0
        daily_atr_ratio = self.daily_atr[-1] / self.daily_atr_sma[-1] if self.daily_atr_sma[-1] != 0 else 1.0
        volatility_adjustment = 1.0 / max(1.0, (atr_ratio + daily_atr_ratio) / 2)  # Average of both timeframes
        
        # Calculate raw position size
        raw_position_size = (risk_amount / risk_per_share) * volatility_adjustment
        
        # Calculate maximum position size based on equity
        max_position_size = self.equity / price
        
        # If raw position size is less than 1, use it as a fraction of equity
        if raw_position_size < 1:
            position_size = min(raw_position_size, 0.99)  # Cap at 99% of equity
        else:
            # Otherwise, use whole number of shares
            position_size = min(int(raw_position_size), int(max_position_size))
            position_size = max(1, position_size)  # Ensure at least 1 share
        
        return position_size
        
    def update_trailing_stop(self, price):
        """Update trailing stop based on price movement and volatility."""
        if self.trailing_stop is None:
            self.trailing_stop = price * (1 - self.trailing_stop_pct)
        else:
            # Adjust trailing stop based on ATR
            atr_multiplier = self.atr[-1] / self.data.Close[-1]
            new_trailing_stop = price * (1 - max(self.trailing_stop_pct, atr_multiplier * 2))
            self.trailing_stop = max(self.trailing_stop, new_trailing_stop)
            
    def calculate_price_entropy(self, prices, window=20):
        """Calculate price entropy using Shannon entropy."""
        # Convert to numpy array
        prices = np.array(prices)
        
        # Initialize result array
        entropy_values = np.zeros(len(prices))
        
        # Calculate entropy for each window
        for i in range(window, len(prices)):
            # Get price changes in the window
            price_changes = np.diff(prices[i-window:i])
            
            # Create histogram of price changes
            hist, _ = np.histogram(price_changes, bins=10, density=True)
            
            # Calculate entropy
            ent = entropy(hist)
            
            # Store entropy value
            entropy_values[i] = ent
        
        # Fill initial values with first valid entropy
        entropy_values[:window] = entropy_values[window]
        
        return entropy_values
        
    def track_signals(self):
        """Track and save indicator signals to CSV file."""
        signals = {
            'timestamp': self.data.index[-1],
            'price': self.data.Close[-1],
            'regime': self.detect_market_regime()[0],
            'roc_value': self.roc[-1],
            'rsi_value': self.rsi[-1],
            'macd_value': self.macd[-1],
            'macd_signal': self.macd_signal[-1],
            'macd_hist': self.macd_hist[-1],
            'macd_cross_above': (self.macd[-2] <= self.macd_signal[-2] and self.macd[-1] > self.macd_signal[-1]),
            'macd_cross_below': (self.macd[-2] >= self.macd_signal[-2] and self.macd[-1] < self.macd_signal[-1]),
            'atr_ratio': self.atr[-1] / self.atr_sma[-1] if self.atr_sma[-1] != 0 else 0,
            'volatility_ok': self.atr[-1] / self.atr_sma[-1] < 2.0 if self.atr_sma[-1] != 0 else True,
            'roc_bullish': self.roc[-1] > 0,
            'roc_bearish': self.roc[-1] < 0,
            'price_cross_above_ma50': (self.data.Close[-2] <= self.ma50[-2] and self.data.Close[-1] > self.ma50[-1]),
            'price_cross_below_ma50': (self.data.Close[-2] >= self.ma50[-2] and self.data.Close[-1] < self.ma50[-1]),
            'rsi_oversold': self.rsi[-1] < 30,
            'rsi_overbought': self.rsi[-1] > 70,
            'stoch_oversold': self.stoch[-1] < 20,
            'stoch_overbought': self.stoch[-1] > 80,
            'williams_oversold': self.williams[-1] < -80,
            'williams_overbought': self.williams[-1] > -20,
            'cci_oversold': self.cci[-1] < -100,
            'cci_overbought': self.cci[-1] > 100,
            'stoch_value': self.stoch[-1],
            'williams_value': self.williams[-1],
            'cci_value': self.cci[-1],
            'price_entropy': self.price_entropy[-1],
            'entropy_sma': self.entropy_sma[-1],
            'entropy_std': self.entropy_std[-1],
            'entropy_zscore': (self.price_entropy[-1] - self.entropy_sma[-1]) / self.entropy_std[-1] if self.entropy_std[-1] != 0 else 0
        }
        
        # Check long entry conditions
        long_conditions = (
            signals['roc_bullish'] and
            not signals['rsi_overbought'] and
            signals['macd_cross_above'] and
            signals['volatility_ok']
        )
        
        # Check short entry conditions
        short_conditions = (
            signals['roc_bearish'] and
            not signals['rsi_oversold'] and
            signals['macd_cross_below'] and
            signals['volatility_ok']
        )
        
        signals['long_entry_possible'] = long_conditions
        signals['short_entry_possible'] = short_conditions
        
        # Convert boolean values to integers for better CSV readability
        for key in signals:
            if isinstance(signals[key], bool):
                signals[key] = int(signals[key])
        
        # Save to CSV
        import os
        import pandas as pd
        
        # Create directory if it doesn't exist
        os.makedirs('stock/csv', exist_ok=True)
        
        # File path
        file_path = 'stock/csv/strat_backtesting_signals.csv'
        
        # Convert to DataFrame
        df = pd.DataFrame([signals])
        
        # Write to CSV
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
                self.signal_line[-1] = 1  # Long signal
            elif entry_signal == "short":
                self.signal_line[-1] = -1  # Short signal
            else:
                self.signal_line[-1] = 0  # No signal
        else:
            self.signal_line[-1] = 0  # No signal while in position
        
        # Check for entry conditions
        if not self.position:
            entry_signal = self.check_entry_conditions()
            
            if entry_signal == "long":
                # Calculate stop loss and position size
                stop_loss = current_price * (1 - self.initial_stop_pct)
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                # Set take profit based on reward ratio
                take_profit = current_price + (current_price - stop_loss) * self.reward_ratio
                
                # Store trade information
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.trailing_stop = None
                
                # Enter position
                self.buy(size=position_size)
                
            elif entry_signal == "short":
                # Calculate stop loss and position size
                stop_loss = current_price * (1 + self.initial_stop_pct)
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                # Set take profit based on reward ratio
                take_profit = current_price - (stop_loss - current_price) * self.reward_ratio
                
                # Store trade information
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.trailing_stop = None
                
                # Enter position
                self.sell(size=position_size)
                
        # Manage open positions
        if self.position:
            # Update trailing stop
            self.update_trailing_stop(current_price)
            
            # Check stop loss and take profit
            if self.position.is_long:
                if current_price <= self.stop_loss or current_price <= self.trailing_stop:
                    self.position.close()
                    self.consecutive_losses += 1
                    self.reset_trade_info()
                elif current_price >= self.take_profit:
                    self.position.close()
                    self.consecutive_losses = 0  # Reset on win
                    self.reset_trade_info()
            else:  # Short position
                if current_price >= self.stop_loss or current_price >= self.trailing_stop:
                    self.position.close()
                    self.consecutive_losses += 1
                    self.reset_trade_info()
                elif current_price <= self.take_profit:
                    self.position.close()
                    self.consecutive_losses = 0  # Reset on win
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
    bt = Backtest(bt_df, SimpleStrategy, cash=initial_cash, commission=0.0035, finalize_trades=True)
    results = bt.run()
    
    # Print results
    print(f"\nBacktest Results for {symbol}:")
    print(results)
    
    # Plot results
    bt.plot(filename=f'backtest_{symbol}.html')
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv('backtest_results.csv', index=False)
    print("\nResults saved to backtest_results.csv")

if __name__ == "__main__":
    main()