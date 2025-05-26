from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from scipy.stats import norm

load_dotenv()

def calculate_delta(option_type, S, K, T, r, sigma):
    """
    Calculate option delta using Black-Scholes formula
    S: current stock price
    K: strike price
    T: time to expiration in years
    r: risk-free rate
    sigma: implied volatility
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:  # put
        return -norm.cdf(-d1)

def calculate_implied_volatility(option_type, S, K, T, r, price):
    """
    Calculate implied volatility using Newton-Raphson method
    """
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.3  # Initial guess
    
    for i in range(MAX_ITERATIONS):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price_estimate = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * np.sqrt(T) * norm.pdf(d1)
        else:
            price_estimate = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            vega = S * np.sqrt(T) * norm.pdf(d1)
        
        diff = price_estimate - price
        
        if abs(diff) < PRECISION:
            return sigma
            
        sigma = sigma - diff/vega
        
    return sigma

class ShortPutStrategy(Strategy):
    def init(self):
        # Technical indicators
        def ema(series, span):
            return pd.Series(series).ewm(span=span).mean()
        
        self.ema20 = self.I(ema, self.data.Close, 20)
        self.ema50 = self.I(ema, self.data.Close, 50)
        self.ema200 = self.I(ema, self.data.Close, 200)
        self.rsi = self.I(self.RSI, self.data.Close, 14)
        
        # Volatility indicators
        self.atr = self.I(self.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Risk-free rate (using 10-year Treasury yield as proxy)
        self.rf_rate = 0.04  # 4% annual rate
        
        # Risk management parameters
        self.max_positions = 3
        self.max_risk_per_trade = 0.02  # 2% of account per trade
        self.max_portfolio_risk = 0.06  # 6% of account total risk
        self.profit_target = 0.5  # 50% of premium collected
        self.stop_loss = 0.2  # 20% of premium collected
        
        # Track positions
        self.positions = []
        self.total_risk = 0

    def next(self):
        # Ensure we have enough data points before trading
        if len(self.data.Close) < 20:
            return
            
        # Update existing positions
        self.update_positions()
        
        # Check if we can open new positions
        if len(self.positions) >= self.max_positions:
            return
            
        # Entry conditions
        if self.check_entry_conditions():
            self.open_new_position()

    def check_entry_conditions(self):
        # Ensure we have enough data points
        if len(self.data.Close) < 20:
            return False
            
        # Trend conditions
        uptrend = (self.ema20[-1] > self.ema50[-1] > self.ema200[-1])
        
        # Volatility conditions
        low_volatility = self.atr[-1] < self.atr[-20]  # Volatility decreasing
        
        # RSI conditions
        oversold = 30 < self.rsi[-1] < 40
        
        # Price action
        price_increasing = self.data.Close[-1] > self.data.Close[-2]
        
        return uptrend and low_volatility and oversold and price_increasing

    def open_new_position(self):
        try:
            # Get current options chain
            url = f"https://paper-api.alpaca.markets/v2/options/contracts?underlying_symbols=AAPL"
            headers = {
                "accept": "application/json",
                "APCA-API-KEY-ID": os.getenv("PAPER_API_KEY"),
                "APCA-API-SECRET-KEY": os.getenv("PAPER_SECRET_KEY")
            }
            
            response = requests.get(url, headers=headers)
            response_data = response.json()
            chains = response_data['option_contracts']
            
            # Filter for put options
            puts = [c for c in chains if c['type'] == 'put']
            
            if puts:
                current_price = self.data.Close[-1]
                
                # Find optimal strike price
                optimal_strike = self.find_optimal_strike(puts, current_price)
                
                if optimal_strike:
                    # Calculate position size based on risk
                    account_value = self.equity
                    risk_amount = account_value * self.max_risk_per_trade
                    position_size = self.calculate_position_size(risk_amount, optimal_strike)
                    
                    if position_size > 0:
                        # Record position
                        self.positions.append({
                            'strike': optimal_strike['strike_price'],
                            'expiration': optimal_strike['expiration_date'],
                            'size': position_size,
                            'entry_price': current_price,
                            'premium': optimal_strike['last_price'],
                            'entry_date': self.data.index[-1]
                        })
                        
                        # Update total risk
                        self.total_risk += risk_amount
                        
                        # Simulate selling the put
                        self.buy(size=position_size)
        
        except Exception as e:
            print(f"Error opening position: {e}")

    def find_optimal_strike(self, puts, current_price):
        # Filter for 30-45 DTE options
        target_dte = 30
        valid_puts = []
        
        for put in puts:
            exp_date = datetime.strptime(put['expiration_date'], '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            
            if 25 <= dte <= 45:  # Allow some flexibility
                valid_puts.append(put)
        
        if not valid_puts:
            return None
            
        # Find strike with delta between -0.2 and -0.3
        optimal_put = None
        min_delta_diff = float('inf')
        
        for put in valid_puts:
            strike = float(put['strike_price'])
            T = (datetime.strptime(put['expiration_date'], '%Y-%m-%d') - datetime.now()).days / 365.0
            
            # Calculate implied volatility
            sigma = calculate_implied_volatility('put', current_price, strike, T, self.rf_rate, float(put['last_price']))
            
            # Calculate delta
            delta = calculate_delta('put', current_price, strike, T, self.rf_rate, sigma)
            
            # Check if delta is in target range
            if -0.3 <= delta <= -0.2:
                delta_diff = abs(delta + 0.25)  # Target -0.25 delta
                if delta_diff < min_delta_diff:
                    min_delta_diff = delta_diff
                    optimal_put = put
        
        return optimal_put

    def calculate_position_size(self, risk_amount, option):
        strike = float(option['strike_price'])
        max_loss = strike - self.data.Close[-1]  # Maximum loss if assigned
        return min(risk_amount / max_loss, 1)  # Limit to 1 contract for now

    def update_positions(self):
        for position in self.positions[:]:  # Create copy to safely remove items
            # Calculate profit/loss
            current_price = self.data.Close[-1]
            days_held = (self.data.index[-1] - position['entry_date']).days
            
            # Check profit target
            if current_price >= position['entry_price'] * (1 + self.profit_target):
                self.close_position(position)
                continue
                
            # Check stop loss
            if current_price <= position['entry_price'] * (1 - self.stop_loss):
                self.close_position(position)
                continue
                
            # Check expiration
            exp_date = datetime.strptime(position['expiration'], '%Y-%m-%d')
            if (exp_date - datetime.now()).days <= 0:
                self.close_position(position)

    def close_position(self, position):
        self.positions.remove(position)
        self.position.close()
        self.total_risk -= position['size'] * (position['strike'] - position['entry_price'])

    @staticmethod
    def RSI(series, period=14):
        delta = pd.Series(series).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ATR(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR using simple moving average
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr

# Load historical data
data = yf.download("AAPL", start="2021-01-01", end="2024-12-31", auto_adjust=True)

# Create a new DataFrame with the required structure
backtest_data = pd.DataFrame(index=data.index)
backtest_data['Open'] = data['Open']
backtest_data['High'] = data['High']
backtest_data['Low'] = data['Low']
backtest_data['Close'] = data['Close']
backtest_data['Volume'] = data['Volume']

# Ensure index is datetime
backtest_data.index = pd.to_datetime(backtest_data.index)

# Run backtest
bt = Backtest(backtest_data, ShortPutStrategy, cash=10000, commission=0.0035)
stats = bt.run()
print(stats)
bt.plot()
