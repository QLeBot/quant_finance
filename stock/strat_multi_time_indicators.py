import truststore
truststore.inject_into_ssl()

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from backtesting import Backtest, Strategy
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, KeltnerChannel
# If your 'ta' version lacks TSI, keep the fallback implementation below.
try:
    from ta.momentum import TSIIndicator
    HAS_TSI = True
except Exception:
    HAS_TSI = False

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
symbols = ["SPY"]
initial_cash = 10000
start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime.now() - datetime.timedelta(days=10)

def add_vwap_1h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3.0

    # VWAP resets daily (based on timestamp date)
    date_key = df['timestamp'].dt.floor('D')
    pv = tp * df['volume']
    df['vwap'] = pv.groupby(date_key).cumsum() / df['volume'].groupby(date_key).cumsum()
    return df

# Request data for different timeframes
request_params_1h = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_1h = stock_client.get_stock_bars(request_params_1h).df
all_data_1h = all_data_1h.reset_index()
all_data_1h['timestamp'] = pd.to_datetime(all_data_1h['timestamp'])

tmp_1h = all_data_1h.copy().set_index('timestamp')

# Alpaca bars are timezone-aware sometimes; resample expects consistent tz handling.
# If needed: tmp_1h.index = tmp_1h.index.tz_convert(None)

tmp_4h = tmp_1h.resample("4h", offset="30min").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "symbol": "last" if "symbol" in tmp_1h.columns else "last"
}).dropna(subset=["open","high","low","close"])

all_data_4h = tmp_4h.reset_index()

all_data_1h = add_vwap_1h(all_data_1h)

#request_params_4h = StockBarsRequest(
#    symbol_or_symbols=symbols,
#    timeframe=TimeFrame.Hour,
#    start=start_date,
#    end=end_date,
#    adjustment="split"
#)
#all_data_4h = stock_client.get_stock_bars(request_params_4h).df
#all_data_4h = all_data_4h.reset_index()

request_params_day = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_day = stock_client.get_stock_bars(request_params_day).df
all_data_day = all_data_day.reset_index()

request_params_week = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Week,
    start=start_date,
    end=end_date,
    adjustment="split"
)
all_data_week = stock_client.get_stock_bars(request_params_week).df
all_data_week = all_data_week.reset_index()

# Convert timestamps to datetime
all_data_4h['timestamp'] = pd.to_datetime(all_data_4h['timestamp'])
all_data_day['timestamp'] = pd.to_datetime(all_data_day['timestamp'])
all_data_week['timestamp'] = pd.to_datetime(all_data_week['timestamp'])

def get_risk_free_rate_tnx():
    """Fetches the latest U.S. 10-Year Treasury yield using yfinance."""
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100
        return latest_yield
    else:
        raise ValueError("No data found for ^TNX.")
    
def _tsi_fallback(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """
    Fallback TSI implementation if ta.momentum.TSIIndicator isn't available.
    TSI = 100 * EMA(EMA(m, r), s) / EMA(EMA(|m|, r), s), where m = diff(close)
    """
    m = close.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()

    abs_m = m.abs()
    abs_ema1 = abs_m.ewm(span=r, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=s, adjust=False).mean()

    tsi = 100 * (ema2 / abs_ema2)
    return tsi

def compute_indicators(df):
    """Compute technical indicators for a given dataframe."""
    #df['ma200'] = df['close'].rolling(window=200).mean()
    #df['rsi'] = RSIIndicator(df['close'], window=10).rsi()
    #macd = MACD(df['close'])
    #df['macd'] = macd.macd()
    #df['macd_signal'] = macd.macd_signal()
    #df['macd_hist'] = macd.macd_diff()
    #return df

    df = df.copy()

    # Basic
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = RSIIndicator(df['close'], window=10).rsi()

    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # --- Advanced additions ---
    # ADX (+DI / -DI)
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()

    # ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()

    # Keltner Channels
    kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=10)
    df['kc_mid'] = kc.keltner_channel_mband()
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low']  = kc.keltner_channel_lband()

    # TSI
    if HAS_TSI:
        df['tsi'] = TSIIndicator(close=df['close'], window_slow=25, window_fast=13).tsi()
    else:
        df['tsi'] = _tsi_fallback(df['close'], r=25, s=13)

    return df

timeframes = [all_data_1h, all_data_4h, all_data_day, all_data_week]
all_data = [compute_indicators(df) for df in timeframes]
all_data_1h, all_data_4h, all_data_day, all_data_week = all_data

class MultiTimeframeRSIMACDStrategy(Strategy):
    rsi_threshold = 30
    allow_short = False
    risk_per_trade = 0.01        # 1% of equity at risk
    atr_mult_stop = 2.0          # stop distance = 2.5 * ATR (1h)
    #atr_mult_tp = 3.0
    max_position_pct = 0.25      # never allocate >25% of equity
    min_hold_bars = 8   # e.g. 8 hours
    cooldown_bars = 4   # wait 4 bars after exit before new entry

    
    def init(self):
        # Convert the backtesting data to a DataFrame for merging
        df_1h = pd.DataFrame({
            'timestamp': self.data.index,
            'close': self.data.Close,
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'volume': self.data.Volume
        })

        self.last_entry_bar = -10**9
        self.last_exit_bar = -10**9

        
        # Initialize indicators with pandas Series
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x, index=self.data.index), window=10).rsi().values, self.data.Close, name='rsi')
        self.macd = self.I(lambda x: MACD(pd.Series(x, index=self.data.index)).macd().values, self.data.Close, name='macd')
        self.macd_signal = self.I(lambda x: MACD(pd.Series(x, index=self.data.index)).macd_signal().values, self.data.Close, name='macd_signal')
        self.macd_hist = self.I(lambda x: MACD(pd.Series(x, index=self.data.index)).macd_diff().values, self.data.Close, name='macd_hist')

        self.ema50 = self.I(lambda x: pd.Series(x, index=self.data.index).ewm(span=50, adjust=False).mean().values, self.data.Close, name="ema50")
        self.ema200 = self.I(lambda x: pd.Series(x, index=self.data.index).ewm(span=200, adjust=False).mean().values, self.data.Close, name="ema200")


        def _atr(high, low, close):
            h = pd.Series(high, index=self.data.index)
            l = pd.Series(low, index=self.data.index)
            c = pd.Series(close, index=self.data.index)
            return AverageTrueRange(h, l, c, window=14).average_true_range().values

        def _adx(high, low, close):
            h = pd.Series(high, index=self.data.index)
            l = pd.Series(low, index=self.data.index)
            c = pd.Series(close, index=self.data.index)
            a = ADXIndicator(h, l, c, window=14)
            return a.adx().values

        def _di_plus(high, low, close):
            h = pd.Series(high, index=self.data.index)
            l = pd.Series(low, index=self.data.index)
            c = pd.Series(close, index=self.data.index)
            a = ADXIndicator(h, l, c, window=14)
            return a.adx_pos().values

        def _di_minus(high, low, close):
            h = pd.Series(high, index=self.data.index)
            l = pd.Series(low, index=self.data.index)
            c = pd.Series(close, index=self.data.index)
            a = ADXIndicator(h, l, c, window=14)
            return a.adx_neg().values

        self.atr_1h = self.I(_atr, self.data.High, self.data.Low, self.data.Close, name="atr_1h")
        self.adx_1h = self.I(_adx, self.data.High, self.data.Low, self.data.Close, name="adx_1h")
        self.di_plus_1h = self.I(_di_plus, self.data.High, self.data.Low, self.data.Close, name="di_plus_1h")
        self.di_minus_1h = self.I(_di_minus, self.data.High, self.data.Low, self.data.Close, name="di_minus_1h")

        
        # Prepare higher timeframe data
        data_4h = all_data_4h.rename(columns={
            'rsi': 'rsi_4h', 'macd': 'macd_4h', 'macd_signal': 'macd_signal_4h', 'macd_hist': 'macd_hist_4h',
            'adx': 'adx_4h', 'di_plus': 'di_plus_4h', 'di_minus': 'di_minus_4h',
            'atr': 'atr_4h', 'kc_mid': 'kc_mid_4h', 'kc_high': 'kc_high_4h', 'kc_low': 'kc_low_4h',
            'tsi': 'tsi_4h'
        })

        data_day = all_data_day.rename(columns={
            'rsi': 'rsi_day', 'macd': 'macd_day', 'macd_signal': 'macd_signal_day', 'macd_hist': 'macd_hist_day',
            'adx': 'adx_day', 'di_plus': 'di_plus_day', 'di_minus': 'di_minus_day',
            'atr': 'atr_day', 'kc_mid': 'kc_mid_day', 'kc_high': 'kc_high_day', 'kc_low': 'kc_low_day',
            'tsi': 'tsi_day'
        })

        data_week = all_data_week.rename(columns={
            'rsi': 'rsi_week', 'macd': 'macd_week', 'macd_signal': 'macd_signal_week', 'macd_hist': 'macd_hist_week',
            'adx': 'adx_week', 'di_plus': 'di_plus_week', 'di_minus': 'di_minus_week',
            'atr': 'atr_week', 'kc_mid': 'kc_mid_week', 'kc_high': 'kc_high_week', 'kc_low': 'kc_low_week',
            'tsi': 'tsi_week'
        })

        # Columns you want to bring from higher TFs (indicators only)
        cols_4h = [
            'timestamp',
            'rsi_4h', 'macd_4h', 'macd_signal_4h', 'macd_hist_4h',
            'adx_4h', 'di_plus_4h', 'di_minus_4h',
            'atr_4h', 'kc_mid_4h', 'kc_high_4h', 'kc_low_4h',
            'tsi_4h'
        ]

        cols_day = [
            'timestamp',
            'rsi_day', 'macd_day', 'macd_signal_day', 'macd_hist_day',
            'adx_day', 'di_plus_day', 'di_minus_day',
            'atr_day', 'kc_mid_day', 'kc_high_day', 'kc_low_day',
            'tsi_day'
        ]

        cols_week = [
            'timestamp',
            'rsi_week', 'macd_week', 'macd_signal_week', 'macd_hist_week',
            'adx_week', 'di_plus_week', 'di_minus_week',
            'atr_week', 'kc_mid_week', 'kc_high_week', 'kc_low_week',
            'tsi_week'
        ]

        data_4h = data_4h[cols_4h].sort_values('timestamp')
        data_day = data_day[cols_day].sort_values('timestamp')
        data_week = data_week[cols_week].sort_values('timestamp')

        
        # Merge higher timeframes into one DataFrame
        self.multi_tf_data = pd.merge_asof(
            pd.merge_asof(
                pd.merge_asof(
                    df_1h.sort_values('timestamp'),
                    data_4h,
                    on='timestamp',
                    direction='backward',
                ),
                data_day,
                on='timestamp',
                direction='backward',
            ),
            data_week,
            on='timestamp',
            direction='backward',
        ).set_index('timestamp')


        # bring in 1H vwap computed in all_data_1h
        vwap_1h = all_data_1h[['timestamp', 'vwap']].sort_values('timestamp')

        self.multi_tf_data = pd.merge_asof(
            self.multi_tf_data.reset_index().sort_values('timestamp'),
            vwap_1h,
            on='timestamp',
            direction='backward'
        ).set_index('timestamp')

        # Set timestamp as index again
        #self.multi_tf_data.set_index('timestamp', inplace=True)
        
        # Print column names for debugging
        print("Available columns in multi_tf_data:")
        print(self.multi_tf_data.columns.tolist())

    def signal_confirmation(self, tf_data, rsi_key, macd_key, macd_signal_key, macd_hist_key, rsi_threshold):
        """Returns a tuple: (long_signal, short_signal) for a given timeframe."""
        if tf_data is None:
            return False, False

        rsi = tf_data[rsi_key]
        macd = tf_data[macd_key]
        macd_signal = tf_data[macd_signal_key]
        macd_hist = tf_data[macd_hist_key]

        long_signal = (rsi > rsi_threshold) and (macd > macd_signal)
        short_signal = (rsi < (100 - rsi_threshold)) and (macd < macd_signal)

        return long_signal, short_signal
    
    def signal_confirmation_v2(self, tf_data, rsi_key, macd_key, macd_signal_key, macd_hist_key, rsi_threshold):
        """Returns a tuple: (long_signal, short_signal) for a given timeframe."""
        if tf_data is None:
            return False, False

        rsi = tf_data[rsi_key]
        macd = tf_data[macd_key]
        macd_signal = tf_data[macd_signal_key]
        macd_hist = tf_data[macd_hist_key]

        rsi_cross = (rsi > rsi_threshold) & (rsi.shift(1) <= rsi_threshold)
        macd_cross = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
        macd_hist_rising = (macd_hist > macd_hist.shift(1))

        long = rsi_cross & macd_cross & macd_hist_rising


        return rsi, macd, macd_signal, macd_hist, rsi_cross, macd_cross
    
    def next(self):
        idx = len(self.data) - 1
        row = self.multi_tf_data.iloc[idx]

        # Regime (weekly)
        trend_long = pd.notna(row['macd_week']) and pd.notna(row['macd_signal_week']) and (row['macd_week'] > row['macd_signal_week'])

        # Day direction filter (optional)
        trend_up = pd.notna(row['di_plus_day']) and pd.notna(row['di_minus_day']) and (row['di_plus_day'] > row['di_minus_day'])
        adx_ok = pd.notna(row['adx_day']) and (row['adx_day'] > 15)   # relaxed from 20

        # 1H structure
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        ema50 = self.ema50[-1]
        prev_ema50 = self.ema50[-2]
        ema200 = self.ema200[-1]

        above_200 = price > ema200
        reclaim_50 = (price > ema50) and (prev_price <= prev_ema50)

        # 1H momentum "freshness" (lightweight)
        macd_hist_up = self.macd_hist[-1] > self.macd_hist[-2]

        # ---- Trailing stop management ----
        if self.position and self.position.is_long and self.trades:
            # Find the most recent OPEN trade (usually the last one)
            trade = next((t for t in reversed(self.trades) if t.is_open), None)
            if trade is not None:
                atr = self.atr_1h[-1]
                if np.isfinite(atr) and atr > 0:
                    new_sl = self.data.Close[-1] - 3.0 * atr
                    # Only raise stop (never loosen)
                    if trade.sl is None or new_sl > trade.sl:
                        trade.sl = new_sl

        # ---- Exit logic: hold winners in weekly uptrend ----
        exit_long = (not trend_long) or (not above_200)  # simple, strong
        if self.position and self.position.is_long:
            if idx - self.last_entry_bar < self.min_hold_bars:
                exit_long = False
            if exit_long:
                self.last_exit_bar = idx
                self.position.close()
                return

        # Cooldown after exit
        if idx - self.last_exit_bar < self.cooldown_bars:
            return

        # ---- Entry logic (less strict for higher exposure) ----
        long_confirmed = trend_long and above_200 and reclaim_50 and (adx_ok or (trend_up and macd_hist_up))
        if long_confirmed and not self.position:
            atr = self.atr_1h[-1]
            if not np.isfinite(atr) or atr <= 0:
                return

            stop_dist = self.atr_mult_stop * atr
            equity = self.equity
            risk_cash = equity * self.risk_per_trade
            size_risk = int(risk_cash / stop_dist)

            max_cash = equity * self.max_position_pct
            size_cap = int(max_cash / price)

            size = max(0, min(size_risk, size_cap))
            if size <= 0:
                return

            sl = price - stop_dist
            self.last_entry_bar = idx
            self.buy(size=size, sl=sl)  # NO TP


# Prepare data for backtesting
data_1h = all_data_1h.copy()
data_1h.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data_1h.set_index('timestamp', inplace=True)

# Run backtest
bt = Backtest(data_1h, MultiTimeframeRSIMACDStrategy, cash=initial_cash, commission=0.0005, finalize_trades=True)
results = bt.run()
print(results)

# Plot results
bt.plot(plot_equity=True, plot_return=True, plot_volume=True, plot_pl=True, plot_trades=True, show_legend=True, resample=True)