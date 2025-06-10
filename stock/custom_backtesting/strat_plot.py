import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import os
from dotenv import load_dotenv

class Data:
    """Class to handle all data functionality for the trading strategy."""
    
    def __init__(self, ticker, start_date, end_date):
        """Initialize the data with data and ticker symbol."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = None

    def yfinance_data(self):
        """Fetch stock data from yfinance."""
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(self.df.head())
        return pl.from_pandas(self.df.reset_index())
    
    def alpaca_data(self):
        """Fetch stock data from Alpaca."""
        self.client = self.load_environment()
        self.df = self.fetch_stock_data()
        print(self.df.head())
        return self.df
    
    def fetch_stock_data(self):
        """Fetch stock data from Alpaca."""
        request_params = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            timeframe=TimeFrame.Day,
            start=self.start_date,
            end=self.end_date,
            adjustment="split"
        )
        bars = self.client.get_stock_bars(request_params)
        print(bars.df.head())
        return pl.from_pandas(bars.df.reset_index())

    def load_environment(self):
        """Load environment variables and initialize Alpaca client."""
        load_dotenv()
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        return client

class Plotter:
    """Class to handle all plotting functionality for the trading strategy."""
    
    def __init__(self, df, ticker):
        """Initialize the plotter with data and ticker symbol."""
        self.df = df
        self.ticker = ticker
    
    def plot_all(self):
        """Plot all strategy visualizations in a single figure with four subplots."""
        import matplotlib.ticker as mtick
        # Adjust the height ratios so all subplots fit well
        fig, axs = plt.subplots(
            4, 1, figsize=(16, 14), sharex=True,
            gridspec_kw={'height_ratios': [2.5, 1.2, 1.2, 2.1], 'hspace': 0.25}
        )
        self.plot_price_and_signals(axs[0], axs[1])
        self.plot_volume_indicators(axs[2])
        self.plot_pv_drawdown(axs[3])
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
    
    def plot_price_and_signals(self, ax1, ax2):
        """Plot price data and trading signals on provided axes."""
        # First subplot: Open/Close
        ax1.plot(self.df["timestamp"], self.df["open"], label="Open Price", color="sandybrown", linestyle="--", alpha=0.5)
        ax1.plot(self.df["timestamp"], self.df["close"], label="Close Price", color="peachpuff", alpha=0.5)
        years = range(2018, datetime.now().year + 1)
        for year in years:
            first_day = self.df.filter(
                (pl.col("timestamp").dt.year() == year) & 
                (pl.col("timestamp").dt.month() == 1) & 
                (pl.col("timestamp").dt.day() == 1)
            )["timestamp"]
            if len(first_day) > 0:
                ax1.axvline(x=first_day[0], color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel(f"{self.ticker} Price")
        ax1.set_title(f"{self.ticker} Open and Close Price")
        buy_signals = self.df.filter(pl.col("signal") == 1)
        sell_signals = self.df.filter(pl.col("signal") == -1)
        ax1.scatter(buy_signals["timestamp"], buy_signals["close"], marker='^', color='green', s=100, label='Buy Signal', alpha=0.5)
        ax1.scatter(sell_signals["timestamp"], sell_signals["close"], marker='v', color='red', s=100, label='Sell Signal', alpha=0.5)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Second subplot: Indicators and signals
        # Fill between MIN20 and MAX20 (light grey)
        ax2.fill_between(
            self.df["timestamp"].to_numpy(),
            self.df["min20"].to_numpy(),
            self.df["max20"].to_numpy(),
            color="lightgrey",
            alpha=0.5,
            label="MIN20-MAX20 Range"
        )

        # Fill between MIN50 and MAX50 (darker grey)
        ax2.fill_between(
            self.df["timestamp"].to_numpy(),
            self.df["min50"].to_numpy(),
            self.df["max50"].to_numpy(),
            color="grey",
            alpha=0.5,
            label="MIN50-MAX50 Range"
        )

        # Plot the main lines
        ax2.plot(self.df["timestamp"], self.df["close"], label="Close Price", color="black")
        ax2.plot(self.df["timestamp"], self.df["sma20"], label="SMA 20", color="tab:blue")
        ax2.plot(self.df["timestamp"], self.df["sma50"], label="SMA 50", color="tab:orange")

        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()

    def plot_volume_indicators(self, ax):
        """Plot volume indicators on the provided axis."""
        # Plot OBV
        ax.plot(self.df["timestamp"], self.df["obv"], label="OBV", color="blue", alpha=0.7)
        
        # Plot A/D Line
        ax.plot(self.df["timestamp"], self.df["ad_line"], label="A/D Line", color="green", alpha=0.7)
        
        # Plot Volume Z-score
        ax2 = ax.twinx()
        ax2.plot(self.df["timestamp"], self.df["volume_zscore"], label="Volume Z-score", color="red", alpha=0.7)
        ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.3)
        ax2.axhline(y=-2.0, color="red", linestyle="--", alpha=0.3)
        
        # Add volume bars
        ax.bar(self.df["timestamp"], self.df["volume"], alpha=0.3, color="gray", label="Volume")
        
        # Add market regime
        ax.fill_between(
            self.df["timestamp"],
            0,
            self.df["market_regime"] * self.df["volume"].max(),
            alpha=0.1,
            color="yellow",
            label="Trending Market"
        )
        
        # Formatting
        ax.set_ylabel("Volume Indicators")
        ax2.set_ylabel("Volume Z-score")
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        # Add title
        ax.set_title("Volume Analysis")

    def plot_pv_drawdown(self, ax):
        """Plot portfolio value, buy & hold, and drawdown as a single subplot with dual y-axes."""
        import matplotlib.ticker as mtick
        # Portfolio value (left y-axis)
        ax.plot(self.df["timestamp"].to_numpy(), self.df["portfolio_value"].to_numpy(), color="green", label="Strategy Portfolio Value")
        ax.plot(self.df["timestamp"].to_numpy(), self.df["bh_portfolio_value"].to_numpy(), color="blue", linestyle="--", label="Buy & Hold Value")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(True, linestyle='--', alpha=0.3)
        # Drawdown (right y-axis)
        ax2 = ax.twinx()
        ax2.fill_between(
            self.df["timestamp"].to_numpy(),
            self.df["drawdown"].to_numpy() * 100,
            0,
            color="red",
            alpha=0.2,
            label="Drawdown"
        )
        ax2.set_ylabel("Drawdown (%)", color="red")
        ax2.tick_params(axis='y', labelcolor="red")
        ax2.set_ylim(-60, 0)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        # Legends and layout
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
        ax.set_xlabel("Date")

class Strategy:
    """Class to handle all strategy functionality."""
    
    def __init__(self, df, ticker, initial_capital=10000):
        """Initialize the strategy with data and ticker symbol."""
        self.df = df
        self.ticker = ticker
        self.initial_capital = initial_capital

    def calculate_indicators(self):
        """Calculate technical indicators for the strategy."""
        # Basic moving averages
        self.df = self.df.with_columns([
            pl.col("close").rolling_mean(window_size=20).alias("sma20"),
            pl.col("close").rolling_mean(window_size=50).alias("sma50"),
            pl.col("close").rolling_max(window_size=20).alias("max20"),
            pl.col("close").rolling_max(window_size=50).alias("max50"),
            pl.col("close").rolling_min(window_size=20).alias("min20"),
            pl.col("close").rolling_min(window_size=50).alias("min50"),
        ])

        # RSI calculation
        delta = self.df.with_columns(pl.col("close").diff().alias("delta"))
        gain = delta.with_columns(pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"))
        loss = delta.with_columns(pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"))
        
        avg_gain = gain.with_columns(pl.col("gain").rolling_mean(window_size=14).alias("avg_gain"))
        avg_loss = loss.with_columns(pl.col("loss").rolling_mean(window_size=14).alias("avg_loss"))
        
        rs = avg_gain.join(avg_loss, on="timestamp").with_columns(
            (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs")
        )
        
        self.df = self.df.join(rs.select(["timestamp", "rs"]), on="timestamp").with_columns(
            (100 - (100 / (1 + pl.col("rs")))).alias("rsi")
        )

        #print(f"after rsi: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # MACD calculation - First calculate EMAs
        self.df = self.df.with_columns([
            pl.col("close").rolling_mean(window_size=12).alias("ema12"),
            pl.col("close").rolling_mean(window_size=26).alias("ema26")
        ])
        
        # Then calculate MACD and signal line in separate steps
        self.df = self.df.with_columns(
            (pl.col("ema12") - pl.col("ema26")).alias("macd")
        )
        
        self.df = self.df.with_columns(
            pl.col("macd").rolling_mean(window_size=9).alias("macd_signal")
        )

        #print(f"after macd: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Bollinger Bands
        self.df = self.df.with_columns([
            pl.col("close").rolling_mean(window_size=20).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).alias("bb_std")
        ])
        
        self.df = self.df.with_columns([
            (pl.col("bb_middle") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("bb_std")).alias("bb_lower")
        ])

        #print(f"after bb: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # ATR calculation
        self.df = self.df.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr1"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr2"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr3")
        ])
        
        self.df = self.df.with_columns(
            pl.max_horizontal(["tr1", "tr2", "tr3"]).alias("tr")
        )
        
        self.df = self.df.with_columns(
            pl.col("tr").rolling_mean(window_size=14).alias("atr")
        )

        #print(f"after atr: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Volume indicators - Calculate in separate steps
        self.df = self.df.with_columns(
            pl.col("volume").rolling_mean(window_size=20).alias("volume_sma")
        )
        
        self.df = self.df.with_columns(
            (pl.col("volume") / pl.col("volume_sma")).alias("volume_ratio")
        )

        #print(f"after volume: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # ADX calculation
        # True Range
        self.df = self.df.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr_high"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr_low")
        ])
        
        self.df = self.df.with_columns(
            pl.max_horizontal(["tr", "tr_high", "tr_low"]).alias("true_range")
        )

        #print(f"after adx: \n {self.df.select(['timestamp', 'close']).head(10)}")
        
        # Directional Movement
        self.df = self.df.with_columns([
            (pl.col("high") - pl.col("high").shift(1)).alias("high_diff"),
            (pl.col("low").shift(1) - pl.col("low")).alias("low_diff")
        ])
        
        # +DM and -DM
        self.df = self.df.with_columns([
            pl.when(pl.col("high_diff") > pl.col("low_diff"))
            .then(pl.max_horizontal([pl.col("high_diff"), 0]))
            .otherwise(0)
            .alias("plus_dm"),
            
            pl.when(pl.col("low_diff") > pl.col("high_diff"))
            .then(pl.max_horizontal([pl.col("low_diff"), 0]))
            .otherwise(0)
            .alias("minus_dm")
        ])

        #print(f"after directional movement: \n {self.df.select(['timestamp', 'close']).head(10)}")
        
        # Smoothed TR and DM
        period = 14
        self.df = self.df.with_columns([
            pl.col("true_range").rolling_mean(window_size=period).alias("smoothed_tr"),
            pl.col("plus_dm").rolling_mean(window_size=period).alias("smoothed_plus_dm"),
            pl.col("minus_dm").rolling_mean(window_size=period).alias("smoothed_minus_dm")
        ])
        
        # +DI and -DI
        self.df = self.df.with_columns([
            (100 * pl.col("smoothed_plus_dm") / pl.col("smoothed_tr")).alias("plus_di"),
            (100 * pl.col("smoothed_minus_dm") / pl.col("smoothed_tr")).alias("minus_di")
        ])
        
        # DX and ADX
        self.df = self.df.with_columns(
            (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / 
             (pl.col("plus_di") + pl.col("minus_di"))).alias("dx")
        )
        
        self.df = self.df.with_columns(
            pl.col("dx").rolling_mean(window_size=period).alias("adx")
        )

        #print(f"after tr, plus_dm, minus_dm: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Kaufman Adaptive Moving Average (KAMA)
        # Efficiency Ratio (ER)
        change = self.df.with_columns(
            (pl.col("close") - pl.col("close").shift(10)).abs().alias("change")
        )
        
        volatility = self.df.with_columns(
            pl.col("close").diff().abs().rolling_sum(window_size=10).alias("volatility")
        )
        
        change = change.select(["timestamp", "change"])
        volatility = volatility.select(["timestamp", "volatility"])
        
        self.df = self.df.join(
            change, 
            on="timestamp"
        )
        self.df = self.df.join(
            volatility, 
            on="timestamp"
        )
        
        # Calculate ER with safety checks
        self.df = self.df.with_columns(
            pl.when(pl.col("volatility") > 0)
            .then(pl.col("change") / pl.col("volatility"))
            .otherwise(0)
            .alias("er")
        )
        
        # Calculate smoothing constant with bounds
        self.df = self.df.with_columns(
            pl.when(pl.col("er").is_not_null())
            .then(
                ((pl.col("er") * (2.0/(2+1) - 2.0/(30+1)) + 2.0/(30+1))**2)
                .clip(0.01, 1.0)  # Bound between 0.01 and 1.0
            )
            .otherwise(0.5)  # Default value if ER is null
            .alias("sc")
        )
        
        # Initialize KAMA with first close price
        self.df = self.df.with_columns(
            pl.col("close").alias("kama")
        )
        
        # Iterative KAMA calculation with safety checks
        for i in range(1, len(self.df)):
            if i > 0:  # Skip first row
                prev_kama = self.df["kama"][i-1]
                curr_close = self.df["close"][i]
                sc = self.df["sc"][i]
                
                # Safety check for None values
                if prev_kama is not None and curr_close is not None and sc is not None:
                    kama = prev_kama + sc * (curr_close - prev_kama)
                    self.df = self.df.with_row_index("row_idx").with_columns(
                        pl.when(pl.col("row_idx") == i)
                        .then(kama)
                        .otherwise(pl.col("kama"))
                        .alias("kama")
                    ).drop("row_idx")
                else:
                    # If any value is None, use the current close price
                    self.df = self.df.with_row_index("row_idx").with_columns(
                        pl.when(pl.col("row_idx") == i)
                        .then(curr_close if curr_close is not None else prev_kama)
                        .otherwise(pl.col("kama"))
                        .alias("kama")
                    ).drop("row_idx")

        #print(f"after kama: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Volume Indicators
        # OBV (On-Balance Volume)
        self.df = self.df.with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-pl.col("volume"))
            .otherwise(0)
            .alias("obv_delta")
        ])
        
        self.df = self.df.with_columns(
            pl.col("obv_delta").cum_sum().alias("obv")
        )

        #print(f"after obv: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Accumulation/Distribution Line
        # Calculate all components in a single step
        self.df = self.df.with_columns([
            # Money Flow Multiplier
            pl.when((pl.col("high") - pl.col("low")) != 0)
            .then(((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low")))
            .otherwise(0)
            .alias("mf_multiplier"),
            
            # Money Flow Volume
            pl.when((pl.col("high") - pl.col("low")) != 0)
            .then(pl.col("volume") * ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low")))
            .otherwise(0)
            .alias("mf_volume")
        ])
        
        # Calculate A/D Line
        self.df = self.df.with_columns(
            pl.col("mf_volume").cum_sum().alias("ad_line")
        )

        # Volume Spike Detection
        self.df = self.df.with_columns([
            pl.col("volume").rolling_mean(window_size=20).alias("volume_ma"),
            pl.col("volume").rolling_std(window_size=20).alias("volume_std")
        ])
        
        self.df = self.df.with_columns(
            ((pl.col("volume") - pl.col("volume_ma")) / pl.col("volume_std")).alias("volume_zscore")
        )

        #print(f"after ad_line: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Mean Reversion Indicators
        # Bollinger Bands Z-score
        self.df = self.df.with_columns([
            ((pl.col("close") - pl.col("bb_middle")) / pl.col("bb_std")).alias("bb_zscore")
        ])

        # RSI Z-score
        self.df = self.df.with_columns([
            pl.col("rsi").rolling_mean(window_size=20).alias("rsi_ma"),
            pl.col("rsi").rolling_std(window_size=20).alias("rsi_std")
        ])
        
        self.df = self.df.with_columns(
            ((pl.col("rsi") - pl.col("rsi_ma")) / pl.col("rsi_std")).alias("rsi_zscore")
        )

        #print(f"after bb_zscore, rsi_zscore: \n {self.df.select(['timestamp', 'close']).head(10)}")

        # Market Regime Detection (Trending vs Sideways)
        self.df = self.df.with_columns([
            pl.col("adx").rolling_mean(window_size=20).alias("adx_ma")
        ])
        
        self.df = self.df.with_columns(
            pl.when(pl.col("adx_ma") > 25)
            .then(1)  # Trending market
            .otherwise(0)  # Sideways market
            .alias("market_regime")
        )

        #print(f"after market_regime: \n {self.df.select(['timestamp', 'close']).head(10)}")

        return self.df

    def calculate_portfolio_metrics(self):
        """Calculate portfolio value, drawdown, and buy & hold benchmark."""
        # Calculate daily returns
        self.df = self.df.with_columns([
            pl.col("close").pct_change().fill_null(0).alias("daily_return")
        ])
        
        # Calculate strategy returns (only when in position)
        self.df = self.df.with_columns([
            (pl.col("daily_return") * pl.col("position")).fill_null(0).alias("strategy_return")
        ])
        
        # Calculate cumulative returns, fill nulls with 1.0
        self.df = self.df.with_columns([
            (1 + pl.col("strategy_return")).cum_prod().fill_null(1.0).alias("cumulative_return")
        ])
        
        # Calculate portfolio value
        self.df = self.df.with_columns([
            (self.initial_capital * pl.col("cumulative_return")).fill_null(self.initial_capital).alias("portfolio_value")
        ])
        
        # Calculate buy & hold benchmark
        self.df = self.df.with_columns([
            (1 + pl.col("daily_return")).cum_prod().fill_null(1.0).alias("bh_cumulative_return")
        ])
        self.df = self.df.with_columns([
            (self.initial_capital * pl.col("bh_cumulative_return")).fill_null(self.initial_capital).alias("bh_portfolio_value")
        ])
        
        # Calculate peak value
        self.df = self.df.with_columns([
            pl.col("portfolio_value").cum_max().fill_null(self.initial_capital).alias("peak_value")
        ])
        
        # Calculate drawdown, fill nulls with 0
        self.df = self.df.with_columns([
            ((pl.col("portfolio_value") - pl.col("peak_value")) / pl.col("peak_value")).fill_null(0.0).alias("drawdown")
        ])
        
        # Calculate max drawdown, fill nulls with 0
        self.df = self.df.with_columns([
            pl.col("drawdown").cum_min().fill_null(0.0).alias("max_drawdown")
        ])
        
        return self.df

    def generate_signals(self):
        """Generate buy/sell signals based on technical indicators."""
        # Initialize signal columns
        self.df = self.df.with_columns([
            pl.lit(0).alias("signal"),  # 0: no signal, 1: buy, -1: sell
            pl.lit(0.0).alias("position"),  # 0: no position, 1: long
            pl.lit(0.0).alias("pnl"),  # Profit and Loss
            pl.lit(0.0).alias("entry_price")  # Track entry price for PnL calculation
        ])
        
        # Generate signals using previous day's indicators
        for i in range(2, len(self.df)):
            # D-2 data for crossover detection
            d2_close = self.df["close"][i-2]
            d2_sma20 = self.df["sma20"][i-2]
            
            # D-1 data for signal generation
            d1_close = self.df["close"][i-1]
            d1_sma20 = self.df["sma20"][i-1]
            d1_sma50 = self.df["sma50"][i-1]
            d1_min20 = self.df["min20"][i-1]
            d1_rsi = self.df["rsi"][i-1]
            d1_macd = self.df["macd"][i-1]
            d1_macd_signal = self.df["macd_signal"][i-1]
            d1_bb_upper = self.df["bb_upper"][i-1]
            d1_bb_lower = self.df["bb_lower"][i-1]
            d1_volume_ratio = self.df["volume_ratio"][i-1]
            d1_adx = self.df["adx"][i-1]
            d1_plus_di = self.df["plus_di"][i-1]
            d1_minus_di = self.df["minus_di"][i-1]
            d1_kama = self.df["kama"][i-1]
            
            # Add new indicators
            d1_obv = self.df["obv"][i-1]
            d1_ad_line = self.df["ad_line"][i-1]
            d1_volume_zscore = self.df["volume_zscore"][i-1]
            d1_bb_zscore = self.df["bb_zscore"][i-1]
            d1_rsi_zscore = self.df["rsi_zscore"][i-1]
            d1_market_regime = self.df["market_regime"][i-1]
            
            # Skip if any required values are None
            if None in (d2_close, d2_sma20, d1_close, d1_sma20, d1_sma50, d1_min20,
                       d1_rsi, d1_macd, d1_macd_signal, d1_bb_upper, d1_bb_lower, 
                       d1_volume_ratio, d1_adx, d1_plus_di, d1_minus_di, d1_kama,
                       d1_obv, d1_ad_line, d1_volume_zscore, d1_bb_zscore, d1_rsi_zscore,
                       d1_market_regime):
                continue
            
            # Enhanced buy conditions
            if (
                d1_rsi < 70 and  # Not overbought
                d1_macd > d1_macd_signal and  # MACD crossover
                d1_volume_ratio > 1.2 and  # Above average volume
                d1_adx > 25 and  # Strong trend
                d1_close > d1_kama  # Price above KAMA
                ):
                
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(1)
                    .otherwise(pl.col("signal"))
                    .alias("signal")
                ).drop("row_idx")
            
            # Enhanced sell conditions
            elif (
                d1_rsi > 70 and  # Overbought
                d1_macd < d1_macd_signal and  # MACD crossover
                d1_volume_ratio < 0.8 and  # Below average volume
                d1_adx > 25 and  # Strong trend
                d1_close < d1_kama  # Price below KAMA
                ):
                
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(-1)
                    .otherwise(pl.col("signal"))
                    .alias("signal")
                ).drop("row_idx")
        
        # Calculate positions and PnL using next day's close price for execution
        position = 0
        entry_price = 0
        
        for i in range(1, len(self.df)):
            signal = self.df["signal"][i]
            execution_price = self.df["close"][i]  # Use close price for execution
            
            if signal == 1 and position == 0:  # Buy
                position = 1
                entry_price = execution_price
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(entry_price)
                    .otherwise(pl.col("entry_price"))
                    .alias("entry_price")
                ).drop("row_idx")
            elif signal == -1 and position == 1:  # Sell
                position = 0
                pnl = (execution_price - entry_price) / entry_price
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(pnl)
                    .otherwise(pl.col("pnl"))
                    .alias("pnl")
                ).drop("row_idx")
            
            self.df = self.df.with_row_index("row_idx").with_columns(
                pl.when(pl.col("row_idx") == i)
                .then(position)
                .otherwise(pl.col("position"))
                .alias("position")
            ).drop("row_idx")
        
        # Calculate portfolio metrics after generating signals
        self.calculate_portfolio_metrics()
        
        return self.df

    def print_strategy_performance(self):
        """Print strategy performance metrics."""
        buy_signals = self.df.filter(pl.col("signal") == 1)
        total_trades = len(buy_signals)
        winning_trades = len(self.df.filter(pl.col("pnl") > 0))
        total_pnl = self.df["pnl"].sum()
        
        # Calculate additional metrics
        final_portfolio_value = self.df["portfolio_value"].last()
        max_drawdown = self.df["max_drawdown"].min()
        bh_final_portfolio_value = self.df["bh_portfolio_value"].last()
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital if final_portfolio_value is not None else 0
        
        print(f"\nStrategy Performance:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}" if final_portfolio_value is not None else "Final Portfolio Value: N/A")
        print(f"Buy & Hold Final Portfolio Value: ${bh_final_portfolio_value:,.2f}" if bh_final_portfolio_value is not None else "Buy & Hold Final Portfolio Value: N/A")
        print(f"Total Return: {total_return*100:.2f}%")
        if max_drawdown is not None:
            print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        else:
            print("Max Drawdown: N/A")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        if total_trades > 0:
            print(f"Win Rate: {winning_trades/total_trades*100:.2f}%")
        else:
            print("Win Rate: N/A (No trades executed)")
        print(f"Total PnL: {total_pnl*100:.2f}%" if total_pnl is not None else "Total PnL: N/A")

def main():
    """Main function to run the strategy."""
    # Parameters
    ticker = "SPY"
    start_date = "2007-01-01"
    end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    initial_capital = 10000
    
    # Fetch and process data
    data = Data(ticker, start_date, end_date)
    df = data.alpaca_data()
    
    # Initialize strategy and calculate indicators
    strategy = Strategy(df, ticker, initial_capital)
    strategy.calculate_indicators()
    #print(strategy.df.select(['timestamp', 'close']).head(10))
    #print(strategy.df['close'].min(), strategy.df['close'].max())
    strategy.generate_signals()
    
    # Create plotter and plot results
    plotter = Plotter(strategy.df, ticker)
    plotter.plot_all()
    
    # Print performance metrics
    strategy.print_strategy_performance()

if __name__ == "__main__":
    main()

