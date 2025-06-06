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
        return pl.from_pandas(self.df.reset_index())
    
    def alpaca_data(self):
        """Fetch stock data from Alpaca."""
        self.client = self.load_environment()
        self.df = self.fetch_stock_data()
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
        """Plot all strategy visualizations in a single figure with three subplots."""
        import matplotlib.ticker as mtick
        fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        self.plot_price_and_signals(axs[0], axs[1])
        self.plot_pv_drawdown(axs[2])
        plt.tight_layout()
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

        # Bollinger Bands
        self.df = self.df.with_columns([
            pl.col("close").rolling_mean(window_size=20).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).alias("bb_std")
        ])
        
        self.df = self.df.with_columns([
            (pl.col("bb_middle") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("bb_std")).alias("bb_lower")
        ])

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

        # Volume indicators - Calculate in separate steps
        self.df = self.df.with_columns(
            pl.col("volume").rolling_mean(window_size=20).alias("volume_sma")
        )
        
        self.df = self.df.with_columns(
            (pl.col("volume") / pl.col("volume_sma")).alias("volume_ratio")
        )

        return self.df

    def calculate_portfolio_metrics(self):
        """Calculate portfolio value, drawdown, and buy & hold benchmark."""
        # Calculate daily returns
        self.df = self.df.with_columns([
            pl.col("close").pct_change().alias("daily_return")
        ])
        
        # Calculate strategy returns (only when in position)
        self.df = self.df.with_columns([
            (pl.col("daily_return") * pl.col("position")).alias("strategy_return")
        ])
        
        # Calculate cumulative returns, fill nulls with 0
        self.df = self.df.with_columns([
            (1 + pl.col("strategy_return")).cum_prod().fill_null(1.0).alias("cumulative_return")
        ])
        
        # Calculate portfolio value
        self.df = self.df.with_columns([
            (self.initial_capital * pl.col("cumulative_return")).alias("portfolio_value")
        ])
        
        # Calculate buy & hold benchmark
        self.df = self.df.with_columns([
            (1 + pl.col("daily_return")).cum_prod().fill_null(1.0).alias("bh_cumulative_return")
        ])
        self.df = self.df.with_columns([
            (self.initial_capital * pl.col("bh_cumulative_return")).alias("bh_portfolio_value")
        ])
        
        # Calculate peak value
        self.df = self.df.with_columns([
            pl.col("portfolio_value").cum_max().alias("peak_value")
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
            
            # Skip if any required values are None
            if None in (d2_close, d2_sma20, d1_close, d1_sma20, d1_sma50, d1_min20,
                       d1_rsi, d1_macd, d1_macd_signal, d1_bb_upper, d1_bb_lower, d1_volume_ratio):
                continue
            
            # Enhanced buy conditions
            if (d1_close > d1_sma20 and 
                d2_close <= d2_sma20 or
                #d1_close > d1_sma50 or
                #d1_close > d1_min20 and
                d1_rsi < 70 and  # Not overbought
                d1_macd > d1_macd_signal and  # MACD crossover
                #d1_close < d1_bb_upper and  # Not above upper Bollinger Band
                d1_volume_ratio > 1.2):  # Above average volume
                
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(1)
                    .otherwise(pl.col("signal"))
                    .alias("signal")
                ).drop("row_idx")
            
            # Enhanced sell conditions
            elif (
                #(d1_close < d1_sma20 and 
                #   d2_close >= d2_sma20) or
                  #d1_close < d1_min20 or
                  d1_rsi > 70 and  # Overbought
                  d1_macd < d1_macd_signal and  # MACD crossover
                  #d1_close > d1_bb_upper or  # Above upper Bollinger Band
                  d1_volume_ratio < 0.8):  # Below average volume
                
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(-1)
                    .otherwise(pl.col("signal"))
                    .alias("signal")
                ).drop("row_idx")
        
        # Calculate positions and PnL using next day's open price for execution
        position = 0
        entry_price = 0
        
        for i in range(len(self.df)):
            signal = self.df["signal"][i]
            execution_price = self.df["open"][i]  # Use open price for execution
            
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
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        print(f"\nStrategy Performance:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Buy & Hold Final Portfolio Value: ${bh_final_portfolio_value:,.2f}")
        print(f"Total Return: {total_return*100:.2f}%")
        if max_drawdown is not None:
            print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        else:
            print("Max Drawdown: N/A")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {winning_trades/total_trades*100:.2f}%")
        print(f"Total PnL: {total_pnl*100:.2f}%")

def main():
    """Main function to run the strategy."""
    # Parameters
    ticker = "SPY"
    start_date = "2018-01-01"
    end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    initial_capital = 10000
    
    # Fetch and process data
    data = Data(ticker, start_date, end_date)
    df = data.alpaca_data()
    
    # Initialize strategy and calculate indicators
    strategy = Strategy(df, ticker, initial_capital)
    strategy.calculate_indicators()
    strategy.generate_signals()
    
    # Create plotter and plot results
    plotter = Plotter(strategy.df, ticker)
    plotter.plot_all()
    
    # Print performance metrics
    strategy.print_strategy_performance()

if __name__ == "__main__":
    main()

