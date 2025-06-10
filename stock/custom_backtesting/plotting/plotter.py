import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime

class Plotter:
    """Class to handle all plotting functionality for the trading strategy."""
    
    def __init__(self, strategies_data, ticker):
        """Initialize the plotter with data and ticker symbol.
        
        Args:
            strategies_data (dict): Dictionary of strategy names and their dataframes
            ticker (str): Ticker symbol
        """
        self.strategies_data = strategies_data
        self.ticker = ticker
        self.colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    def plot_all(self):
        """Plot all strategy visualizations in a single figure with four subplots."""
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
        df = list(self.strategies_data.values())[0]  # Use first strategy's data for price
        ax1.plot(df["timestamp"], df["open"], label="Open Price", color="sandybrown", linestyle="--", alpha=0.5)
        ax1.plot(df["timestamp"], df["close"], label="Close Price", color="peachpuff", alpha=0.5)
        
        years = range(2018, datetime.now().year + 1)
        for year in years:
            first_day = df.filter(
                (pl.col("timestamp").dt.year() == year) & 
                (pl.col("timestamp").dt.month() == 1) & 
                (pl.col("timestamp").dt.day() == 1)
            )["timestamp"]
            if len(first_day) > 0:
                ax1.axvline(x=first_day[0], color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_ylabel(f"{self.ticker} Price")
        ax1.set_title(f"{self.ticker} Open and Close Price")
        
        # Plot signals for each strategy
        for i, (strategy_name, strategy_df) in enumerate(self.strategies_data.items()):
            color = self.colors[i % len(self.colors)]
            buy_signals = strategy_df.filter(pl.col("signal") == 1)
            sell_signals = strategy_df.filter(pl.col("signal") == -1)
            ax1.scatter(buy_signals["timestamp"], buy_signals["close"], 
                       marker='^', color=color, s=100, 
                       label=f'{strategy_name} Buy', alpha=0.5)
            ax1.scatter(sell_signals["timestamp"], sell_signals["close"], 
                       marker='v', color=color, s=100, 
                       label=f'{strategy_name} Sell', alpha=0.5)
        
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Second subplot: Indicators and signals
        # Fill between MIN20 and MAX20 (light grey)
        ax2.fill_between(
            df["timestamp"].to_numpy(),
            df["min20"].to_numpy(),
            df["max20"].to_numpy(),
            color="lightgrey",
            alpha=0.5,
            label="MIN20-MAX20 Range"
        )

        # Fill between MIN50 and MAX50 (darker grey)
        ax2.fill_between(
            df["timestamp"].to_numpy(),
            df["min50"].to_numpy(),
            df["max50"].to_numpy(),
            color="grey",
            alpha=0.5,
            label="MIN50-MAX50 Range"
        )

        # Plot the main lines
        ax2.plot(df["timestamp"], df["close"], label="Close Price", color="black")
        ax2.plot(df["timestamp"], df["sma20"], label="SMA 20", color="tab:blue")
        ax2.plot(df["timestamp"], df["sma50"], label="SMA 50", color="tab:orange")

        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()

    def plot_volume_indicators(self, ax):
        """Plot volume indicators on the provided axis."""
        df = list(self.strategies_data.values())[0]  # Use first strategy's data
        
        # Plot OBV
        ax.plot(df["timestamp"], df["obv"], label="OBV", color="blue", alpha=0.7)
        
        # Plot A/D Line
        ax.plot(df["timestamp"], df["ad_line"], label="A/D Line", color="green", alpha=0.7)
        
        # Plot Volume Z-score
        ax2 = ax.twinx()
        ax2.plot(df["timestamp"], df["volume_zscore"], label="Volume Z-score", color="red", alpha=0.7)
        ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.3)
        ax2.axhline(y=-2.0, color="red", linestyle="--", alpha=0.3)
        
        # Add volume bars
        ax.bar(df["timestamp"], df["volume"], alpha=0.3, color="gray", label="Volume")
        
        # Add market regime
        ax.fill_between(
            df["timestamp"],
            0,
            df["market_regime"] * df["volume"].max(),
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
        # Portfolio value (left y-axis)
        for i, (strategy_name, strategy_df) in enumerate(self.strategies_data.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(strategy_df["timestamp"].to_numpy(), 
                   strategy_df["portfolio_value"].to_numpy(), 
                   color=color, 
                   label=f"{strategy_name} Portfolio Value")
            
            if i == 0:  # Only plot buy & hold once
                ax.plot(strategy_df["timestamp"].to_numpy(), 
                       strategy_df["bh_portfolio_value"].to_numpy(), 
                       color="black", 
                       linestyle="--", 
                       label="Buy & Hold Value")
        
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Drawdown (right y-axis)
        ax2 = ax.twinx()
        for i, (strategy_name, strategy_df) in enumerate(self.strategies_data.items()):
            color = self.colors[i % len(self.colors)]
            ax2.fill_between(
                strategy_df["timestamp"].to_numpy(),
                strategy_df["drawdown"].to_numpy() * 100,
                0,
                color=color,
                alpha=0.2,
                label=f"{strategy_name} Drawdown"
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