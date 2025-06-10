import polars as pl
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy implementation."""
    
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
            # D-1 data for signal generation
            d1_close = self.df["close"][i-1]
            d1_rsi = self.df["rsi"][i-1]
            d1_bb_upper = self.df["bb_upper"][i-1]
            d1_bb_lower = self.df["bb_lower"][i-1]
            d1_bb_zscore = self.df["bb_zscore"][i-1]
            d1_rsi_zscore = self.df["rsi_zscore"][i-1]
            d1_volume_zscore = self.df["volume_zscore"][i-1]
            d1_market_regime = self.df["market_regime"][i-1]
            
            # Skip if any required values are None
            if None in (d1_close, d1_rsi, d1_bb_upper, d1_bb_lower, 
                       d1_bb_zscore, d1_rsi_zscore, d1_volume_zscore,
                       d1_market_regime):
                continue
            
            # Buy conditions for mean reversion
            if (
                d1_bb_zscore < -2.0 and  # Price significantly below mean
                d1_rsi < 30 and  # Oversold
                d1_volume_zscore > 1.0 and  # Above average volume
                d1_market_regime == 0  # Sideways market
                ):
                
                self.df = self.df.with_row_index("row_idx").with_columns(
                    pl.when(pl.col("row_idx") == i)
                    .then(1)
                    .otherwise(pl.col("signal"))
                    .alias("signal")
                ).drop("row_idx")
            
            # Sell conditions for mean reversion
            elif (
                d1_bb_zscore > 2.0 and  # Price significantly above mean
                d1_rsi > 70 and  # Overbought
                d1_volume_zscore > 1.0 and  # Above average volume
                d1_market_regime == 0  # Sideways market
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