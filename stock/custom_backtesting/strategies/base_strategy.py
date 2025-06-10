import polars as pl

class BaseStrategy:
    """Base class for all trading strategies."""
    
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

        # MACD calculation
        self.df = self.df.with_columns([
            pl.col("close").rolling_mean(window_size=12).alias("ema12"),
            pl.col("close").rolling_mean(window_size=26).alias("ema26")
        ])
        
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

        # Volume indicators
        self.df = self.df.with_columns(
            pl.col("volume").rolling_mean(window_size=20).alias("volume_sma")
        )
        
        self.df = self.df.with_columns(
            (pl.col("volume") / pl.col("volume_sma")).alias("volume_ratio")
        )

        # ADX calculation
        self.df = self.df.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr_high"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr_low")
        ])
        
        self.df = self.df.with_columns(
            pl.max_horizontal(["tr", "tr_high", "tr_low"]).alias("true_range")
        )
        
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

        # Kaufman Adaptive Moving Average (KAMA)
        change = self.df.with_columns(
            (pl.col("close") - pl.col("close").shift(10)).abs().alias("change")
        )
        
        volatility = self.df.with_columns(
            pl.col("close").diff().abs().rolling_sum(window_size=10).alias("volatility")
        )
        
        change = change.select(["timestamp", "change"])
        volatility = volatility.select(["timestamp", "volatility"])
        
        self.df = self.df.join(change, on="timestamp")
        self.df = self.df.join(volatility, on="timestamp")
        
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
                .clip(0.01, 1.0)
            )
            .otherwise(0.5)
            .alias("sc")
        )
        
        # Initialize KAMA with first close price
        self.df = self.df.with_columns(
            pl.col("close").alias("kama")
        )
        
        # Iterative KAMA calculation
        for i in range(1, len(self.df)):
            if i > 0:
                prev_kama = self.df["kama"][i-1]
                curr_close = self.df["close"][i]
                sc = self.df["sc"][i]
                
                if prev_kama is not None and curr_close is not None and sc is not None:
                    kama = prev_kama + sc * (curr_close - prev_kama)
                    self.df = self.df.with_row_index("row_idx").with_columns(
                        pl.when(pl.col("row_idx") == i)
                        .then(kama)
                        .otherwise(pl.col("kama"))
                        .alias("kama")
                    ).drop("row_idx")
                else:
                    self.df = self.df.with_row_index("row_idx").with_columns(
                        pl.when(pl.col("row_idx") == i)
                        .then(curr_close if curr_close is not None else prev_kama)
                        .otherwise(pl.col("kama"))
                        .alias("kama")
                    ).drop("row_idx")

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

        # Accumulation/Distribution Line
        self.df = self.df.with_columns([
            pl.when((pl.col("high") - pl.col("low")) != 0)
            .then(((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low")))
            .otherwise(0)
            .alias("mf_multiplier"),
            
            pl.when((pl.col("high") - pl.col("low")) != 0)
            .then(pl.col("volume") * ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low")))
            .otherwise(0)
            .alias("mf_volume")
        ])
        
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

        # Mean Reversion Indicators
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

        # Market Regime Detection
        self.df = self.df.with_columns([
            pl.col("adx").rolling_mean(window_size=20).alias("adx_ma")
        ])
        
        self.df = self.df.with_columns(
            pl.when(pl.col("adx_ma") > 25)
            .then(1)
            .otherwise(0)
            .alias("market_regime")
        )

        return self.df

    def calculate_portfolio_metrics(self):
        """Calculate portfolio value, drawdown, and buy & hold benchmark."""
        # Calculate daily returns
        self.df = self.df.with_columns([
            pl.col("close").pct_change().fill_null(0).alias("daily_return")
        ])
        
        # Calculate strategy returns
        self.df = self.df.with_columns([
            (pl.col("daily_return") * pl.col("position")).fill_null(0).alias("strategy_return")
        ])
        
        # Calculate cumulative returns
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
        
        # Calculate drawdown
        self.df = self.df.with_columns([
            ((pl.col("portfolio_value") - pl.col("peak_value")) / pl.col("peak_value")).fill_null(0.0).alias("drawdown")
        ])
        
        # Calculate max drawdown
        self.df = self.df.with_columns([
            pl.col("drawdown").cum_min().fill_null(0.0).alias("max_drawdown")
        ])
        
        return self.df

    def generate_signals(self):
        """Generate buy/sell signals based on technical indicators."""
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def print_strategy_performance(self):
        """Print strategy performance metrics."""
        buy_signals = self.df.filter(pl.col("signal") == 1)
        total_trades = len(buy_signals)
        winning_trades = len(self.df.filter(pl.col("pnl") > 0))
        total_pnl = self.df["pnl"].sum()
        
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