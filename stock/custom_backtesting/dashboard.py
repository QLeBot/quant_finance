import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from datetime import datetime, timedelta
from data.data_handler import DataHandler
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.strat_1 import Strategy_1
from strategies.strat_2 import Strategy_2

def create_price_chart(df, strategy_name):
    """Create an interactive price chart with trading signals."""
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        ),
        row=1, col=1
    )

    # Add buy signals
    buy_signals = df.filter(pl.col("signal") == 1)
    fig.add_trace(
        go.Scatter(
            x=buy_signals["timestamp"],
            y=buy_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=15, color="green"),
            name="Buy Signal"
        ),
        row=1, col=1
    )

    # Add sell signals
    sell_signals = df.filter(pl.col("signal") == -1)
    fig.add_trace(
        go.Scatter(
            x=sell_signals["timestamp"],
            y=sell_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=15, color="red"),
            name="Sell Signal"
        ),
        row=1, col=1
    )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Volume",
            marker_color="rgba(0,0,255,0.3)"
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{strategy_name} - Price and Signals",
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

def create_performance_chart(df, strategy_name):
    """Create an interactive performance chart showing portfolio value and drawdown."""
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add portfolio value
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["portfolio_value"],
            name="Portfolio Value",
            line=dict(color="blue")
        ),
        row=1, col=1
    )

    # Add buy & hold value
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bh_portfolio_value"],
            name="Buy & Hold",
            line=dict(color="gray", dash="dash")
        ),
        row=1, col=1
    )

    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["drawdown"] * 100,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red")
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{strategy_name} - Performance",
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Drawdown (%)",
        height=800
    )

    return fig

def display_strategy_metrics(df):
    """Display key performance metrics for a strategy."""
    final_portfolio_value = df["portfolio_value"].last()
    max_drawdown = df["max_drawdown"].min()
    bh_final_portfolio_value = df["bh_portfolio_value"].last()
    total_return = (final_portfolio_value - 10000) / 10000 * 100
    
    buy_signals = df.filter(pl.col("signal") == 1)
    total_trades = len(buy_signals)
    winning_trades = len(df.filter(pl.col("pnl") > 0))
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Portfolio Value", f"${final_portfolio_value:,.2f}")
        st.metric("Buy & Hold Value", f"${bh_final_portfolio_value:,.2f}")
    
    with col2:
        st.metric("Total Return", f"{total_return:.2f}%")
        st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
    
    with col3:
        st.metric("Total Trades", total_trades)
        st.metric("Win Rate", f"{win_rate:.2f}%")
    
    with col4:
        st.metric("Winning Trades", winning_trades)
        st.metric("Losing Trades", total_trades - winning_trades)

def main():
    st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")
    st.title("Trading Strategy Dashboard")

    # Parameters
    ticker = st.sidebar.text_input("Ticker Symbol", "SPY")
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime(2007, 1, 1)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        datetime.now() - timedelta(days=5)
    )
    initial_capital = st.sidebar.number_input(
        "Initial Capital",
        value=10000,
        step=1000
    )

    # Fetch and process data
    data_handler = DataHandler(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df = data_handler.alpaca_data()

    # Initialize strategies
    strategies = {
        "Strategy 1": Strategy_1(df.clone(), ticker, initial_capital),
        "Strategy 2": Strategy_2(df.clone(), ticker, initial_capital),
        "Trend Following": TrendFollowingStrategy(df.clone(), ticker, initial_capital),
        "Mean Reversion": MeanReversionStrategy(df.clone(), ticker, initial_capital)
    }

    # Create tabs for each strategy
    tabs = st.tabs(list(strategies.keys()))

    # Process and display each strategy
    for tab, (strategy_name, strategy) in zip(tabs, strategies.items()):
        with tab:
            st.header(strategy_name)
            
            # Calculate indicators and generate signals
            strategy.calculate_indicators()
            strategy.generate_signals()
            
            # Display performance metrics
            st.subheader("Performance Metrics")
            display_strategy_metrics(strategy.df)
            
            # Create and display charts
            st.subheader("Price and Signals")
            price_chart = create_price_chart(strategy.df, strategy_name)
            st.plotly_chart(price_chart, use_container_width=True)
            
            st.subheader("Performance Analysis")
            performance_chart = create_performance_chart(strategy.df, strategy_name)
            st.plotly_chart(performance_chart, use_container_width=True)

if __name__ == "__main__":
    main() 