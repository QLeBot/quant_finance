from datetime import datetime, timedelta
from data.data_handler import DataHandler
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.strat_1 import Strategy_1
from strategies.strat_2 import Strategy_2
from plotting.plotter import Plotter

def main():
    """Main function to run multiple strategies."""
    # Parameters
    ticker = "SPY"
    start_date = "2007-01-01"
    end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    initial_capital = 10000
    
    # Fetch and process data
    data_handler = DataHandler(ticker, start_date, end_date)
    df = data_handler.alpaca_data()
    
    # Initialize strategies
    strategies = {
        #"Trend Following": TrendFollowingStrategy(df.clone(), ticker, initial_capital),
        #"Mean Reversion": MeanReversionStrategy(df.clone(), ticker, initial_capital),
        "Strategy 1": Strategy_1(df.clone(), ticker, initial_capital),
        "Strategy 2": Strategy_2(df.clone(), ticker, initial_capital)
    }
    
    # Calculate indicators and generate signals for each strategy
    strategies_data = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name} strategy...")
        strategy.calculate_indicators()
        strategy.generate_signals()
        strategies_data[name] = strategy.df
        strategy.print_strategy_performance()
    
    # Create plotter and plot results
    plotter = Plotter(strategies_data, ticker)
    plotter.plot_all()

if __name__ == "__main__":
    main() 