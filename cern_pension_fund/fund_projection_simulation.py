import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fund_portfolio_data import saa, returns_2023

class FundProjectionSimulator:
    def __init__(self, initial_value=4.46e9):  # 4.46 billion CHF initial value
        self.initial_value = initial_value
        self.saa = saa
        self.returns_2023 = returns_2023
        self.asset_classes = list(saa.keys())
        
        # Define return ranges for each asset class based on 2023 performance
        # These ranges are used to generate random returns
        self.return_ranges = {
            "fixed_income": (-0.02, 0.05),  # Conservative range
            "equity": (-0.15, 0.20),        # Wide range for equities
            "real_estate": (-0.10, 0.15),   # Improved range for real estate
            "infrastructure": (-0.05, 0.10),
            "timber_farmland": (-0.05, 0.10),
            "private_equity": (-0.10, 0.20),
            "hedge_funds": (-0.05, 0.10),
            "metals_commodities": (-0.10, 0.15),
            "cash": (0.01, 0.05)            # Conservative range for cash
        }
        
        # Define allocation variation range (±2%)
        self.allocation_variation = 0.02

    def generate_random_allocation(self):
        """Generate a random allocation with small variations from the base SAA"""
        variations = np.random.uniform(-self.allocation_variation, self.allocation_variation, len(self.asset_classes))
        new_allocation = {}
        total = 0
        
        # Apply variations and ensure they sum to 1
        for i, asset in enumerate(self.asset_classes):
            new_allocation[asset] = max(0, self.saa[asset] + variations[i])
            total += new_allocation[asset]
        
        # Normalize to ensure sum is 1
        for asset in new_allocation:
            new_allocation[asset] /= total
            
        return new_allocation

    def generate_random_returns(self):
        """Generate random returns for each asset class within defined ranges"""
        returns = {}
        for asset, (min_return, max_return) in self.return_ranges.items():
            returns[asset] = np.random.uniform(min_return, max_return)
        return returns

    def simulate_year(self, current_value, allocation=None, returns=None):
        """Simulate one year of fund performance"""
        if allocation is None:
            allocation = self.generate_random_allocation()
        if returns is None:
            returns = self.generate_random_returns()
            
        # Calculate weighted return
        weighted_return = sum(allocation[asset] * returns[asset] for asset in self.asset_classes)
        
        # Calculate new value
        new_value = current_value * (1 + weighted_return)
        
        return new_value, allocation, returns

    def run_simulation(self, years=10, num_simulations=1000):
        """Run Monte Carlo simulation for multiple years"""
        results = np.zeros((years + 1, num_simulations))
        results[0] = self.initial_value
        
        for sim in range(num_simulations):
            current_value = self.initial_value
            for year in range(1, years + 1):
                current_value, _, _ = self.simulate_year(current_value)
                results[year, sim] = current_value
                
        return results

    def analyze_results(self, results):
        """Analyze simulation results"""
        mean_values = np.mean(results, axis=1)
        median_values = np.median(results, axis=1)
        percentile_5 = np.percentile(results, 5, axis=1)
        percentile_95 = np.percentile(results, 95, axis=1)
        
        return {
            'mean': mean_values,
            'median': median_values,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95
        }

    def plot_results(self, results, analysis):
        """Plot simulation results"""
        years = np.arange(len(results))
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, analysis['mean'], label='Mean', color='blue')
        plt.plot(years, analysis['median'], label='Median', color='green')
        plt.fill_between(years, analysis['percentile_5'], analysis['percentile_95'], 
                        alpha=0.2, color='gray', label='90% Confidence Interval')
        
        plt.title('Fund Value Projection (Monte Carlo Simulation)')
        plt.xlabel('Years')
        plt.ylabel('Fund Value (CHF)')
        plt.legend()
        plt.grid(True)
        
        # Format y-axis to show billions
        plt.gca().yaxis.set_major_formatter(lambda x, p: f'{x/1e9:.2f}B')
        
        return plt

def main():
    # Initialize simulator
    simulator = FundProjectionSimulator()
    
    # Run simulation
    years = 10
    num_simulations = 1000
    results = simulator.run_simulation(years, num_simulations)
    
    # Analyze results
    analysis = simulator.analyze_results(results)
    
    # Plot results
    plt = simulator.plot_results(results, analysis)
    plt.show()
    
    # Print key statistics
    print("\nSimulation Results:")
    print(f"Initial Value: CHF {simulator.initial_value/1e9:.2f}B")
    print(f"Final Mean Value: CHF {analysis['mean'][-1]/1e9:.2f}B")
    print(f"Final Median Value: CHF {analysis['median'][-1]/1e9:.2f}B")
    print(f"Final 5th Percentile: CHF {analysis['percentile_5'][-1]/1e9:.2f}B")
    print(f"Final 95th Percentile: CHF {analysis['percentile_95'][-1]/1e9:.2f}B")
    
    # Calculate annualized return
    initial_value = simulator.initial_value
    final_mean = analysis['mean'][-1]
    annualized_return = (final_mean / initial_value) ** (1/years) - 1
    print(f"\nAnnualized Return: {annualized_return*100:.2f}%")

if __name__ == "__main__":
    main() 