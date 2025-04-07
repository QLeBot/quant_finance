import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fund_entry import calculate_total_contributions
from fund_exit import calculate_transfer_value, calculate_retirement_pension
from staff_data import staff_category_grade_estimate, graduates_category_grade_estimate, staff_salary, graduate_fellow_salary

# Load contributions
staff_df = pd.read_csv("cern_pension_fund/staff_contributions.csv")
grad_fellow_df = pd.read_csv("cern_pension_fund/grad_fellow_contributions.csv")

# Summarize contributions
staff_totals = calculate_total_contributions(staff_df)
grad_fellow_totals = calculate_total_contributions(grad_fellow_df)

# Total monthly contributions
total_monthly_contributions = staff_totals["Total Contribution"] + grad_fellow_totals["Total Contribution"]

# Parameters for simulation
initial_value = 4.46e9  # Initial managed assets value in CHF
monthly_contribution = total_monthly_contributions  # Total monthly contributions in CHF
years = 10
months = years * 12

# Assumptions
annual_return_rate = 0.05  # Expected annual return rate
monthly_return_rate = annual_return_rate / 12
volatility = 0.1  # Annual volatility of returns
monthly_volatility = volatility / np.sqrt(12)

# Simulation
np.random.seed(42)
simulations = 1000
fund_values = np.zeros((months, simulations))

for i in range(simulations):
    fund_value = initial_value
    for t in range(months):
        # Monthly return simulation
        monthly_return = np.random.normal(monthly_return_rate, monthly_volatility)
        fund_value = fund_value * (1 + monthly_return) + monthly_contribution
        fund_values[t, i] = fund_value

# Calculate the mean fund value over time
mean_fund_value = np.mean(fund_values, axis=1)

# Plot the simulation results
plt.figure(figsize=(12, 6))
plt.plot(mean_fund_value, label='Mean Fund Value', color='blue')
plt.fill_between(range(months),
                 np.percentile(fund_values, 5, axis=1),
                 np.percentile(fund_values, 95, axis=1),
                 color='lightblue', alpha=0.5, label='90% Confidence Interval')
plt.title('Pension Fund Growth Simulation Over 10 Years')
plt.xlabel('Months')
plt.ylabel('Fund Value (CHF)')
plt.legend()
plt.grid(True)
plt.show()

# Final fund value statistics
final_fund_values = fund_values[-1, :]
final_mean = np.mean(final_fund_values)
final_std = np.std(final_fund_values)
cvar_5 = np.percentile(final_fund_values, 5)

print(f"Final Mean Fund Value: CHF {final_mean:,.2f}")
print(f"Standard Deviation: CHF {final_std:,.2f}")
print(f"5% CVaR: CHF {cvar_5:,.2f}")