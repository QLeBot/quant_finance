import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mocking the staff_data module with assumed values
staff_salary = {
    "grade_1": {"salary": 5000, "c_factor": 1.2},
    "grade_2": {"salary": 6000, "c_factor": 1.3},
    "grade_3": {"salary": 7000, "c_factor": 1.4},
}

graduate_fellow_salary = {
    "grade_1": {"category": "graduate", "salary": 5748},
    "grade_2": {"category": "fellow", "salary": 6367},
}

staff_category_grade_estimate = {
    "category_1": {"staff_count": 100, "estimated_grade_range": [1, 2]},
    "category_2": {"staff_count": 200, "estimated_grade_range": [2, 3]},
}

graduates_category_grade_estimate = {
    "category_1": {"graduates_count": 50, "estimated_grade_range": [1]},
    "category_2": {"graduates_count": 30, "estimated_grade_range": [2]},
}

def calculate_pension_contribution(category: str, basic_salary: float = None, c_factor: float = None) -> dict:
    reference_salaries = {
        "graduate": 5748,
        "fellow": 6367
    }

    if category.lower() == "staff":
        if basic_salary is None or c_factor is None:
            raise ValueError("For staff, both basic_salary and c_factor must be provided.")
        reference_salary = basic_salary * c_factor
    elif category.lower() in reference_salaries:
        reference_salary = reference_salaries[category.lower()]
    else:
        raise ValueError("Category must be 'staff', 'graduate', or 'fellow'.")

    employee_contrib = reference_salary * 0.1264
    organization_contrib = reference_salary * 0.1896
    total_contrib = employee_contrib + organization_contrib

    return {
        "reference_salary": round(reference_salary, 2),
        "employee_contribution": round(employee_contrib, 2),
        "organization_contribution": round(organization_contrib, 2),
        "total_contribution": round(total_contrib, 2)
    }

def estimate_staff_contributions_df():
    records = []

    for category, info in staff_category_grade_estimate.items():
        count = info["staff_count"]
        grade_range = info["estimated_grade_range"]
        avg_grade = round(sum(grade_range) / len(grade_range))
        grade_key = f"grade_{avg_grade}"

        salary_info = staff_salary[grade_key]
        salary = salary_info["salary"]
        c_factor = salary_info["c_factor"]

        contrib = calculate_pension_contribution("staff", salary, c_factor)

        records.append({
            "Category": category,
            "Type": "Staff",
            "Headcount": count,
            "Employee Contribution": contrib["employee_contribution"] * count,
            "Organization Contribution": contrib["organization_contribution"] * count
        })

    return pd.DataFrame(records)

def estimate_graduate_fellow_contributions_df():
    records = []

    for category, info in graduates_category_grade_estimate.items():
        count = info.get("graduates_count", 0)
        if count == 0:
            continue

        grade_range = info["estimated_grade_range"]
        avg_grade = round(sum(grade_range) / len(grade_range))
        grade_key = f"grade_{avg_grade}"

        grade_info = graduate_fellow_salary[grade_key]
        person_type = grade_info["category"]

        contrib = calculate_pension_contribution(person_type)

        records.append({
            "Category": category,
            "Type": person_type.capitalize(),
            "Headcount": count,
            "Employee Contribution": contrib["employee_contribution"] * count,
            "Organization Contribution": contrib["organization_contribution"] * count
        })

    return pd.DataFrame(records)

def calculate_total_contributions(df):
    totals = df[["Employee Contribution", "Organization Contribution"]].sum()
    totals["Total Contribution"] = totals.sum()
    return totals.round(2)

def calculate_transfer_value(years_of_service: int, salary: float) -> float:
    if years_of_service < 10:
        return 0.147 * salary * years_of_service
    else:
        return 0.147 * salary * 10 + 0.22 * salary * (years_of_service - 10)

def calculate_retirement_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")

    pension_rate = 0.0185 if not joined_before_2011 else 0.02
    pension_percentage = min(years_of_service * pension_rate, 0.7)
    pension = salary * pension_percentage

    return pension

# Calculate total contributions
staff_df = estimate_staff_contributions_df()
grad_fellow_df = estimate_graduate_fellow_contributions_df()

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
beneficiaries = 3600

# Assumptions
annual_return_rate = 0.05  # Expected annual return rate
monthly_return_rate = annual_return_rate / 12
volatility = 0.1  # Annual volatility of returns
monthly_volatility = volatility / np.sqrt(12)

# Simulation
np.random.seed(42)
simulations = 1000
fund_values = np.zeros((months, simulations))

# Assume average salary and years of service for exit calculations
average_salary = 6000
average_years_of_service = 20

for i in range(simulations):
    fund_value = initial_value
    for t in range(months):
        # Monthly return simulation
        monthly_return = np.random.normal(monthly_return_rate, monthly_volatility)
        fund_value = fund_value * (1 + monthly_return) + monthly_contribution

        # Simulate exits
        if t % 12 == 0:  # Assume exits are calculated annually
            exits = np.random.binomial(beneficiaries, 0.05)  # Assume 5% exit rate per year
            transfer_value_total = calculate_transfer_value(average_years_of_service, average_salary) * exits
            retirement_pension_total = calculate_retirement_pension(average_years_of_service, average_salary) * exits
            fund_value -= (transfer_value_total + retirement_pension_total)

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
plt.title('Pension Fund Growth Simulation Over 10 Years with Exits')
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
