import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import staff data
from staff_data import staff_salary, graduate_fellow_salary
from staff_data import staff_category_grade_estimate, graduates_category_grade_estimate



def calculate_pension_contribution(category: str, basic_salary: float = None, c_factor: float = None) -> dict:
    """
    Calculate monthly pension fund contributions for a given category.

    Based on information from https://pensionfund.cern.ch/en/members-0/newcomers-0

    Parameters:
    - category (str): 'staff', 'graduate', or 'fellow'
    - basic_salary (float): Base salary (only for staff)
    - c_factor (float): C factor to apply (only for staff)

    Returns:
    - dict with employee, organization, and total contribution
    """
    # Reference salaries for non-staff
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


def save_contributions_to_csv(df, filename):
    df.to_csv(filename, index=False)


def calculate_total_contributions(df):
    totals = df[["Employee Contribution", "Organization Contribution"]].sum()
    totals["Total Contribution"] = totals.sum()
    return totals.round(2)


if __name__ == "__main__":
    staff_df = estimate_staff_contributions_df()
    grad_fellow_df = estimate_graduate_fellow_contributions_df()

    # Save to CSV
    save_contributions_to_csv(staff_df, "cern_pension_fund/staff_contributions.csv")
    save_contributions_to_csv(grad_fellow_df, "cern_pension_fund/grad_fellow_contributions.csv")

    # Print summaries
    print("=== Staff Total Contributions ===")
    print(calculate_total_contributions(staff_df))

    print("\n=== Graduate/Fellow Total Contributions ===")
    print(calculate_total_contributions(grad_fellow_df))

    print("\n✅ CSV files saved!")
