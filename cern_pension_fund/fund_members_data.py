import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FUND BENEFICIARIES
estimate_number_of_beneficiaries = 3600

# TAKEN FROM https://cds.cern.ch/record/2897705/files/CERN-HR-STAFF-STAT-2023.pdf
# 2023 data

arrivals = 148
departures = 140

staff_headcount = 2666
graduates_fellows_headcount = 1002
total_headcount = staff_headcount + graduates_fellows_headcount

staff_limited_duration_contract_headcount = 891
staff_illimited_duration_contract_headcount = 1775
staff_ld_to_ic_conversion = 47

staff_average_age_ld = 37.5
staff_average_age_ic = 51
graduates_fellows_average_age = 28.8

# AGE DISTRIBUTION
# Staff members age distribution
staff_under_25 = 8
staff_26_to_30 = 104
staff_31_to_35 = 324
staff_36_to_40 = 391
staff_41_to_45 = 415
staff_46_to_50 = 398
staff_51_to_55 = 429
staff_56_to_60 = 393
staff_61_to_65 = 203
staff_over_65 = 1

total_staff = staff_under_25 + staff_26_to_30 + staff_31_to_35 + staff_36_to_40 + staff_41_to_45 + staff_46_to_50 + staff_51_to_55 + staff_56_to_60 + staff_61_to_65 + staff_over_65

# NATIONALITY
staff_nationality = {
    "AT": {"name": "Austria", "headcount": 59, "life_expectancy": 82.05},
    "BE": {"name": "Belgium", "headcount": 100, "life_expectancy": 82.3},
    "BG": {"name": "Bulgaria", "headcount": 13, "life_expectancy": 75.8},
    "CH": {"name": "Switzerland", "headcount": 199, "life_expectancy": 83.45},
    "CY": {"name": "Cyprus", "headcount": 2, "life_expectancy": 82.9},
    "CZ": {"name": "Czechia", "headcount": 10, "life_expectancy": 79.9},
    "DE": {"name": "Germany", "headcount": 166, "life_expectancy": 81.1},
    "DK": {"name": "Denmark", "headcount": 16, "life_expectancy": 81.8},
    "ES": {"name": "Spain", "headcount": 186, "life_expectancy": 83.77},
    "FI": {"name": "Finland", "headcount": 27, "life_expectancy": 81.6},
    "FR": {"name": "France", "headcount": 951, "life_expectancy": 83.0},
    "GB": {"name": "United Kingdom", "headcount": 182, "life_expectancy": 82.06},
    "GR": {"name": "Greece", "headcount": 66, "life_expectancy": 81.8},
    "HR": {"name": "Croatia", "headcount": 2, "life_expectancy": 78.6},
    "HU": {"name": "Hungary", "headcount": 15, "life_expectancy": 76.7},
    "IL": {"name": "Israel", "headcount": 1, "life_expectancy": 82.7},
    "IN": {"name": "India", "headcount": 11, "life_expectancy": 67.74},
    "IT": {"name": "Italy", "headcount": 328, "life_expectancy": 83.5},
    "LT": {"name": "Lithuania", "headcount": 3, "life_expectancy": 77.6},
    "LV": {"name": "Latvia", "headcount": 4, "life_expectancy": 75.6},
    "NL": {"name": "Netherlands", "headcount": 60, "life_expectancy": 81.9},
    "NO": {"name": "Norway", "headcount": 14, "life_expectancy": 83.1},
    "PK": {"name": "Pakistan", "headcount": 5, "life_expectancy": 66.43},
    "PL": {"name": "Poland", "headcount": 94, "life_expectancy": 78.4},
    "PT": {"name": "Portugal", "headcount": 61, "life_expectancy": 82.5},
    "RO": {"name": "Romania", "headcount": 24, "life_expectancy": 76.4},
    "RS": {"name": "Serbia", "headcount": 5, "life_expectancy": 76.2},
    "SE": {"name": "Sweden", "headcount": 26, "life_expectancy": 82.4},
    "SI": {"name": "Slovenia", "headcount": 1, "life_expectancy": 82.0},
    "SK": {"name": "Slovakia", "headcount": 14, "life_expectancy": 78.2},
    "TR": {"name": "Turkey", "headcount": 7, "life_expectancy": 77.3},
    "UA": {"name": "Ukraine", "headcount": 2, "life_expectancy": 68.59},
    "NMS": {"name": "Non-Member States", "headcount": 12, "life_expectancy": 73.16}
}

# SALARY DISTRIBUTION AND C-FACTOR
# c_factor is a multiplication factor used to calculate contributions based on salary
# It is not an indication of salary distribution, but rather a factor to adjust the base salary
# for contribution calculations
staff_salary = {
    "grade_1": {"salary": 4082, "c_factor": 1.3400},
    "grade_2": {"salary": 4857, "c_factor": 1.3202},
    "grade_3": {"salary": 5780, "c_factor": 1.2995},
    "grade_4": {"salary": 6879, "c_factor": 1.2785},
    "grade_5": {"salary": 8186, "c_factor": 1.2580},
    "grade_6": {"salary": 9004, "c_factor": 1.2473},
    "grade_7": {"salary": 10715, "c_factor": 1.2296}
}

staff_category_grade_estimate = {
    "Research Physicists": {
        "staff_count": 85,
        "age_distribution": {
            "≤25": 0, "26–30": 0, "31–35": 18, "36–40": 23, "41–45": 8,
            "46–50": 7, "51–55": 8, "56–60": 8, "61–65": 13, ">65": 0
        },
        "estimated_grade_range": [6, 7]
    },
    "Scientific & Engineering work": {
        "staff_count": 1263,
        "age_distribution": {
            "≤25": 1, "26–30": 38, "31–35": 162, "36–40": 191, "41–45": 189,
            "46–50": 179, "51–55": 188, "56–60": 201, "61–65": 113, ">65": 1
        },
        "estimated_grade_range": [4, 6]
    },
    "Technical work": {
        "staff_count": 809,
        "age_distribution": {
            "≤25": 7, "26–30": 47, "31–35": 94, "36–40": 100, "41–45": 139,
            "46–50": 129, "51–55": 136, "56–60": 113, "61–65": 44, ">65": 0
        },
        "estimated_grade_range": [3, 5]
    },
    "Manual work": {
        "staff_count": 50,
        "age_distribution": {
            "≤25": 0, "26–30": 2, "31–35": 4, "36–40": 16, "41–45": 10,
            "46–50": 8, "51–55": 5, "56–60": 5, "61–65": 0, ">65": 0
        },
        "estimated_grade_range": [1, 3]
    },
    "Professional Admin work": {
        "staff_count": 184,
        "age_distribution": {
            "≤25": 0, "26–30": 0, "31–35": 9, "36–40": 27, "41–45": 25,
            "46–50": 30, "51–55": 42, "56–60": 35, "61–65": 16, ">65": 0
        },
        "estimated_grade_range": [4, 6]
    },
    "Office and Admin work": {
        "staff_count": 275,
        "age_distribution": {
            "≤25": 0, "26–30": 17, "31–35": 37, "36–40": 34, "41–45": 44,
            "46–50": 45, "51–55": 50, "56–60": 31, "61–65": 17, ">65": 0
        },
        "estimated_grade_range": [1, 3]
    }
}

graduate_fellow_salary = {
    "grade_1": {"name": "Early Career", "salary": 4569, "category": "graduate"},
    "grade_2": {"name": "BSc Early Career", "salary": 5134, "category": "graduate"},
    "grade_3": {"name": "MSc Early Career", "salary": 5647, "category": "graduate"},
    "grade_4": {"name": "MSc 2-4 Years Experienced", "salary": 6212, "category": "fellow"},
    "grade_5": {"name": "MSc 4-6 Years Experienced", "salary": 6828, "category": "fellow"},
    "grade_6": {"name": "PhD 0-3 Years Experienced", "salary": 6828, "category": "fellow"},
    "grade_7": {"name": "PhD 3-6 Years Experienced", "salary": 7239, "category": "fellow"}
}

graduates_category_grade_estimate = {
    "Research Physicists": {
        "graduates_count": 106,
        "average_age": 32.5,
        "estimated_grade_range": [6, 7]
    },
    "Scientific & Engineering work": {
        "graduates_count": 668,
        "average_age": 28.9,
        "estimated_grade_range": [3, 5]
    },
    "Technical work": {
        "graduates_count": 140,
        "average_age": 25.6,
        "estimated_grade_range": [1, 3]
    },
    "Manual work": {
        "graduates_count": 0
    },
    "Professional Admin work": {
        "graduates_count": 66,
        "average_age": 29.4,
        "estimated_grade_range": [3, 5]
    },
    "Office and Admin work": {
        "graduates_count": 22,
        "average_age": 25.7,
        "estimated_grade_range": [1, 4]
    }
}

# From 2023 financial statements
members_pre_01_01_2012 = 1483
members_post_01_01_2012 = 2186
total_members = members_pre_01_01_2012 + members_post_01_01_2012

deferred_pensions = 316
retirement_pensions = 2235
surviving_spouses_pensions = 829
orphans_pensions = 38
disability_pensions = 21

total_beneficiaries = deferred_pensions + retirement_pensions + surviving_spouses_pensions + orphans_pensions + disability_pensions

beneficiaries_over_100_years_old = 7
beneficiaries_under_21_years_old = 18

average_quarterly_volumes_services = {
    "2023": {"payments": 9880, "contributions": 12380, "transfer_values": 75, "purchase_period_additions": 14, "handling_deaths": 41, "update_data": 170, "attestations": 40, "feedback": 2},
}

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

def calculate_transfer_value(years_of_service: int, salary: float) -> float:
    """
    Calculate transfer value at the end of the contract.
    This is the amount of money that the employee will receive at the end of the contract.
    This is not related to the retirement pension.

    The amount of your transfer value is calculated as follows:
    - 14.7% of your reference salary for each of the first 10 years of pensionable service
    - 22% of your reference salary for each further year
    """
    if years_of_service < 10:
        return 0.147 * salary * years_of_service
    else:
        return 0.147 * salary * 10 + 0.22 * salary * (years_of_service - 10)

def calculate_retirement_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    """
    Calculate retirement pension based on age and salary.

    Requirements:
    - 5 years of service
    - Retirement age: 67
    - the retirement pension amount is equal to 1.85% of the average of the last 36 months' reference salaries,
      at the time of the end of your contract, per year of membership (maximum: 70% for 37 years and 10 months)
    """
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")
    
    pension_rate = 0.0185 if not joined_before_2011 else 0.02
    pension_percentage = min(years_of_service * pension_rate, 0.7)
    return salary * pension_percentage

def calculate_deferred_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    """
    Calculate deferred pension based on age and salary.
    Requirements: 5 years of service
    """
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")
    
    pension_rate = 0.0185 if not joined_before_2011 else 0.02
    return salary * pension_rate * years_of_service

def calculate_surviving_spouse_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    """
    Calculate surviving spouse pension based on age and salary.
    """
    pension_rate = 0.011
    return salary * pension_rate * years_of_service

def calculate_orphan_pension(years_of_service: int, salary: float, number_of_children: int) -> float:
    """
    Calculate orphan pension based on age and salary.

    Requirements:
    - payable to unemployed children until 20 years old
    - payable to full-time students until 25 years old
    """
    if number_of_children == 0:
        return 0
    elif number_of_children == 1:
        return 0.24 * salary
    elif number_of_children == 2:
        return 0.34 * salary
    else:
        return 0.40 * salary

def calculate_disability_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    """
    Calculate disability pension based on age and salary.
    Requirements: 5 years of service
    """
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")
    
    pension_rate = 0.0185 if not joined_before_2011 else 0.02
    return salary * pension_rate * years_of_service

def estimate_staff_contributions_df():
    """Estimate staff contributions and return as DataFrame"""
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
    """Estimate graduate and fellow contributions and return as DataFrame"""
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

def calculate_average_metrics():
    """Calculate average years of service and salary based on staff data"""
    # Calculate average age for each category
    total_staff_age = 0
    total_staff_count = 0
    
    for category, info in staff_category_grade_estimate.items():
        age_dist = info["age_distribution"]
        count = info["staff_count"]
        
        # Calculate weighted average age for this category
        category_age = (
            age_dist["≤25"] * 23 +  # Using midpoint of age ranges
            age_dist["26–30"] * 28 +
            age_dist["31–35"] * 33 +
            age_dist["36–40"] * 38 +
            age_dist["41–45"] * 43 +
            age_dist["46–50"] * 48 +
            age_dist["51–55"] * 53 +
            age_dist["56–60"] * 58 +
            age_dist["61–65"] * 63 +
            age_dist[">65"] * 67
        ) / count
        
        total_staff_age += category_age * count
        total_staff_count += count
    
    average_age = total_staff_age / total_staff_count
    
    # Calculate average salary
    total_salary = 0
    total_count = 0
    
    for category, info in staff_category_grade_estimate.items():
        count = info["staff_count"]
        grade_range = info["estimated_grade_range"]
        avg_grade = round(sum(grade_range) / len(grade_range))
        grade_key = f"grade_{avg_grade}"
        
        salary_info = staff_salary[grade_key]
        salary = salary_info["salary"]
        c_factor = salary_info["c_factor"]
        
        total_salary += salary * c_factor * count
        total_count += count
    
    average_salary = total_salary / total_count
    
    # Estimate years of service based on average age
    # Assuming average joining age of 30
    average_years_of_service = average_age - 30
    
    return {
        "average_age": average_age,
        "average_salary": average_salary,
        "average_years_of_service": average_years_of_service
    }

def save_contributions_to_csv(df, filename):
    """Save contributions DataFrame to CSV file"""
    df.to_csv(filename, index=False)

def calculate_total_contributions(df):
    """Calculate total contributions from DataFrame"""
    totals = df[["Employee Contribution", "Organization Contribution"]].sum()
    totals["Total Contribution"] = totals.sum()
    return totals.round(2)

if __name__ == "__main__":
    staff_df = estimate_staff_contributions_df()
    grad_fellow_df = estimate_graduate_fellow_contributions_df()

    # Save to CSV
    save_contributions_to_csv(staff_df, "cern_pension_fund/csv/staff_contributions.csv")
    save_contributions_to_csv(grad_fellow_df, "cern_pension_fund/csv/grad_fellow_contributions.csv")

    # Print summaries
    print("=== Staff Total Contributions ===")
    total_staff_contributions = calculate_total_contributions(staff_df)
    print(total_staff_contributions)

    print("\n=== Graduate/Fellow Total Contributions ===")
    total_grad_fellow_contributions = calculate_total_contributions(grad_fellow_df)
    print(total_grad_fellow_contributions)

    print("\n=== Total Contributions ===")
    print(f"{total_staff_contributions + total_grad_fellow_contributions}")

    print("\n✅ CSV files saved!") 