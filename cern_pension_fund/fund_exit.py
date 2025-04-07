import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import staff data
from staff_data import staff_salary, graduate_fellow_salary
from staff_data import staff_category_grade_estimate, graduates_category_grade_estimate

# From 2023 financial statements
# https://pensionfund.cern.ch/sites/default/files/2024-09/Annual%20Report%20and%20Financial%20Statements%202023.pdf

members_pre_01_01_2012 = 1483
members_post_01_01_2012 = 2186

total_members = members_pre_01_01_2012 + members_post_01_01_2012

deferred_pensions = 316
retirement_pensions = 2235
surviving_spouses_pensions = 829
orphans_pensions = 38
disability_pensions = 21

total_beneficiaries = deferred_pensions + retirement_pensions + surviving_spouses_pensions + orphans_pensions + disability_pensions

print(f"Total beneficiaries: {total_beneficiaries}")

beneficiaries_over_100_years_old = 7
beneficiaries_under_21_years_old = 18

def calculate_transfer_value(years_of_service: int, salary: float) -> float:
    """
    Calculate transfer value at the end of the contract. 
    This is the amount of money that the employee will receive at the end of the contract.
    This is not related to the retirement pension.

    The amount of your transfer value is calculated as follows:
    - 14.7% of your reference salary for each of the first 10 years of pensionable service
    - 22% of your reference salary for each further year

    Parameters:
    - years_of_service (int): Years of service of the employee
    - salary (float): Salary of the employee

    Returns:
    - float: Transfer value
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
    - the retirement pension amount is equal to 1.85% of the average of the last 36 months' reference salaries, at the time of the end of your contract, per year of membership 
    (maximum: 70% for 37 years and 10 months)

    Parameters:
    - last_36_month_salary (float): The average salary of the last 36 months of service.
    - years_of_service (float): The total number of years of service (must be at least 5).
    - joined_before_2011 (bool): Whether the member joined the fund before 31 December 2011.

    Returns:
    - float: Retirement pension
    """
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")
    
    # Pension rate based on join date
    pension_rate = 0.0185 if not joined_before_2011 else 0.02
    
    # Calculate pension based on the number of years of service
    pension_percentage = min(years_of_service * pension_rate, 0.7)  # Max pension is 70%

    # Calculate the retirement pension
    pension = salary * pension_percentage
    
    return pension

def calculate_deferred_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    """
    Calculate deferred pension based on age and salary.

    Requirements:
    - 5 years of service
    """
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")
    
    # Pension rate based on join date
    pension_rate = 0.0185 if not joined_before_2011 else 0.02
    
    # Calculate deferred pension
    pension = salary * pension_rate * years_of_service
    
    return pension


def calculate_surviving_spouse_pension(years_of_service: int, salary: float, joined_before_2011: bool = False) -> float:
    """
    Calculate surviving spouse pension based on age and salary.

    Requirements:


    Parameters:
    - years_of_service (int): Years of service of the employee
    - salary (float): Salary of the employee
    - joined_before_2011 (bool): Whether the member joined the fund before 31 December 2011.

    Returns:
    - float: Surviving spouse pension

    """
    # Pension rate based on join date
    pension_rate = 0.011
    
    # Calculate surviving spouse pension
    pension = salary * pension_rate * years_of_service
    
    return pension

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

    Requirements:
    - equals the retirement pension the member would have received at the applicable retirement age

    Parameters:
    - years_of_service (int): Years of service of the employee
    - salary (float): Salary of the employee
    - joined_before_2011 (bool): Whether the member joined the fund before 31 December 2011.

    Returns:
    - float: Disability pension
    """
    if years_of_service < 5:
        raise ValueError("Minimum service required is 5 years.")
    
    # Pension rate based on join date
    pension_rate = 0.0185 if not joined_before_2011 else 0.02

    # Calculate disability pension
    pension = salary * pension_rate * years_of_service
    
    return pension

def calculate_total_pensions(beneficiaries_over_100_years_old: int, beneficiaries_under_21_years_old: int) -> float:
    """
    Calculate total beneficiaries pensions.
    """
    retirement_pensions = calculate_retirement_pension(years_of_service=37, salary=100000)
    deferred_pensions = calculate_deferred_pension(years_of_service=37, salary=100000)
    surviving_spouse_pensions = calculate_surviving_spouse_pension(years_of_service=37, salary=100000)
    orphan_pensions = calculate_orphan_pension(years_of_service=37, salary=100000, number_of_children=0)
    disability_pensions = calculate_disability_pension(years_of_service=37, salary=100000)



