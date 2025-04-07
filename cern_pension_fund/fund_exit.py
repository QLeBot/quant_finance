import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import staff data
from staff_data import staff_salary, graduate_fellow_salary
from staff_data import staff_category_grade_estimate, graduates_category_grade_estimate

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

