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

print(f"Total staff: {total_staff}")

# Calculate the percentage of staff in each age group

# NATIONALITY

# Staff nationality distribution
staff_nationality = {
    "AT" : {
        "name": "Austria",
        "headcount": 59,
        "life_expectancy": 82.05,
    },
    "BE" : {
        "name": "Belgium",
        "headcount": 100,
        "life_expectancy": 82.3,
    },
    "BG" : {
        "name": "Bulgaria",
        "headcount": 13,
        "life_expectancy": 75.8,
    },
    "CH" : {
        "name": "Switzerland",
        "headcount": 199,
        "life_expectancy": 83.45,
    },
    "CY" : {
        "name": "Cyprus",
        "headcount": 2,
        "life_expectancy": 82.9,
    },
    "CZ" : {
        "name": "Czechia",
        "headcount": 10,
        "life_expectancy": 79.9,
    },
    "DE" : {
        "name": "Germany",
        "headcount": 166,
        "life_expectancy": 81.1,
    },
    "DK" : {
        "name": "Denmark",
        "headcount": 16,
        "life_expectancy": 81.8,
    },
    "ES" : {
        "name": "Spain",
        "headcount": 186,
        "life_expectancy": 83.77,
    },
    "FI" : {
        "name": "Finland",
        "headcount": 27,
        "life_expectancy": 81.6,
    },
    "FR" : {
        "name": "France",
        "headcount": 951,
        "life_expectancy": 83.0,
    },
    "GB" : {
        "name": "United Kingdom",
        "headcount": 182,
        "life_expectancy": 82.06,
    },
    "GR" : {
        "name": "Greece",
        "headcount": 66,
        "life_expectancy": 81.8,
    },
    "HR" : {
        "name": "Croatia",
        "headcount": 2,
        "life_expectancy": 78.6,
    },
    "HU" : {
        "name": "Hungary",
        "headcount": 15,
        "life_expectancy": 76.7,
    },
    "IL" : {
        "name": "Israel",
        "headcount": 1,
        "life_expectancy": 82.7,
    },
    "IN" : {
        "name": "India",
        "headcount": 11,
        "life_expectancy": 67.74,
    },
    "IT" : {
        "name": "Italy",
        "headcount": 328,
        "life_expectancy": 83.5,
    },
    "LT" : {
        "name": "Lithuania",
        "headcount": 3,
        "life_expectancy": 77.6,
    },
    "LV" : {
        "name": "Latvia",
        "headcount": 4,
        "life_expectancy": 75.6,
    },
    "NL" : {
        "name": "Netherlands",
        "headcount": 60,
        "life_expectancy": 81.9,
    },
    "NO" : {
        "name": "Norway",
        "headcount": 14,
        "life_expectancy": 83.1,
    },
    "PK" : {
        "name": "Pakistan",
        "headcount": 5,
        "life_expectancy": 66.43,
    },
    "PL" : {
        "name": "Poland",
        "headcount": 94,
        "life_expectancy": 78.4,
    },
    "PT" : {
        "name": "Portugal",
        "headcount": 61,
        "life_expectancy": 82.5,
    },
    "RO" : {
        "name": "Romania",
        "headcount": 24,
        "life_expectancy": 76.4,
    },
    "RS" : {
        "name": "Serbia",
        "headcount": 5,
        "life_expectancy": 76.2,
    },
    "SE" : {
        "name": "Sweden",
        "headcount": 26,
        "life_expectancy": 82.4,
    },
    "SI" : {
        "name": "Slovenia",
        "headcount": 1,
        "life_expectancy": 82.0,
    },
    "SK" : {
        "name": "Slovakia",
        "headcount": 14,
        "life_expectancy": 78.2,
    },
    "TR" : {
        "name": "Turkey",
        "headcount": 7,
        "life_expectancy": 77.3,
    },
    "UA" : {
        "name": "Ukraine",
        "headcount": 2,
        "life_expectancy": 68.59,
    },
    "NMS" : {
        "name": "Non-Member States",
        "headcount": 12,
        "life_expectancy": 73.16,
    }
}

# SALARY DISTRIBUTION
# Staff salary distribution
staff_salary = {
    "grade_1" : {
        "salary": 4082,
        "c_factor": 1.3400
    },
    "grade_2" : {
        "salary": 4857,
        "c_factor": 1.3202
    },
    "grade_3" : {
        "salary": 5780,
        "c_factor": 1.2995
    },
    "grade_4" : {
        "salary": 6879,
        "c_factor": 1.2785
    },
    "grade_5" : {
        "salary": 8186,
        "c_factor": 1.2580
    },
    "grade_6" : {
        "salary": 9004,
        "c_factor": 1.2473
    },
    "grade_7" : {
        "salary": 10715,
        "c_factor": 1.2296
    }
}

staff_grade_1 = 4082
staff_grade_2 = 4857
staff_grade_3 = 5780
staff_grade_4 = 6879
staff_grade_5 = 8186
staff_grade_6 = 9004
staff_grade_7 = 10715

staff_category_grade_estimate = {
    "Research Physicists": {
        "staff_count": 85,
        "age_distribution": {
            "≤25": 0,
            "26–30": 0,
            "31–35": 18,
            "36–40": 23,
            "41–45": 8,
            "46–50": 7,
            "51–55": 8,
            "56–60": 8,
            "61–65": 13,
            ">65": 0
        },
        "estimated_grade_range": [6, 7]
    },
    "Scientific & Engineering work": {
        "staff_count": 1263,
        "age_distribution": {
            "≤25": 1,
            "26–30": 38,
            "31–35": 162,
            "36–40": 191,
            "41–45": 189,
            "46–50": 179,
            "51–55": 188,
            "56–60": 201,
            "61–65": 113,
            ">65": 1
        },
        "estimated_grade_range": [4, 6]
    },
    "Technical work": {
        "staff_count": 809,
        "age_distribution": {
            "≤25": 7,
            "26–30": 47,
            "31–35": 94,
            "36–40": 100,
            "41–45": 139,
            "46–50": 129,
            "51–55": 136,
            "56–60": 113,
            "61–65": 44,
            ">65": 0
        },
        "estimated_grade_range": [3, 5]
    },
    "Manual work": {
        "staff_count": 50,
        "age_distribution": {
            "≤25": 0,
            "26–30": 2,
            "31–35": 4,
            "36–40": 16,
            "41–45": 10,
            "46–50": 8,
            "51–55": 5,
            "56–60": 5,
            "61–65": 0,
            ">65": 0
        },
        "estimated_grade_range": [1, 3]
    },
    "Professional Admin work": {
        "staff_count": 184,
        "age_distribution": {
            "≤25": 0,
            "26–30": 0,
            "31–35": 9,
            "36–40": 27,
            "41–45": 25,
            "46–50": 30,
            "51–55": 42,
            "56–60": 35,
            "61–65": 16,
            ">65": 0
        },
        "estimated_grade_range": [4, 6]
    },
    "Office and Admin work": {
        "staff_count": 275,
        "age_distribution": {
            "≤25": 0,
            "26–30": 17,
            "31–35": 37,
            "36–40": 34,
            "41–45": 44,
            "46–50": 45,
            "51–55": 50,
            "56–60": 31,
            "61–65": 17,
            ">65": 0  # there's 1 >65 total in the dataset; we assign it here for completeness
        },
        "estimated_grade_range": [1, 3]
    }
}

# graduates salary distribution
#graduate reference salary
early_career_graduates = 4569
bsc_early_career_graduates = 5134
msc_early_career_graduates = 5647
#fellow reference salary
msc_2_4_years_experienced_graduates = 6212
msc_4_6_years_experienced_graduates = 6828
phd_0_3_years_experienced_graduates = 6828
phd_3_6_years_experienced_graduates = 7239

graduate_fellow_salary = {
    "grade_1" : {
        "name": "Early Career",
        "salary": 4569,
        "category": "graduate"
    },
    "grade_2" : {
        "name": "BSc Early Career",
        "salary": 5134,
        "category": "graduate"
    },
    "grade_3" : {
        "name": "MSc Early Career",
        "salary": 5647,
        "category": "graduate"
    },
    "grade_4" : {
        "name": "MSc 2-4 Years Experienced",
        "salary": 6212,
        "category": "fellow"
    },
    "grade_5" : {
        "name": "MSc 4-6 Years Experienced",
        "salary": 6828,
        "category": "fellow"
    },
    "grade_6" : {
        "name": "PhD 0-3 Years Experienced",
        "salary": 6828,
        "category": "fellow"
    },
    "grade_7" : {
        "name": "PhD 3-6 Years Experienced",
        "salary": 7239,
        "category": "fellow"
    }
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