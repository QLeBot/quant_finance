import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from fund_members_data import staff_salary, graduate_fellow_salary
from fund_members_data import staff_category_grade_estimate, graduates_category_grade_estimate
from fund_members_data import calculate_pension_contribution, estimate_staff_contributions_df, estimate_graduate_fellow_contributions_df
from fund_members_data import calculate_retirement_pension, calculate_transfer_value, calculate_deferred_pension, calculate_surviving_spouse_pension, calculate_orphan_pension, calculate_disability_pension
from fund_members_data import calculate_average_metrics
from fund_members_data import retirement_pensions, deferred_pensions, surviving_spouses_pensions, orphans_pensions, disability_pensions
from fund_portfolio_data import saa, returns_2023
from fund_projection_simulation import FundProjectionSimulator
from fund_portfolio_rebalancer import PortfolioRebalancer

# Page config
st.set_page_config(
    page_title="CERN Pension Fund Dashboard",
    #page_icon="📊",
    layout="wide"
)

# Title
st.title("CERN Pension Fund Dashboard")
st.markdown("""
This dashboard provides insights into the CERN Pension Fund's current status and future projections.
Use the sidebar to adjust simulation parameters and explore different scenarios.
""")

# Add warning message
st.warning("""
⚠️ **Disclaimer**: The projections and scenarios presented in this dashboard are based on available data and assumptions. They should not be considered as financial advice or guarantees of future performance.
""")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Economic parameters
st.sidebar.subheader("Economic Parameters")
annual_return_rate = st.sidebar.slider(
    "Expected Annual Return Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
) / 100

volatility = st.sidebar.slider(
    "Annual Volatility (%)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.1
) / 100

inflation_rate = st.sidebar.slider(
    "Expected Annual Inflation Rate (%)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.1
) / 100

# Time horizon
years = st.sidebar.slider(
    "Simulation Period (Years)",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

# Investment strategy
st.sidebar.subheader("Investment Strategy")
strategy = st.sidebar.selectbox(
    "Select Investment Strategy",
    ["Conservative", "Balanced", "Aggressive"],
    index=1
)

# Adjust return and volatility based on strategy
if strategy == "Conservative":
    annual_return_rate = max(annual_return_rate - 0.01, 0.02)
    volatility = max(volatility - 0.02, 0.05)
elif strategy == "Aggressive":
    annual_return_rate = min(annual_return_rate + 0.01, 0.08)
    volatility = min(volatility + 0.02, 0.15)

# Main content
tab1, tab2 = st.tabs(["Current Status", "Scenario Analysis"])

with tab1:
    st.header("Current Fund Status")
    
    # Calculate current contributions
    staff_df = estimate_staff_contributions_df()
    grad_fellow_df = estimate_graduate_fellow_contributions_df()
    
    # Total monthly contributions
    total_monthly_contributions = (
        staff_df["Employee Contribution"].sum() + 
        staff_df["Organization Contribution"].sum() +
        grad_fellow_df["Employee Contribution"].sum() +
        grad_fellow_df["Organization Contribution"].sum()
    )
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Monthly Contributions",
            f"CHF {total_monthly_contributions:,.2f}"
        )
    with col2:
        st.metric(
            "Number of Beneficiaries",
            "3,600"
        )
    with col3:
        st.metric(
            "Initial Fund Value",
            "CHF 4.46B"
        )
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Contribution breakdown
        st.subheader("Contribution Breakdown")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Staff", "Graduates/Fellows"],
            y=[
                staff_df["Employee Contribution"].sum() + staff_df["Organization Contribution"].sum(),
                grad_fellow_df["Employee Contribution"].sum() + grad_fellow_df["Organization Contribution"].sum()
            ],
            name="Total Contributions"
        ))
        fig.update_layout(
            title="Monthly Contributions by Member Type",
            xaxis_title="Member Type",
            yaxis_title="Amount (CHF)",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Asset allocation pie chart
        st.subheader("Strategic Asset Allocation")
        
        # Format the SAA data for better display
        formatted_saa = {
            "Fixed Income": saa["fixed_income"] * 100,
            "Equity": saa["equity"] * 100,
            "Real Estate": saa["real_estate"] * 100,
            "Infrastructure": saa["infrastructure"] * 100,
            "Timber & Farmland": saa["timber_farmland"] * 100,
            "Private Equity": saa["private_equity"] * 100,
            "Hedge Funds": saa["hedge_funds"] * 100,
            "Metals & Commodities": saa["metals_commodities"] * 100,
            "Cash": saa["cash"] * 100
        }
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(formatted_saa.keys()),
            values=list(formatted_saa.values()),
            hole=0.3,
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        
        fig.update_layout(
            title="Current Strategic Asset Allocation",
            showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Add new section for cash flow analysis
    st.subheader("Cash Flow Analysis")
    
    # Calculate monthly contributions
    staff_df = estimate_staff_contributions_df()
    grad_fellow_df = estimate_graduate_fellow_contributions_df()
    
    total_monthly_contributions = (
        staff_df["Employee Contribution"].sum() + 
        staff_df["Organization Contribution"].sum() +
        grad_fellow_df["Employee Contribution"].sum() +
        grad_fellow_df["Organization Contribution"].sum()
    )
    
    # Calculate monthly payments
    avg_metrics = calculate_average_metrics()
    avg_salary = avg_metrics["average_salary"]
    avg_years = avg_metrics["average_years_of_service"]
    
    # Calculate monthly pension amounts
    monthly_retirement_pension = calculate_retirement_pension(avg_years, avg_salary)
    monthly_deferred_pension = calculate_deferred_pension(avg_years, avg_salary)
    monthly_surviving_spouse = calculate_surviving_spouse_pension(avg_years, avg_salary)
    monthly_orphan = calculate_orphan_pension(avg_years, avg_salary, 1)  # Assuming 1 child on average
    monthly_disability = calculate_disability_pension(avg_years, avg_salary)
    
    # Calculate total monthly payments
    total_monthly_payments = (
        retirement_pensions * monthly_retirement_pension +
        deferred_pensions * monthly_deferred_pension +
        surviving_spouses_pensions * monthly_surviving_spouse +
        orphans_pensions * monthly_orphan +
        disability_pensions * monthly_disability
    )
    
    # Display average metrics
    st.subheader("Average Member Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Age",
            f"{avg_metrics['average_age']:.1f} years"
        )
    with col2:
        st.metric(
            "Average Years of Service",
            f"{avg_metrics['average_years_of_service']:.1f} years"
        )
    with col3:
        st.metric(
            "Average Reference Salary",
            f"CHF {avg_metrics['average_salary']:,.2f}"
        )
    
    # Create time series data
    months = years * 12
    dates = pd.date_range(start='2023-01-01', periods=months, freq='M')
    
    # Simulate growth in contributions and payments
    contribution_growth = 0.02  # 2% annual growth in contributions
    payment_growth = 0.03  # 3% annual growth in payments
    
    contributions = [total_monthly_contributions * (1 + contribution_growth/12)**i for i in range(months)]
    payments = [total_monthly_payments * (1 + payment_growth/12)**i for i in range(months)]
    difference = [c - p for c, p in zip(contributions, payments)]
    
    # Create the plot
    fig = go.Figure()
    
    # Add contributions line
    fig.add_trace(go.Scatter(
        x=dates,
        y=contributions,
        name='Contributions',
        line=dict(color='green', width=2)
    ))
    
    # Add payments line
    fig.add_trace(go.Scatter(
        x=dates,
        y=payments,
        name='Payments',
        line=dict(color='red', width=2)
    ))
    
    # Add difference line
    fig.add_trace(go.Scatter(
        x=dates,
        y=difference,
        name='Surplus/Deficit',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title="Monthly Cash Flow Projection",
        xaxis_title="Date",
        yaxis_title="Amount (CHF)",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Monthly Contributions",
            f"CHF {total_monthly_contributions:,.2f}"
        )
    with col2:
        st.metric(
            "Current Monthly Payments",
            f"CHF {total_monthly_payments:,.2f}"
        )
    with col3:
        st.metric(
            "Current Monthly Surplus",
            f"CHF {total_monthly_contributions - total_monthly_payments:,.2f}"
        )


with tab2:
    st.header("Scenario Analysis")
    
    # Add rebalancing parameters to sidebar
    st.sidebar.subheader("Portfolio Rebalancing Parameters")
    max_deviation = st.sidebar.slider(
        "Maximum Deviation from Base SAA (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    ) / 100
    
    # Create different scenarios
    scenarios = {
        "Base Case": {
            "return": annual_return_rate,
            "volatility": volatility,
            "inflation": inflation_rate
        },
        "Optimistic": {
            "return": annual_return_rate + 0.02,
            "volatility": volatility - 0.02,
            "inflation": inflation_rate - 0.01
        },
        "Pessimistic": {
            "return": annual_return_rate - 0.02,
            "volatility": volatility + 0.02,
            "inflation": inflation_rate + 0.01
        }
    }
    
    # Initialize simulator and rebalancer
    simulator = FundProjectionSimulator()
    rebalancer = PortfolioRebalancer()
    
    # Run simulations for each scenario
    scenario_results = {}
    scenario_allocations = {}
    scenario_rationales = {}
    
    for scenario_name, params in scenarios.items():
        # Get rebalanced allocation
        new_allocation = rebalancer.rebalance_portfolio(scenario_name, max_deviation)
        scenario_allocations[scenario_name] = new_allocation
        scenario_rationales[scenario_name] = rebalancer.get_rebalancing_rationale(scenario_name, new_allocation)
        
        # Adjust return ranges based on scenario
        simulator.return_ranges = {
            "fixed_income": (params["return"] - 0.02, params["return"] + 0.02),
            "equity": (params["return"] - 0.15, params["return"] + 0.15),
            "real_estate": (params["return"] - 0.10, params["return"] + 0.10),
            "infrastructure": (params["return"] - 0.05, params["return"] + 0.05),
            "timber_farmland": (params["return"] - 0.05, params["return"] + 0.05),
            "private_equity": (params["return"] - 0.10, params["return"] + 0.10),
            "hedge_funds": (params["return"] - 0.05, params["return"] + 0.05),
            "metals_commodities": (params["return"] - 0.10, params["return"] + 0.10),
            "cash": (params["return"] - 0.01, params["return"] + 0.01)
        }
        
        # Run simulation with new allocation
        results = simulator.run_simulation(years, num_simulations=100)
        analysis = simulator.analyze_results(results)
        scenario_results[scenario_name] = analysis
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot scenario comparison
        fig = go.Figure()
        for scenario_name, analysis in scenario_results.items():
            fig.add_trace(go.Scatter(
                x=list(range(years + 1)),
                y=analysis['mean'],
                name=scenario_name,
                mode='lines'
            ))
        
        fig.update_layout(
            title="Fund Value Projection by Scenario",
            xaxis_title="Years",
            yaxis_title="Fund Value (CHF)",
            yaxis=dict(tickformat=".2s"),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display final values for each scenario
        st.subheader("Final Fund Values by Scenario")
        for scenario_name, analysis in scenario_results.items():
            st.metric(
                scenario_name,
                f"CHF {analysis['mean'][-1]/1e9:.2f}B",
                f"{((analysis['mean'][-1]/simulator.initial_value)**(1/years)-1)*100:.1f}% annual return"
            )
    
    # Create pie charts for each scenario's asset allocation
    st.subheader("Asset Allocation by Scenario")
    cols = st.columns(len(scenarios))
    
    for i, (scenario_name, allocation) in enumerate(scenario_allocations.items()):
        with cols[i]:
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=0.3,
                textinfo='label+percent',
                insidetextorientation='radial'
            )])
            
            fig.update_layout(
                title=f"{scenario_name} Allocation",
                showlegend=False,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show rebalancing rationale
            #with st.expander("Rebalancing Rationale"):
            #    for rationale in scenario_rationales[scenario_name]:
            #        st.write(rationale)
    
    # Display detailed statistics for each scenario
    st.subheader("Detailed Scenario Statistics")
    for scenario_name, analysis in scenario_results.items():
        with st.expander(f"{scenario_name} Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Initial Value",
                    f"CHF {simulator.initial_value/1e9:.2f}B"
                )
            with col2:
                st.metric(
                    "Final Mean Value",
                    f"CHF {analysis['mean'][-1]/1e9:.2f}B"
                )
            with col3:
                st.metric(
                    "Annualized Return",
                    f"{((analysis['mean'][-1]/simulator.initial_value)**(1/years)-1)*100:.2f}%"
                )
            
            # Display confidence intervals
            st.write("90% Confidence Interval:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "5th Percentile",
                    f"CHF {analysis['percentile_5'][-1]/1e9:.2f}B"
                )
            with col2:
                st.metric(
                    "95th Percentile",
                    f"CHF {analysis['percentile_95'][-1]/1e9:.2f}B"
                ) 