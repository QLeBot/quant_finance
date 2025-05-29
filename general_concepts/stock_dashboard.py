import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import yfinance as yf
from datetime import datetime
import streamlit as st
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="wide"
)

st.title("Stock Analysis Dashboard")

def get_metric_style(metric_name, value):
    """Determine the style of a metric based on its value and type."""
    if value is None:
        return None
    
    # Valuation metrics (lower is better)
    if metric_name in ['Price per earning', 'Enterprise value per ebitda', 'Price per free cash flow', 
                      'Enterprise value per free cash flow', 'Debt to equity']:
        if value < 15:  # Good threshold
            return "color: #006400; border-left: 4px solid #006400; padding: 10px; margin: 5px 0;"
        elif value < 30:  # Neutral threshold
            return "color: #FF8C00; border-left: 4px solid #FF8C00; padding: 10px; margin: 5px 0;"
        else:  # Bad threshold
            return "color: #8B0000; border-left: 4px solid #8B0000; padding: 10px; margin: 5px 0;"
    
    # Profitability metrics (higher is better)
    elif metric_name in ['ROE', 'ROIC', 'Gross margin', 'Operating margin', 'Net margin']:
        if value > 15:  # Good threshold
            return "color: #006400; border-left: 4px solid #006400; padding: 10px; margin: 5px 0;"
        elif value > 5:  # Neutral threshold
            return "color: #FF8C00; border-left: 4px solid #FF8C00; padding: 10px; margin: 5px 0;"
        else:  # Bad threshold
            return "color: #8B0000; border-left: 4px solid #8B0000; padding: 10px; margin: 5px 0;"
    
    # Liquidity metrics (higher is better)
    elif metric_name in ['Current ratio', 'Quick ratio', 'Cash ratio']:
        if value > 2:  # Good threshold
            return "color: #006400; border-left: 4px solid #006400; padding: 10px; margin: 5px 0;"
        elif value > 1:  # Neutral threshold
            return "color: #FF8C00; border-left: 4px solid #FF8C00; padding: 10px; margin: 5px 0;"
        else:  # Bad threshold
            return "color: #8B0000; border-left: 4px solid #8B0000; padding: 10px; margin: 5px 0;"
    
    # Dividend metrics
    elif metric_name in ['Dividend yield']:
        if value > 0.04:  # Good threshold (4%)
            return "color: #006400; border-left: 4px solid #006400; padding: 10px; margin: 5px 0;"
        elif value > 0.02:  # Neutral threshold (2%)
            return "color: #FF8C00; border-left: 4px solid #FF8C00; padding: 10px; margin: 5px 0;"
        else:  # Bad threshold
            return "color: #8B0000; border-left: 4px solid #8B0000; padding: 10px; margin: 5px 0;"
    
    elif metric_name in ['Payout ratio']:
        if value < 0.6:  # Good threshold (60%)
            return "color: #006400; border-left: 4px solid #006400; padding: 10px; margin: 5px 0;"
        elif value < 0.8:  # Neutral threshold (80%)
            return "color: #FF8C00; border-left: 4px solid #FF8C00; padding: 10px; margin: 5px 0;"
        else:  # Bad threshold
            return "color: #8B0000; border-left: 4px solid #8B0000; padding: 10px; margin: 5px 0;"
    
    return None

def display_metric(label, value, style):
    """Display a metric with custom styling."""
    st.markdown(f"""
        <div style="{style}">
            <p style="margin: 0;"><strong>{label}:</strong> {value}</p>
        </div>
    """, unsafe_allow_html=True)

def get_data(ticker, frequency):
    """Calculate financial ratios for each year."""
    try:
        # Get financial statements
        balance_sheet = ticker.get_balance_sheet(freq=frequency)
        income_stmt = ticker.get_income_stmt(freq=frequency)
        cash_flow = ticker.get_cash_flow(freq=frequency)
        
        # Get basic info that doesn't change with year
        sector = ticker.info['sector']
        industry = ticker.info['industry']
        market_cap = ticker.info['marketCap']
        
        # Get common dates across all statements
        common_dates = balance_sheet.columns.intersection(income_stmt.columns).intersection(cash_flow.columns)
        if len(common_dates) == 0:
            print(f"No common dates found for {ticker.info['symbol']}")
            return pd.DataFrame()
        
        # Initialize empty list to store ratios for each year
        ratios_list = []
        
        # Calculate ratios for each year
        for date in common_dates:
            try:
                # Get values for this year
                total_debt = balance_sheet.loc['TotalDebt', date] if 'TotalDebt' in balance_sheet.index else None
                total_equity = balance_sheet.loc['TotalEquityGrossMinorityInterest', date] if 'TotalEquityGrossMinorityInterest' in balance_sheet.index else None
                current_assets = balance_sheet.loc['CurrentAssets', date] if 'CurrentAssets' in balance_sheet.index else None
                inventory = balance_sheet.loc['Inventory', date] if 'Inventory' in balance_sheet.index else None
                current_liabilities = balance_sheet.loc['CurrentLiabilities', date] if 'CurrentLiabilities' in balance_sheet.index else None
                net_income = income_stmt.loc['NetIncome', date] if 'NetIncome' in income_stmt.index else None
                revenue = income_stmt.loc['TotalRevenue', date] if 'TotalRevenue' in income_stmt.index else None
                ebitda = income_stmt.loc['EBITDA', date] if 'EBITDA' in income_stmt.index else None
                free_cash_flow = cash_flow.loc['FreeCashFlow', date] if 'FreeCashFlow' in cash_flow.index else None
                gross_profit = income_stmt.loc['GrossProfit', date] if 'GrossProfit' in income_stmt.index else None
                operating_income = income_stmt.loc['OperatingIncome', date] if 'OperatingIncome' in income_stmt.index else None
                
                # Calculate enterprise value for this year
                enterprise_value = market_cap + total_debt if total_debt is not None else None
                
                # Calculate ratios
                ratios = {
                    'Sector': sector,
                    'Industry': industry,
                    'Market cap': market_cap,
                    'Date': date,
                    'Enterprise value': enterprise_value,
                    'Price per share': market_cap / revenue if revenue and revenue != 0 else None,
                    'Price per earning': market_cap / net_income if net_income and net_income != 0 else None,
                    'Enterprise value per ebitda': enterprise_value / ebitda if ebitda and ebitda != 0 else None,
                    'Price per free cash flow': market_cap / free_cash_flow if free_cash_flow and free_cash_flow != 0 else None,
                    'Enterprise value per free cash flow': enterprise_value / free_cash_flow if free_cash_flow and free_cash_flow != 0 else None,
                    'ROE': (net_income / total_equity * 100) if total_equity and total_equity != 0 and net_income else None,
                    'ROIC': (net_income / (total_equity + total_debt) * 100) if (total_equity and total_debt and (total_equity + total_debt) != 0 and net_income) else None,
                    'Cash ratio': current_assets / total_debt if total_debt and total_debt != 0 and current_assets else None,
                    'Current ratio': current_assets / current_liabilities if current_liabilities and current_liabilities != 0 and current_assets else None,
                    'Quick ratio': (current_assets - inventory) / current_liabilities if current_liabilities and current_liabilities != 0 and current_assets and inventory is not None else None,
                    'Debt to equity': total_debt / total_equity if total_equity and total_equity != 0 and total_debt else None,
                    'Dividend yield': ticker.info.get('dividendYield', 0),
                    'Payout ratio': ticker.info.get('payoutRatio', 0),
                    'Gross margin': (gross_profit / revenue * 100) if revenue and revenue != 0 and gross_profit else None,
                    'Operating margin': (operating_income / revenue * 100) if revenue and revenue != 0 and operating_income else None,
                    'Net margin': (net_income / revenue * 100) if revenue and revenue != 0 and net_income else None
                }
                ratios_list.append(ratios)
            except Exception as e:
                print(f"Error calculating ratios for {ticker.info['symbol']} for date {date}: {str(e)}")
                continue
        
        # Convert to DataFrame
        ratios_df = pd.DataFrame(ratios_list)
        return ratios_df
    except Exception as e:
        print(f"Error getting financial data for {ticker.info['symbol']}: {str(e)}")
        return pd.DataFrame()

# divide the page into two columns
col1, col2 = st.columns(2)

with col1:
    ticker_symbol = st.text_input("Enter Ticker Symbol", value="", max_chars=10).upper()

with col2:
    frequency = st.selectbox("Select Frequency", ["yearly", "quarterly"])

if not ticker_symbol:
    st.error("Please enter a ticker symbol")
    st.stop()

else:
    ticker = yf.Ticker(ticker_symbol)
    data = get_data(ticker, frequency)
    
    # Company Overview Section
    st.header(f"Company Overview: {ticker_symbol}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sector", data['Sector'].iloc[0])
        st.metric("Industry", data['Industry'].iloc[0])
        st.metric("Market Cap", f"${data['Market cap'].iloc[0]:,.2f}")
    
    with col2:
        st.metric("Enterprise Value", f"${data['Enterprise value'].iloc[0]:,.2f}")
        st.metric("Dividend Yield", f"{data['Dividend yield'].iloc[0]:.2%}")
        st.metric("Payout Ratio", f"{data['Payout ratio'].iloc[0]:.2%}")
    
    with col3:
        st.metric("Latest Date", data['Date'].iloc[0].strftime('%Y-%m-%d'))
    
    # Create tabs for different categories of metrics
    tab1, tab2, tab3, tab4 = st.tabs(["Valuation Metrics", "Profitability Metrics", "Financial Health", "Historical Data"])
    
    with tab1:
        st.subheader("Valuation Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Price per share'], mode='lines+markers', name='Price/Share'))
            fig.update_layout(title='Price per Share Over Time', xaxis_title='Date', yaxis_title='Price/Share')
            st.plotly_chart(fig, use_container_width=True)
            
            latest_price_share = data['Price per share'].iloc[0]
            latest_price_earning = data['Price per earning'].iloc[0]
            
            display_metric("Price per Share", f"${latest_price_share:,.2f}", 
                         get_metric_style('Price per earning', latest_price_earning))
            display_metric("Price per Earning", f"{latest_price_earning:,.2f}", 
                         get_metric_style('Price per earning', latest_price_earning))
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Enterprise value per ebitda'], mode='lines+markers', name='EV/EBITDA'))
            fig.update_layout(title='Enterprise Value/EBITDA Over Time', xaxis_title='Date', yaxis_title='EV/EBITDA')
            st.plotly_chart(fig, use_container_width=True)
            
            latest_ev_ebitda = data['Enterprise value per ebitda'].iloc[0]
            latest_price_fcf = data['Price per free cash flow'].iloc[0]
            
            display_metric("Enterprise Value/EBITDA", f"{latest_ev_ebitda:,.2f}", 
                         get_metric_style('Enterprise value per ebitda', latest_ev_ebitda))
            display_metric("Price per Free Cash Flow", f"{latest_price_fcf:,.2f}", 
                         get_metric_style('Price per free cash flow', latest_price_fcf))
    
    with tab2:
        st.subheader("Profitability Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['ROE'], mode='lines+markers', name='ROE'))
            fig.update_layout(title='Return on Equity (ROE) Over Time', xaxis_title='Date', yaxis_title='ROE (%)')
            st.plotly_chart(fig, use_container_width=True)
            
            latest_roe = data['ROE'].iloc[0]
            latest_roic = data['ROIC'].iloc[0]
            
            display_metric("ROE", f"{latest_roe:.2f}%", 
                         get_metric_style('ROE', latest_roe))
            display_metric("ROIC", f"{latest_roic:.2f}%", 
                         get_metric_style('ROIC', latest_roic))
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Gross margin'], mode='lines+markers', name='Gross Margin'))
            fig.update_layout(title='Margins Over Time', xaxis_title='Date', yaxis_title='Margin (%)')
            st.plotly_chart(fig, use_container_width=True)
            
            latest_gross_margin = data['Gross margin'].iloc[0]
            latest_operating_margin = data['Operating margin'].iloc[0]
            latest_net_margin = data['Net margin'].iloc[0]
            
            display_metric("Gross Margin", f"{latest_gross_margin:.2f}%", 
                         get_metric_style('Gross margin', latest_gross_margin))
            display_metric("Operating Margin", f"{latest_operating_margin:.2f}%", 
                         get_metric_style('Operating margin', latest_operating_margin))
            display_metric("Net Margin", f"{latest_net_margin:.2f}%", 
                         get_metric_style('Net margin', latest_net_margin))
    
    with tab3:
        st.subheader("Financial Health")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Current ratio'], mode='lines+markers', name='Current Ratio'))
            fig.update_layout(title='Liquidity Ratios Over Time', xaxis_title='Date', yaxis_title='Ratio')
            st.plotly_chart(fig, use_container_width=True)
            
            latest_current_ratio = data['Current ratio'].iloc[0]
            latest_quick_ratio = data['Quick ratio'].iloc[0]
            latest_cash_ratio = data['Cash ratio'].iloc[0]
            
            display_metric("Current Ratio", f"{latest_current_ratio:.2f}", 
                         get_metric_style('Current ratio', latest_current_ratio))
            display_metric("Quick Ratio", f"{latest_quick_ratio:.2f}", 
                         get_metric_style('Quick ratio', latest_quick_ratio))
            display_metric("Cash Ratio", f"{latest_cash_ratio:.2f}", 
                         get_metric_style('Cash ratio', latest_cash_ratio))
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Debt to equity'], mode='lines+markers', name='Debt/Equity'))
            fig.update_layout(title='Debt to Equity Over Time', xaxis_title='Date', yaxis_title='Debt/Equity')
            st.plotly_chart(fig, use_container_width=True)
            
            latest_debt_equity = data['Debt to equity'].iloc[0]
            latest_dividend_yield = data['Dividend yield'].iloc[0]
            latest_payout_ratio = data['Payout ratio'].iloc[0]
            
            display_metric("Debt to Equity", f"{latest_debt_equity:.2f}", 
                         get_metric_style('Debt to equity', latest_debt_equity))
            display_metric("Dividend Yield", f"{latest_dividend_yield:.2%}", 
                         get_metric_style('Dividend yield', latest_dividend_yield))
            display_metric("Payout Ratio", f"{latest_payout_ratio:.2%}", 
                         get_metric_style('Payout ratio', latest_payout_ratio))
    
    with tab4:
        st.subheader("Historical Data")
        st.dataframe(data.sort_values('Date', ascending=False).style.format({
            'Market cap': '${:,.2f}',
            'Enterprise value': '${:,.2f}',
            'Price per share': '${:,.2f}',
            'Price per earning': '{:,.2f}',
            'Enterprise value per ebitda': '{:,.2f}',
            'Price per free cash flow': '${:,.2f}',
            'Enterprise value per free cash flow': '${:,.2f}',
            'ROE': '{:.2f}%',
            'ROIC': '{:.2f}%',
            'Cash ratio': '{:.2f}',
            'Current ratio': '{:.2f}',
            'Quick ratio': '{:.2f}',
            'Debt to equity': '{:.2f}',
            'Dividend yield': '{:.2%}',
            'Payout ratio': '{:.2%}',
            'Gross margin': '{:.2f}%',
            'Operating margin': '{:.2f}%',
            'Net margin': '{:.2f}%'
        }))


