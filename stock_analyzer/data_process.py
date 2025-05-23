import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import yfinance as yf

def get_info(ticker):
    ticker = yf.Ticker(ticker)
    return ticker.info['industry'], ticker.info['sector'], ticker.info['country'], ticker.info['marketCap']

def calculate_ratios(ticker):
    """Calculate financial ratios for each year."""
    try:
        # Get financial statements
        balance_sheet = ticker.get_balance_sheet(freq='yearly')
        income_stmt = ticker.get_income_stmt(freq='yearly')
        cash_flow = ticker.get_cash_flow(freq='yearly')
        
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
    
