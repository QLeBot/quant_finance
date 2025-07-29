"""
Powens Data Handler

This module provides data handling and integration capabilities for Powens financial data
with existing quant finance analysis tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import yfinance as yf
from .powens_client import PowensClient
from .account_aggregator import AccountAggregator

class PowensDataHandler:
    """
    Data handler for Powens financial data integration
    
    Provides data processing, analysis, and integration with existing quant finance tools
    """
    
    def __init__(self, powens_client: PowensClient):
        """
        Initialize Powens data handler
        
        Args:
            powens_client: Initialized Powens client
        """
        self.client = powens_client
        self.aggregator = AccountAggregator(powens_client)
        self.data_cache = {}
    
    def get_portfolio_data(self, user_id: str, 
                          include_transactions: bool = True,
                          include_balances: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive portfolio data for analysis
        
        Args:
            user_id: User identifier
            include_transactions: Include transaction data
            include_balances: Include balance data
            
        Returns:
            Comprehensive portfolio data
        """
        portfolio_data = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "accounts": self.aggregator.get_all_accounts(user_id).to_dict('records'),
            "summary": self.aggregator.get_portfolio_summary(user_id),
            "metrics": self.aggregator.calculate_portfolio_metrics(user_id)
        }
        
        if include_transactions:
            # Get last 3 months of transactions
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            portfolio_data["transactions"] = self.aggregator.get_transactions_summary(
                user_id, start_date, end_date
            ).to_dict('records')
        
        if include_balances:
            portfolio_data["balances"] = self._get_all_balances(user_id)
        
        return portfolio_data
    
    def _get_all_balances(self, user_id: str) -> List[Dict[str, Any]]:
        """Get balances for all accounts"""
        accounts_df = self.aggregator.get_all_accounts(user_id)
        balances = []
        
        for _, account in accounts_df.iterrows():
            try:
                balance_data = self.client.get_balances(user_id, account['id'])
                balance_data['account_id'] = account['id']
                balance_data['account_name'] = account['name']
                balances.append(balance_data)
            except Exception as e:
                print(f"Error fetching balance for account {account['id']}: {e}")
                continue
        
        return balances
    
    def create_portfolio_dataframe(self, user_id: str) -> pd.DataFrame:
        """
        Create a pandas DataFrame with portfolio data for analysis
        
        Args:
            user_id: User identifier
            
        Returns:
            DataFrame with portfolio data
        """
        accounts_df = self.aggregator.get_all_accounts(user_id)
        
        if accounts_df.empty:
            return pd.DataFrame()
        
        # Add additional computed columns
        accounts_df['balance_share'] = accounts_df['total_balance'] / accounts_df['total_balance'].sum()
        accounts_df['days_since_sync'] = (datetime.now() - accounts_df['last_sync']).dt.days
        
        return accounts_df
    
    def get_cash_flow_dataframe(self, user_id: str, 
                               months: int = 6) -> pd.DataFrame:
        """
        Get cash flow data as DataFrame for analysis
        
        Args:
            user_id: User identifier
            months: Number of months to analyze
            
        Returns:
            DataFrame with cash flow data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        transactions_df = self.aggregator.get_transactions_summary(user_id, start_date, end_date)
        
        if transactions_df.empty:
            return pd.DataFrame()
        
        # Add useful columns for analysis
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
        transactions_df['week'] = transactions_df['date'].dt.to_period('W')
        transactions_df['day_of_week'] = transactions_df['date'].dt.day_name()
        transactions_df['is_positive'] = transactions_df['amount'] > 0
        
        return transactions_df
    
    def calculate_risk_metrics(self, user_id: str) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for the portfolio
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of risk metrics
        """
        accounts_df = self.aggregator.get_all_accounts(user_id)
        
        if accounts_df.empty:
            return {}
        
        total_balance = accounts_df['total_balance'].sum()
        
        # Concentration risk
        account_weights = accounts_df['total_balance'] / total_balance
        herfindahl_index = (account_weights ** 2).sum()
        
        # Institution concentration
        institution_weights = accounts_df.groupby('institution_name')['total_balance'].sum() / total_balance
        institution_concentration = (institution_weights ** 2).sum()
        
        # Account type diversification
        type_weights = accounts_df.groupby('account_type')['total_balance'].sum() / total_balance
        type_concentration = (type_weights ** 2).sum()
        
        # Sync risk (accounts not recently synced)
        sync_risk = (accounts_df['days_since_sync'] > 7).sum() / len(accounts_df)
        
        risk_metrics = {
            "total_balance": total_balance,
            "herfindahl_index": herfindahl_index,
            "institution_concentration": institution_concentration,
            "type_concentration": type_concentration,
            "sync_risk": sync_risk,
            "diversification_score": 1 - herfindahl_index,
            "institution_diversification": 1 - institution_concentration,
            "type_diversification": 1 - type_concentration,
            "number_of_accounts": len(accounts_df),
            "number_of_institutions": accounts_df['institution_name'].nunique(),
            "number_of_account_types": accounts_df['account_type'].nunique()
        }
        
        return risk_metrics
    
    def get_performance_analysis(self, user_id: str, 
                                months: int = 12) -> Dict[str, Any]:
        """
        Analyze portfolio performance over time
        
        Args:
            user_id: User identifier
            months: Number of months to analyze
            
        Returns:
            Performance analysis results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        transactions_df = self.aggregator.get_transactions_summary(user_id, start_date, end_date)
        
        if transactions_df.empty:
            return {"message": "No transaction data available for performance analysis"}
        
        # Monthly performance analysis
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
        monthly_performance = transactions_df.groupby('month')['amount'].sum()
        
        # Calculate performance metrics
        total_flow = monthly_performance.sum()
        avg_monthly_flow = monthly_performance.mean()
        flow_volatility = monthly_performance.std()
        
        # Calculate positive vs negative months
        positive_months = (monthly_performance > 0).sum()
        negative_months = (monthly_performance < 0).sum()
        
        # Calculate consistency metrics
        consistency_score = positive_months / len(monthly_performance) if len(monthly_performance) > 0 else 0
        
        performance_analysis = {
            "analysis_period_months": months,
            "total_cash_flow": total_flow,
            "average_monthly_flow": avg_monthly_flow,
            "flow_volatility": flow_volatility,
            "positive_months": positive_months,
            "negative_months": negative_months,
            "consistency_score": consistency_score,
            "monthly_performance": monthly_performance.to_dict(),
            "best_month": monthly_performance.idxmax().strftime('%Y-%m') if not monthly_performance.empty else None,
            "worst_month": monthly_performance.idxmin().strftime('%Y-%m') if not monthly_performance.empty else None
        }
        
        return performance_analysis
    
    def export_to_csv(self, user_id: str, output_dir: str = "powens_data"):
        """
        Export portfolio data to CSV files
        
        Args:
            user_id: User identifier
            output_dir: Output directory for CSV files
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export accounts data
        accounts_df = self.aggregator.get_all_accounts(user_id)
        if not accounts_df.empty:
            accounts_df.to_csv(f"{output_dir}/accounts_{user_id}.csv", index=False)
        
        # Export transactions data
        transactions_df = self.aggregator.get_transactions_summary(user_id)
        if not transactions_df.empty:
            transactions_df.to_csv(f"{output_dir}/transactions_{user_id}.csv", index=False)
        
        # Export portfolio summary
        summary = self.aggregator.get_portfolio_summary(user_id)
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{output_dir}/portfolio_summary_{user_id}.csv", index=False)
        
        # Export risk metrics
        risk_metrics = self.calculate_risk_metrics(user_id)
        risk_df = pd.DataFrame([risk_metrics])
        risk_df.to_csv(f"{output_dir}/risk_metrics_{user_id}.csv", index=False)
        
        print(f"Data exported to {output_dir}/")
    
    def get_webview_url(self, user_id: str, redirect_url: str) -> str:
        """
        Get webview URL for user authentication
        
        Args:
            user_id: User identifier
            redirect_url: URL to redirect after authentication
            
        Returns:
            Webview authentication URL
        """
        return self.client.create_webview_link(user_id, redirect_url) 