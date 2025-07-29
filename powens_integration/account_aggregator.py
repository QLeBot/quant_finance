"""
Account Aggregator for Powens Integration

This module provides account aggregation functionality for multiple financial accounts
and portfolio analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from .powens_client import PowensClient

class AccountAggregator:
    """
    Account aggregator for managing multiple financial accounts
    
    Provides portfolio analysis, risk metrics, and account synchronization
    """
    
    def __init__(self, powens_client: PowensClient):
        """
        Initialize account aggregator
        
        Args:
            powens_client: Initialized Powens client
        """
        self.client = powens_client
        self.accounts_cache = {}
        self.transactions_cache = {}
        self.balances_cache = {}
    
    def get_all_accounts(self, user_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get all accounts for a user as a DataFrame
        
        Args:
            user_id: User identifier
            force_refresh: Force refresh from API
            
        Returns:
            DataFrame with account information
        """
        cache_key = f"accounts_{user_id}"
        
        if not force_refresh and cache_key in self.accounts_cache:
            return self.accounts_cache[cache_key]
        
        accounts = self.client.get_accounts(user_id)
        df = pd.DataFrame(accounts)
        
        # Add useful computed columns
        if not df.empty:
            df['total_balance'] = df['balance'].fillna(0) + df['available_balance'].fillna(0)
            df['account_type'] = df['type'].fillna('unknown')
            df['last_sync'] = pd.to_datetime(df['last_sync_date'])
        
        self.accounts_cache[cache_key] = df
        return df
    
    def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get portfolio summary across all accounts
        
        Args:
            user_id: User identifier
            
        Returns:
            Portfolio summary statistics
        """
        accounts_df = self.get_all_accounts(user_id)
        
        if accounts_df.empty:
            return {
                "total_balance": 0,
                "total_accounts": 0,
                "account_types": {},
                "institutions": {},
                "last_sync": None
            }
        
        summary = {
            "total_balance": accounts_df['total_balance'].sum(),
            "total_accounts": len(accounts_df),
            "account_types": accounts_df['account_type'].value_counts().to_dict(),
            "institutions": accounts_df['institution_name'].value_counts().to_dict(),
            "last_sync": accounts_df['last_sync'].max().isoformat() if not accounts_df['last_sync'].isna().all() else None
        }
        
        return summary
    
    def get_transactions_summary(self, user_id: str, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get transactions summary across all accounts
        
        Args:
            user_id: User identifier
            start_date: Start date for transaction range
            end_date: End date for transaction range
            
        Returns:
            DataFrame with aggregated transactions
        """
        accounts_df = self.get_all_accounts(user_id)
        
        if accounts_df.empty:
            return pd.DataFrame()
        
        all_transactions = []
        
        for _, account in accounts_df.iterrows():
            try:
                transactions = self.client.get_transactions(
                    user_id, 
                    account['id'], 
                    start_date, 
                    end_date
                )
                
                for transaction in transactions:
                    transaction['account_id'] = account['id']
                    transaction['account_name'] = account['name']
                    transaction['institution_name'] = account['institution_name']
                    all_transactions.append(transaction)
                    
            except Exception as e:
                print(f"Error fetching transactions for account {account['id']}: {e}")
                continue
        
        if not all_transactions:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_transactions)
        
        # Convert date columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Convert amount to numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        return df
    
    def calculate_portfolio_metrics(self, user_id: str) -> Dict[str, float]:
        """
        Calculate portfolio risk and performance metrics
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of portfolio metrics
        """
        accounts_df = self.get_all_accounts(user_id)
        
        if accounts_df.empty:
            return {}
        
        total_balance = accounts_df['total_balance'].sum()
        
        # Calculate concentration metrics
        account_weights = accounts_df['total_balance'] / total_balance
        herfindahl_index = (account_weights ** 2).sum()  # Concentration measure
        
        # Calculate diversification score (inverse of concentration)
        diversification_score = 1 - herfindahl_index
        
        # Calculate account type diversification
        type_weights = accounts_df.groupby('account_type')['total_balance'].sum() / total_balance
        type_diversification = 1 - (type_weights ** 2).sum()
        
        metrics = {
            "total_balance": total_balance,
            "number_of_accounts": len(accounts_df),
            "herfindahl_index": herfindahl_index,
            "diversification_score": diversification_score,
            "type_diversification": type_diversification,
            "average_account_balance": accounts_df['total_balance'].mean(),
            "largest_account_share": accounts_df['total_balance'].max() / total_balance
        }
        
        return metrics
    
    def sync_all_accounts(self, user_id: str) -> Dict[str, Any]:
        """
        Sync all accounts for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Sync results for each account
        """
        accounts_df = self.get_all_accounts(user_id)
        
        if accounts_df.empty:
            return {"message": "No accounts found", "synced_accounts": 0}
        
        sync_results = {}
        successful_syncs = 0
        
        for _, account in accounts_df.iterrows():
            try:
                result = self.client.sync_account(user_id, account['id'])
                sync_results[account['id']] = {
                    "status": "success",
                    "result": result,
                    "account_name": account['name']
                }
                successful_syncs += 1
            except Exception as e:
                sync_results[account['id']] = {
                    "status": "error",
                    "error": str(e),
                    "account_name": account['name']
                }
        
        # Clear caches after sync
        self._clear_caches(user_id)
        
        return {
            "total_accounts": len(accounts_df),
            "successful_syncs": successful_syncs,
            "failed_syncs": len(accounts_df) - successful_syncs,
            "results": sync_results
        }
    
    def get_cash_flow_analysis(self, user_id: str, 
                              months: int = 6) -> Dict[str, Any]:
        """
        Analyze cash flow patterns across accounts
        
        Args:
            user_id: User identifier
            months: Number of months to analyze
            
        Returns:
            Cash flow analysis results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        transactions_df = self.get_transactions_summary(user_id, start_date, end_date)
        
        if transactions_df.empty:
            return {"message": "No transaction data available"}
        
        # Group by month and calculate net cash flow
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
        monthly_flow = transactions_df.groupby('month')['amount'].sum()
        
        # Calculate cash flow metrics
        avg_monthly_flow = monthly_flow.mean()
        flow_volatility = monthly_flow.std()
        positive_months = (monthly_flow > 0).sum()
        total_months = len(monthly_flow)
        
        analysis = {
            "period_months": months,
            "total_transactions": len(transactions_df),
            "average_monthly_flow": avg_monthly_flow,
            "flow_volatility": flow_volatility,
            "positive_flow_months": positive_months,
            "total_months": total_months,
            "positive_flow_ratio": positive_months / total_months if total_months > 0 else 0,
            "monthly_flows": monthly_flow.to_dict()
        }
        
        return analysis
    
    def _clear_caches(self, user_id: str):
        """Clear cached data for a user"""
        cache_keys = [key for key in self.accounts_cache.keys() if user_id in key]
        for key in cache_keys:
            del self.accounts_cache[key] 