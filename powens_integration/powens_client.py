"""
Powens API Client

This module provides a client for interacting with the Powens API.
Powens offers financial data aggregation and account management services.
"""

import os
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

class PowensClient:
    """
    Client for interacting with Powens API
    
    Powens provides:
    - Account aggregation
    - Financial data APIs
    - Webview components for user authentication
    - Real-time financial data
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.powens.com"):
        """
        Initialize Powens client
        
        Args:
            api_key: Powens API key (will load from POWENS_API_KEY env var if not provided)
            base_url: Base URL for Powens API
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("POWENS_API_KEY")
        if not self.api_key:
            raise ValueError("POWENS_API_KEY environment variable is required")
            
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def get_accounts(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get user's aggregated accounts
        
        Args:
            user_id: User identifier
            
        Returns:
            List of account information
        """
        endpoint = f"{self.base_url}/v1/users/{user_id}/accounts"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    def get_transactions(self, user_id: str, account_id: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get transactions for a specific account
        
        Args:
            user_id: User identifier
            account_id: Account identifier
            start_date: Start date for transaction range
            end_date: End date for transaction range
            
        Returns:
            List of transactions
        """
        endpoint = f"{self.base_url}/v1/users/{user_id}/accounts/{account_id}/transactions"
        
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_balances(self, user_id: str, account_id: str) -> Dict[str, Any]:
        """
        Get current balance for an account
        
        Args:
            user_id: User identifier
            account_id: Account identifier
            
        Returns:
            Account balance information
        """
        endpoint = f"{self.base_url}/v1/users/{user_id}/accounts/{account_id}/balances"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    def create_webview_link(self, user_id: str, redirect_url: str, 
                           institutions: Optional[List[str]] = None) -> str:
        """
        Create a webview link for user authentication
        
        Args:
            user_id: User identifier
            redirect_url: URL to redirect after authentication
            institutions: List of institution IDs to show (optional)
            
        Returns:
            Webview authentication URL
        """
        endpoint = f"{self.base_url}/v1/webview/links"
        
        data = {
            "user_id": user_id,
            "redirect_url": redirect_url,
            "institutions": institutions or []
        }
        
        response = self.session.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()["webview_url"]
    
    def get_institutions(self) -> List[Dict[str, Any]]:
        """
        Get list of available financial institutions
        
        Returns:
            List of institution information
        """
        endpoint = f"{self.base_url}/v1/institutions"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    def sync_account(self, user_id: str, account_id: str) -> Dict[str, Any]:
        """
        Trigger account synchronization
        
        Args:
            user_id: User identifier
            account_id: Account identifier
            
        Returns:
            Sync status information
        """
        endpoint = f"{self.base_url}/v1/users/{user_id}/accounts/{account_id}/sync"
        response = self.session.post(endpoint)
        response.raise_for_status()
        return response.json()
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile information
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile data
        """
        endpoint = f"{self.base_url}/v1/users/{user_id}/profile"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json() 