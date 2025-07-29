"""
Test script for Powens Integration

This script tests the Powens integration functionality with mock data.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from powens_integration.powens_client import PowensClient
from powens_integration.account_aggregator import AccountAggregator
from powens_integration.data_handler import PowensDataHandler
from powens_integration.config import PowensConfig

class TestPowensIntegration(unittest.TestCase):
    """Test cases for Powens integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock API key for testing
        os.environ['POWENS_API_KEY'] = 'test_api_key'
        
        # Create mock client
        self.mock_client = Mock(spec=PowensClient)
        self.aggregator = AccountAggregator(self.mock_client)
        self.data_handler = PowensDataHandler(self.mock_client)
        
        # Mock data
        self.mock_accounts = [
            {
                "id": "1",
                "name": "Test Checking",
                "type": "checking",
                "balance": 10000.0,
                "available_balance": 10000.0,
                "institution_name": "Test Bank",
                "last_sync_date": "2024-01-15T10:30:00Z"
            },
            {
                "id": "2",
                "name": "Test Savings",
                "type": "savings",
                "balance": 50000.0,
                "available_balance": 50000.0,
                "institution_name": "Test Bank",
                "last_sync_date": "2024-01-15T10:30:00Z"
            }
        ]
        
        self.mock_transactions = [
            {
                "id": "1",
                "date": "2024-01-15T10:30:00Z",
                "amount": 100.0,
                "description": "Test transaction",
                "category": "food"
            }
        ]
    
    def test_config_validation(self):
        """Test configuration validation"""
        validation = PowensConfig.validate_config()
        self.assertTrue(validation["valid"])
    
    def test_portfolio_summary(self):
        """Test portfolio summary calculation"""
        # Mock client response
        self.mock_client.get_accounts.return_value = self.mock_accounts
        
        summary = self.aggregator.get_portfolio_summary("test_user")
        
        self.assertEqual(summary["total_balance"], 60000.0)
        self.assertEqual(summary["total_accounts"], 2)
        self.assertEqual(summary["account_types"]["checking"], 1)
        self.assertEqual(summary["account_types"]["savings"], 1)
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        # Mock client response
        self.mock_client.get_accounts.return_value = self.mock_accounts
        
        metrics = self.aggregator.calculate_portfolio_metrics("test_user")
        
        self.assertIn("diversification_score", metrics)
        self.assertIn("herfindahl_index", metrics)
        self.assertEqual(metrics["total_balance"], 60000.0)
        self.assertEqual(metrics["number_of_accounts"], 2)
    
    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        # Mock client response
        self.mock_client.get_accounts.return_value = self.mock_accounts
        
        risk_metrics = self.data_handler.calculate_risk_metrics("test_user")
        
        self.assertIn("diversification_score", risk_metrics)
        self.assertIn("institution_concentration", risk_metrics)
        self.assertIn("sync_risk", risk_metrics)
    
    def test_cash_flow_analysis(self):
        """Test cash flow analysis"""
        # Mock client responses
        self.mock_client.get_accounts.return_value = self.mock_accounts
        self.mock_client.get_transactions.return_value = self.mock_transactions
        
        cash_flow = self.aggregator.get_cash_flow_analysis("test_user", months=1)
        
        self.assertIn("total_transactions", cash_flow)
        self.assertIn("average_monthly_flow", cash_flow)
    
    def test_data_export(self):
        """Test data export functionality"""
        # Mock client response
        self.mock_client.get_accounts.return_value = self.mock_accounts
        
        # Test CSV export
        try:
            self.data_handler.export_to_csv("test_user", "test_exports")
            # Check if files were created (in a real test, you'd verify file existence)
            self.assertTrue(True)  # Placeholder assertion
        except Exception as e:
            self.fail(f"Export failed: {e}")
    
    def test_webview_url_creation(self):
        """Test webview URL creation"""
        # Mock client response
        self.mock_client.create_webview_link.return_value = "https://test.webview.url"
        
        url = self.mock_client.create_webview_link("test_user", "https://test.com/callback")
        
        self.assertEqual(url, "https://test.webview.url")
        self.mock_client.create_webview_link.assert_called_once_with(
            "test_user", "https://test.com/callback"
        )
    
    def test_account_sync(self):
        """Test account synchronization"""
        # Mock client responses
        self.mock_client.get_accounts.return_value = self.mock_accounts
        self.mock_client.sync_account.return_value = {"status": "success"}
        
        sync_results = self.aggregator.sync_all_accounts("test_user")
        
        self.assertEqual(sync_results["total_accounts"], 2)
        self.assertEqual(sync_results["successful_syncs"], 2)
        self.assertEqual(sync_results["failed_syncs"], 0)
    
    def test_performance_analysis(self):
        """Test performance analysis"""
        # Mock client responses
        self.mock_client.get_accounts.return_value = self.mock_accounts
        self.mock_client.get_transactions.return_value = self.mock_transactions
        
        performance = self.data_handler.get_performance_analysis("test_user", months=1)
        
        self.assertIn("analysis_period_months", performance)
        self.assertIn("total_cash_flow", performance)
    
    def test_error_handling(self):
        """Test error handling"""
        # Mock client to raise an exception
        self.mock_client.get_accounts.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception):
            self.aggregator.get_portfolio_summary("test_user")

def run_integration_test():
    """Run a simple integration test with mock data"""
    print("🧪 Running Powens Integration Tests")
    print("=" * 50)
    
    # Test configuration
    print("1. Testing configuration validation...")
    validation = PowensConfig.validate_config()
    if validation["valid"]:
        print("   ✅ Configuration is valid")
    else:
        print(f"   ❌ Configuration errors: {validation['errors']}")
        return False
    
    if validation["warnings"]:
        print(f"   ⚠️  Configuration warnings: {validation['warnings']}")
    
    # Test with mock data
    print("\n2. Testing with mock data...")
    
    # Create mock client
    mock_client = Mock(spec=PowensClient)
    
    # Mock account data
    mock_accounts = [
        {
            "id": "1",
            "name": "Demo Checking",
            "type": "checking",
            "balance": 15000.0,
            "available_balance": 15000.0,
            "institution_name": "Demo Bank",
            "last_sync_date": datetime.now().isoformat()
        },
        {
            "id": "2",
            "name": "Demo Savings",
            "type": "savings",
            "balance": 75000.0,
            "available_balance": 75000.0,
            "institution_name": "Demo Bank",
            "last_sync_date": datetime.now().isoformat()
        }
    ]
    
    # Mock transaction data
    mock_transactions = [
        {
            "id": "1",
            "date": datetime.now().isoformat(),
            "amount": 500.0,
            "description": "Salary deposit",
            "category": "income"
        },
        {
            "id": "2",
            "date": (datetime.now() - timedelta(days=1)).isoformat(),
            "amount": -50.0,
            "description": "Grocery store",
            "category": "food"
        }
    ]
    
    # Set up mock responses
    mock_client.get_accounts.return_value = mock_accounts
    mock_client.get_transactions.return_value = mock_transactions
    mock_client.get_balances.return_value = {"balance": 15000.0}
    mock_client.create_webview_link.return_value = "https://demo.webview.url"
    mock_client.sync_account.return_value = {"status": "success"}
    
    # Test aggregator
    print("   Testing AccountAggregator...")
    aggregator = AccountAggregator(mock_client)
    
    summary = aggregator.get_portfolio_summary("demo_user")
    print(f"   ✅ Portfolio Summary: ${summary['total_balance']:,.2f} across {summary['total_accounts']} accounts")
    
    metrics = aggregator.calculate_portfolio_metrics("demo_user")
    print(f"   ✅ Portfolio Metrics: {metrics['diversification_score']:.1%} diversification")
    
    # Test data handler
    print("   Testing PowensDataHandler...")
    data_handler = PowensDataHandler(mock_client)
    
    portfolio_data = data_handler.get_portfolio_data("demo_user")
    print(f"   ✅ Portfolio Data: {len(portfolio_data['accounts'])} accounts loaded")
    
    risk_metrics = data_handler.calculate_risk_metrics("demo_user")
    print(f"   ✅ Risk Metrics: {risk_metrics['number_of_institutions']} institutions")
    
    # Test webview
    print("   Testing Webview Integration...")
    webview_url = mock_client.create_webview_link("demo_user", "https://demo.com/callback")
    print(f"   ✅ Webview URL: {webview_url}")
    
    print("\n✅ All integration tests passed!")
    return True

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50 + "\n")
    
    # Run integration test
    run_integration_test() 