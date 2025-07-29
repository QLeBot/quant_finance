"""
Example Usage of Powens Integration

This script demonstrates how to use the Powens integration for financial data analysis.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from powens_client import PowensClient
from account_aggregator import AccountAggregator
from data_handler import PowensDataHandler

def main():
    """Main example function"""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Powens client
    try:
        client = PowensClient()
        print("✅ Powens client initialized successfully")
    except ValueError as e:
        print(f"❌ Error initializing Powens client: {e}")
        print("Please set POWENS_API_KEY environment variable")
        return
    
    # Initialize data handler
    data_handler = PowensDataHandler(client)
    aggregator = AccountAggregator(client)
    
    # Example user ID (replace with actual user ID)
    user_id = "example_user_123"
    
    print(f"\n🔍 Analyzing portfolio for user: {user_id}")
    print("=" * 50)
    
    # 1. Get portfolio summary
    try:
        summary = aggregator.get_portfolio_summary(user_id)
        print(f"\n📊 Portfolio Summary:")
        print(f"   Total Balance: ${summary['total_balance']:,.2f}")
        print(f"   Total Accounts: {summary['total_accounts']}")
        print(f"   Account Types: {summary['account_types']}")
        print(f"   Institutions: {summary['institutions']}")
    except Exception as e:
        print(f"❌ Error getting portfolio summary: {e}")
    
    # 2. Calculate portfolio metrics
    try:
        metrics = aggregator.calculate_portfolio_metrics(user_id)
        print(f"\n📈 Portfolio Metrics:")
        print(f"   Diversification Score: {metrics.get('diversification_score', 0):.3f}")
        print(f"   Herfindahl Index: {metrics.get('herfindahl_index', 0):.3f}")
        print(f"   Average Account Balance: ${metrics.get('average_account_balance', 0):,.2f}")
        print(f"   Largest Account Share: {metrics.get('largest_account_share', 0):.1%}")
    except Exception as e:
        print(f"❌ Error calculating portfolio metrics: {e}")
    
    # 3. Get cash flow analysis
    try:
        cash_flow = aggregator.get_cash_flow_analysis(user_id, months=6)
        print(f"\n💰 Cash Flow Analysis (6 months):")
        print(f"   Total Transactions: {cash_flow.get('total_transactions', 0)}")
        print(f"   Average Monthly Flow: ${cash_flow.get('average_monthly_flow', 0):,.2f}")
        print(f"   Flow Volatility: ${cash_flow.get('flow_volatility', 0):,.2f}")
        print(f"   Positive Flow Ratio: {cash_flow.get('positive_flow_ratio', 0):.1%}")
    except Exception as e:
        print(f"❌ Error getting cash flow analysis: {e}")
    
    # 4. Calculate risk metrics
    try:
        risk_metrics = data_handler.calculate_risk_metrics(user_id)
        print(f"\n⚠️  Risk Metrics:")
        print(f"   Institution Concentration: {risk_metrics.get('institution_concentration', 0):.3f}")
        print(f"   Type Concentration: {risk_metrics.get('type_concentration', 0):.3f}")
        print(f"   Sync Risk: {risk_metrics.get('sync_risk', 0):.1%}")
        print(f"   Number of Institutions: {risk_metrics.get('number_of_institutions', 0)}")
    except Exception as e:
        print(f"❌ Error calculating risk metrics: {e}")
    
    # 5. Get performance analysis
    try:
        performance = data_handler.get_performance_analysis(user_id, months=12)
        print(f"\n📊 Performance Analysis (12 months):")
        print(f"   Total Cash Flow: ${performance.get('total_cash_flow', 0):,.2f}")
        print(f"   Consistency Score: {performance.get('consistency_score', 0):.1%}")
        print(f"   Best Month: {performance.get('best_month', 'N/A')}")
        print(f"   Worst Month: {performance.get('worst_month', 'N/A')}")
    except Exception as e:
        print(f"❌ Error getting performance analysis: {e}")
    
    # 6. Export data to CSV
    try:
        data_handler.export_to_csv(user_id, "powens_exports")
        print(f"\n💾 Data exported to powens_exports/ directory")
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
    
    # 7. Get available institutions
    try:
        institutions = client.get_institutions()
        print(f"\n🏦 Available Institutions: {len(institutions)} institutions found")
        # Show first 5 institutions as example
        for i, inst in enumerate(institutions[:5]):
            print(f"   {i+1}. {inst.get('name', 'Unknown')} ({inst.get('country', 'Unknown')})")
    except Exception as e:
        print(f"❌ Error getting institutions: {e}")

def webview_example():
    """Example of creating a webview link for user authentication"""
    
    load_dotenv()
    
    try:
        client = PowensClient()
        
        # Example webview link creation
        user_id = "example_user_123"
        redirect_url = "https://your-app.com/callback"
        
        webview_url = client.create_webview_link(user_id, redirect_url)
        print(f"\n🔗 Webview Authentication URL:")
        print(f"   {webview_url}")
        print(f"\n   User ID: {user_id}")
        print(f"   Redirect URL: {redirect_url}")
        
    except Exception as e:
        print(f"❌ Error creating webview link: {e}")

def sync_example():
    """Example of syncing all accounts"""
    
    load_dotenv()
    
    try:
        client = PowensClient()
        aggregator = AccountAggregator(client)
        
        user_id = "example_user_123"
        
        print(f"\n🔄 Syncing accounts for user: {user_id}")
        sync_results = aggregator.sync_all_accounts(user_id)
        
        print(f"   Total Accounts: {sync_results['total_accounts']}")
        print(f"   Successful Syncs: {sync_results['successful_syncs']}")
        print(f"   Failed Syncs: {sync_results['failed_syncs']}")
        
        # Show detailed results
        for account_id, result in sync_results['results'].items():
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"   {status} {result['account_name']}: {result['status']}")
            
    except Exception as e:
        print(f"❌ Error syncing accounts: {e}")

if __name__ == "__main__":
    print("🚀 Powens Integration Example")
    print("=" * 50)
    
    # Run main analysis
    main()
    
    # Run webview example
    webview_example()
    
    # Run sync example
    sync_example()
    
    print(f"\n✅ Example completed!")
    print("\n📚 Next steps:")
    print("   1. Set up your POWENS_API_KEY environment variable")
    print("   2. Replace 'example_user_123' with actual user IDs")
    print("   3. Customize the analysis parameters for your needs")
    print("   4. Integrate with your existing quant finance tools") 