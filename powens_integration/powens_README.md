# Powens Integration for Quant Finance

This module provides integration with [Powens](https://docs.powens.com/console-webview), a financial data aggregation platform that offers account aggregation, financial data APIs, and webview components for user authentication.

## Features

- **Account Aggregation**: Connect and manage multiple financial accounts
- **Portfolio Analysis**: Comprehensive portfolio metrics and risk analysis
- **Transaction History**: Detailed transaction data across all accounts
- **Risk Metrics**: Diversification analysis and concentration risk assessment
- **Webview Integration**: Secure user authentication flow
- **Data Export**: Export portfolio data to CSV, JSON, and Excel formats
- **Real-time Sync**: Synchronize account data with financial institutions

## Installation

### 1. Environment Setup

Add the required dependencies to your environment:

```bash
# Add to environment.yml
- requests

# Or install via pip
pip install requests python-dotenv
```

### 2. Environment Variables

Create a `.env` file in your project root:

```env
# Required
POWENS_API_KEY=your_powens_api_key_here

# Optional - Customize as needed
POWENS_API_BASE_URL=https://api.powens.com
POWENS_WEBVIEW_BASE_URL=https://webview.powens.com
POWENS_REDIRECT_URL=http://localhost:3000/callback
POWENS_CACHE_DURATION=3600
POWENS_MAX_RETRIES=3
POWENS_REQUEST_TIMEOUT=30
POWENS_DEFAULT_ANALYSIS_MONTHS=6
POWENS_MAX_TRANSACTION_HISTORY=90
POWENS_SYNC_RISK_THRESHOLD=168
POWENS_CONCENTRATION_WARNING_THRESHOLD=0.5
POWENS_EXPORT_DIR=powens_data
```

### 3. Get Powens API Key

1. Sign up for a Powens account at [https://console.powens.com](https://console.powens.com)
2. Navigate to the API section in your console
3. Generate an API key for your application
4. Add the API key to your `.env` file

## Quick Start

### Basic Usage

```python
from powens_integration import PowensClient, AccountAggregator, PowensDataHandler

# Initialize client
client = PowensClient()

# Initialize data handler
data_handler = PowensDataHandler(client)

# Get portfolio data
user_id = "your_user_id"
portfolio_data = data_handler.get_portfolio_data(user_id)

print(f"Total Balance: ${portfolio_data['summary']['total_balance']:,.2f}")
print(f"Number of Accounts: {portfolio_data['summary']['total_accounts']}")
```

### Account Aggregation

```python
from powens_integration import AccountAggregator

aggregator = AccountAggregator(client)

# Get all accounts
accounts_df = aggregator.get_all_accounts(user_id)

# Get portfolio summary
summary = aggregator.get_portfolio_summary(user_id)

# Calculate portfolio metrics
metrics = aggregator.calculate_portfolio_metrics(user_id)

# Sync all accounts
sync_results = aggregator.sync_all_accounts(user_id)
```

### Risk Analysis

```python
# Calculate risk metrics
risk_metrics = data_handler.calculate_risk_metrics(user_id)

print(f"Diversification Score: {risk_metrics['diversification_score']:.1%}")
print(f"Institution Concentration: {risk_metrics['institution_concentration']:.1%}")
print(f"Sync Risk: {risk_metrics['sync_risk']:.1%}")
```

### Webview Authentication

```python
# Create webview link for user authentication
webview_url = client.create_webview_link(
    user_id="user_123",
    redirect_url="https://your-app.com/callback"
)

print(f"Authentication URL: {webview_url}")
```

## Integration with Existing Dashboard

### React/Next.js Integration

The module includes a React component for integration with your existing dashboard:

```tsx
import { PowensIntegration } from "@/components/powens-integration"

export default function Dashboard() {
  return (
    <div>
      <h1>Financial Dashboard</h1>
      <PowensIntegration />
    </div>
  )
}
```

### API Endpoints

Create API endpoints to handle Powens data:

```python
# Example FastAPI endpoint
from fastapi import FastAPI, HTTPException
from powens_integration import PowensDataHandler

app = FastAPI()
data_handler = PowensDataHandler(PowensClient())

@app.get("/api/portfolio/{user_id}")
async def get_portfolio(user_id: str):
    try:
        return data_handler.get_portfolio_data(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync/{user_id}")
async def sync_accounts(user_id: str):
    try:
        aggregator = AccountAggregator(PowensClient())
        return aggregator.sync_all_accounts(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Data Structure

### Account Data

```python
{
    "id": "account_123",
    "name": "Main Checking",
    "type": "checking",
    "balance": 15420.50,
    "available_balance": 15420.50,
    "institution_name": "Chase Bank",
    "last_sync_date": "2024-01-15T10:30:00Z"
}
```

### Portfolio Summary

```python
{
    "total_balance": 186081.50,
    "total_accounts": 3,
    "account_types": {"checking": 1, "savings": 1, "investment": 1},
    "institutions": {"Chase Bank": 2, "Fidelity": 1},
    "last_sync": "2024-01-15T10:30:00Z"
}
```

### Risk Metrics

```python
{
    "diversification_score": 0.67,
    "institution_concentration": 0.56,
    "type_concentration": 0.33,
    "sync_risk": 0.0,
    "number_of_institutions": 2,
    "number_of_account_types": 3
}
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POWENS_API_KEY` | Your Powens API key | - | Yes |
| `POWENS_API_BASE_URL` | Powens API base URL | `https://api.powens.com` | No |
| `POWENS_WEBVIEW_BASE_URL` | Webview base URL | `https://webview.powens.com` | No |
| `POWENS_REDIRECT_URL` | Redirect URL after auth | `http://localhost:3000/callback` | No |
| `POWENS_CACHE_DURATION` | Cache duration in seconds | `3600` | No |
| `POWENS_MAX_RETRIES` | Maximum API retries | `3` | No |
| `POWENS_REQUEST_TIMEOUT` | Request timeout in seconds | `30` | No |

### Configuration Validation

```python
from powens_integration.config import PowensConfig

# Validate configuration
validation = PowensConfig.validate_config()

if not validation["valid"]:
    print("Configuration errors:", validation["errors"])
    exit(1)

if validation["warnings"]:
    print("Configuration warnings:", validation["warnings"])
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    client = PowensClient()
    data = client.get_accounts(user_id)
except ValueError as e:
    print(f"Configuration error: {e}")
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

### Complete Portfolio Analysis

```python
from powens_integration import PowensDataHandler
from datetime import datetime

# Initialize
data_handler = PowensDataHandler(PowensClient())
user_id = "user_123"

# Get comprehensive portfolio data
portfolio_data = data_handler.get_portfolio_data(
    user_id=user_id,
    include_transactions=True,
    include_balances=True
)

# Calculate risk metrics
risk_metrics = data_handler.calculate_risk_metrics(user_id)

# Get performance analysis
performance = data_handler.get_performance_analysis(user_id, months=12)

# Export data
data_handler.export_to_csv(user_id, "portfolio_analysis")

print("Portfolio analysis completed!")
```

### Cash Flow Analysis

```python
# Get cash flow data
cash_flow_df = data_handler.get_cash_flow_dataframe(user_id, months=6)

# Analyze spending patterns
monthly_spending = cash_flow_df[cash_flow_df['amount'] < 0].groupby('month')['amount'].sum()
print("Monthly spending:", monthly_spending)

# Analyze income patterns
monthly_income = cash_flow_df[cash_flow_df['amount'] > 0].groupby('month')['amount'].sum()
print("Monthly income:", monthly_income)
```

## Security Considerations

1. **API Key Security**: Never commit your API key to version control
2. **Environment Variables**: Use environment variables for sensitive configuration
3. **HTTPS**: Always use HTTPS in production for webview redirects
4. **Data Privacy**: Ensure compliance with financial data privacy regulations
5. **Rate Limiting**: Respect Powens API rate limits

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `POWENS_API_KEY` is set in your environment
2. **Network Errors**: Check your internet connection and firewall settings
3. **Rate Limiting**: Implement exponential backoff for API retries
4. **Data Sync Issues**: Check account sync status and retry if needed

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

- [Powens Documentation](https://docs.powens.com/console-webview)
- [Powens Console](https://console.powens.com)
- [API Reference](https://docs.powens.com/api)

## License

This integration module is part of the Quant Finance project and follows the same license terms. 