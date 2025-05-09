import os
from dotenv import load_dotenv
import json
import time
from polygon import RESTClient

load_dotenv()

client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))

ticker = ["AAPL", "MSFT"]

for t in ticker:
	financials = []
	count = 1
	for f in client.vx.list_stock_financials(ticker=t, timeframe="quarterly", order="desc", limit="1", sort="filing_date"):
		#financials.append(f)
		print("\nBalance Sheet Data:")
		print(f"Company Name: {f.company_name}")
		print(f"Filing Date: {f.fiscal_year}")
		print(f"Report Period: {f.fiscal_period}")
		print(count)
		count += 1
		time.sleep(12)

    
	# Get the first financial statement
	#first_financial = financials[0]
	
	#for fin in financials:
	#	print("\nBalance Sheet Data:")
	#	print(f"Company Name: {fin.company_name}")
	#	print(f"Filing Date: {fin.filing_date}")
	#	print(f"Report Period: {fin.fiscal_period}")
	#	print(f"Total Assets: ${fin.financials.balance_sheet.assets.value:,.2f}")
	#	print(f"Current Assets: ${fin.financials.balance_sheet.current_assets.value:,.2f}")
	#	print(f"Inventory: ${fin.financials.balance_sheet.inventory.value:,.2f}")
	#time.sleep(2)

# Access balance sheet data
#balance_sheet = first_financial.financials.balance_sheet
#print("\nBalance Sheet Data:")
#print(f"Company Name: {first_financial.company_name}")
#print(f"Filing Date: {first_financial.filing_date}")
#print(f"Report Period: {first_financial.fiscal_period}")
#
#print(f"Total Assets: ${balance_sheet.assets.value:,.2f}")
#print(f"Current Assets: ${balance_sheet.current_assets.value:,.2f}")
#print(f"Inventory: ${balance_sheet.inventory.value:,.2f}")

# You can access other financial data similarly
# For example, if you want to see all available attributes:
#print("\nAvailable attributes in the financial statement:")
#for attr in dir(first_financial):
#    if not attr.startswith('_'):  # Skip private attributes
#        print(f"- {attr}")
