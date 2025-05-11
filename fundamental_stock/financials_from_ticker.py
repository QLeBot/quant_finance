from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import json
from datetime import datetime
import pandas as pd

# Set up headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument(f'user-agent={headers["User-Agent"]}')

# Use webdriver-manager to handle driver installation
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

def table_parsing(content):
    data = []
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    #print(f"lines : {lines}")

    #print(f"line 0 : {lines[0]}")
    # First line is the header
    header = lines[0].split()
    # when possible, convert to datetime
    for i, item in enumerate(header):
        try:
            #header[i] = datetime.strptime(item, "%m/%d/%Y").date()
            #header[i] = pd.to_datetime(item).quarter
            header[i] = pd.to_datetime(item)
        except ValueError:
            header[i] = item
    data.append(header)

    # The rest follows the pattern: 1 title line + data lines (length determined by header length)
    i = 1
    while i < len(lines):
        title = lines[i]
        values = []
        for value in lines[i + 1:i + len(header)]:
            try:
                value = float(value.replace(",", ""))
            except ValueError:
                value = None
            values.append(value)
        row = [title] + values
        data.append(row)
        i += len(header)

    return data

def get_financials(driver):
    #driver.get(url)
    time.sleep(2)  # Add a small delay to let the page load
    #print(driver.page_source)
    print("page loaded")

    # bypass cookie consent
    try:
        button_cookie = driver.find_element(By.XPATH, '//button[@class="btn secondary reject-all"]')
        button_cookie.click()
        time.sleep(2)
        print("cookie consent dismissed")
    except:
        print("no cookie consent")

    # default to annual
    #button_annual = driver.find_element(By.XPATH, '//button[@id="tab-annual"]')
    table = driver.find_element(By.XPATH, '//div[@class="table yf-9ft13"]')
    # get content from table
    content = table.text
    data_annual  = table_parsing(content)
    
    # switch to quarterly
    button_quarterly = driver.find_element(By.XPATH, '//button[@id="tab-quarterly"]')
    button_quarterly.click()
    time.sleep(2)
    print("switched to quarterly")
    table = driver.find_element(By.XPATH, '//div[@class="table yf-9ft13"]')
    # get content from table
    content = table.text
    data_quarterly = table_parsing(content)

    # convert to dataframe with data[0] as headers
    df_annual = pd.DataFrame(data_annual[1:], columns=data_annual[0])
    df_quarterly = pd.DataFrame(data_quarterly[1:], columns=data_quarterly[0])
    #print(f"df_annual \n: {df_annual.head()}")
    #print(f"df_quarterly \n: {df_quarterly.head()}")
    return df_annual, df_quarterly

def get_financials_from_ticker(ticker):
    url = "https://finance.yahoo.com/quote/{ticker}/financials/"
    url = url.format(ticker=ticker)
    print(f"url : {url}")
    # get type of financial statement
    financial_type = url.split("/")[-2]
    print(f"financial_type : {financial_type}")

    df_annual_financials, df_quarterly_financials = get_financials(driver, url)
    #print(f"df_annual \n: {df_annual.head()}")
    #print(f"df_quarterly \n: {df_quarterly.head()}")

    # button_financials = driver.find_element(By.XPATH, '//a[@href="/quote/AAPL/financials/"]')
    button_balance_sheet = driver.find_element(By.XPATH, '//a[@href="/quote/AAPL/balance-sheet/"]')
    button_balance_sheet.click()
    time.sleep(2)
    print("clicked on button_balance_sheet")
    df_annual_balance_sheet, df_quarterly_balance_sheet = get_financials(driver)

    button_cash_flow = driver.find_element(By.XPATH, '//a[@href="/quote/AAPL/cash-flow/"]')
    button_cash_flow.click()
    time.sleep(2)
    print("clicked on button_cash_flow")
    df_annual_cash_flow, df_quarterly_cash_flow = get_financials(driver)
    
    # save to csv
    df_annual_financials.to_csv(f"fundamental_stock/scraper_csv/{ticker}_financials_annual.csv", index=False)
    df_quarterly_financials.to_csv(f"fundamental_stock/scraper_csv/{ticker}_financials_quarterly.csv", index=False)
    df_annual_balance_sheet.to_csv(f"fundamental_stock/scraper_csv/{ticker}_balance_sheet_annual.csv", index=False)
    df_quarterly_balance_sheet.to_csv(f"fundamental_stock/scraper_csv/{ticker}_balance_sheet_quarterly.csv", index=False)
    df_annual_cash_flow.to_csv(f"fundamental_stock/scraper_csv/{ticker}_cash_flow_annual.csv", index=False)
    df_quarterly_cash_flow.to_csv(f"fundamental_stock/scraper_csv/{ticker}_cash_flow_quarterly.csv", index=False)

try :
    get_financials_from_ticker("AAPL")
except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    if 'driver' in locals():
        driver.quit()