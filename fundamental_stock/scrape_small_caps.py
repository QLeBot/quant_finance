from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
#chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument(f'user-agent={headers["User-Agent"]}')

# Use webdriver-manager to handle driver installation
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

try :
    url = "https://finance.yahoo.com/research-hub/screener/equity/?start=0&count=100"
    print(f"url : {url}")
    driver.get(url)

    # Refuser tout
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="btn secondary reject-all"]'))).click()

    # Remove Filter
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="icon-btn fin-size-medium tw-border-l-[var(--separator)] hover:tw-border-l-[var(--color-interactive-level-2-hovered)] yf-1cfb8vd"]'))).click()
    #<button class="icon-btn fin-size-medium tw-border-l-[var(--separator)] hover:tw-border-l-[var(--color-interactive-level-2-hovered)] yf-1cfb8vd" data-ylk="sec:screener-filter;subsec:custom-screener;elm:intent-delete;elmt:screener-filter;slk:Region;itc:1" title="Remove Filter" data-rapid_p="20" data-v9y="1"><div aria-hidden="true" class="icon fin-icon inherit-icn sz-medium yf-9qlxtu"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path></svg></div> </button>
    
    # Market Cap
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="link2-btn fin-size-small menuBtn hover:tw-bg-[var(--table-hover-emph)] rightAlign yf-1cfb8vd"]'))).click()
    #driver.find_element(By.XPATH, '//button[@class="link2-btn fin-size-small menuBtn hover:tw-bg-[var(--table-hover-emph)] rightAlign yf-1cfb8vd"]').click()

    # Custom CheckBox
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//input[@id="custom"]'))).click()
    #driver.find_element(By.XPATH, '//input[@id="custom"]').click()

   #time.sleep(1200)

    # Between operator
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/main/section/section/section/article/section/div/div[3]/div/div[3]/div/div/div/section/div[1]/div[2]/div[1]/div[4]/button[@class="rounded yf-1cfb8vd"]'))).click()
    #driver.find_element(By.XPATH, '//input[@id="custom"]').click()
    time.sleep(2)

    print("between click success")

    # Market Cap Min
    driver.find_element(By.XPATH, '//*[@id="-63012"][@class="yf-i4lnim"]').send_keys("300M")
    time.sleep(2)

    # Market Cap Max
    driver.find_element(By.XPATH, '//*[@id="-399078"][@class="yf-i4lnim"]').send_keys("2B")
    time.sleep(2)

    # Apply Button
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="primary-btn fin-size-small rounded yf-1cfb8vd"]'))).click()
    time.sleep(2)
    
    # Get first tickers
    first_ticker = driver.find_element(By.XPATH, '//a[@class="ticker x-small hover logo stacked yf-5ogvqh"]')
    # get href from first_ticker
    href = first_ticker.get_attribute('href')
    print(f"href : {href}")
    
    


except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    if 'driver' in locals():
        driver.quit()