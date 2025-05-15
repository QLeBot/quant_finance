from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import os
import json
from datetime import datetime
import pandas as pd

# Set up headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36',
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
driver.maximize_window()

actions = ActionChains(driver)

# default US when loading the page
regions = [
    #{"code": "us",
    #"name": "United States"
    #},
    #{"code": "fr",
    #"name": "France"
    #},
    #{"code": "ar",
    #"name": "Argentina"
    #},
    #{"code": "at",
    #"name": "Austria"
    #},
    #{"code": "au",
    #"name": "Australia"
    #},
    # problem with Belgium
    {"code": "be",
    "name": "Belgium"
    },
    #{"code": "br",
    #"name": "Brazil"
    #},
    # problem with Belgium
    #{"code": "ca",
    #"name": "Canada"
    #},
    #{"code": "ch",
    #"name": "Switzerland"
    #},
    #{"code": "cl",
    #"name": "Chile"
    #},
    #{"code": "cn",
    #"name": "China"
    #},
    #{"code": "cz",
    #"name": "Czechia"
    #},
    #{"code": "de",
    #"name": "Germany"
    #},
    #{"code": "dk",
    #"name": "Denmark"
    #},
    #{"code": "ee",
    #"name": "Estonia"
    #},
    #{"code": "gb",
    #"name": "United Kingdom"
    #},
    #{"code": "gr",
    #"name": "Greece"
    #},
    #{"code": "hk",
    #"name": "Hong Kong SAR China"
    #},
    #{"code": "hu",
    #"name": "Hungary"
    #},
    #{"code": "id",
    #"name": "Indonesia"
    #},
    #{"code": "is",
    #"name": "Iceland"
    #},
    #{"code": "it",
    #"name": "Italy"
    #},
    #{"code": "jp",
    #"name": "Japan"
    #},
    #{"code": "kr",
    #"name": "South Korea"
    #},
    #{"code": "kw",
    #"name": "Kuwait"
    #},
    #{"code": "lk",
    #"name": "Sri Lanka"
    #},
    #{"code": "lt",
    #"name": "Lithuania"
    #},
    #{"code": "lv",
    #"name": "Latvia"
    #},
    #{"code": "mx",
    #"name": "Mexico"
    #},
    #{"code": "my",
    #"name": "Malaysia"
    #},
    #{"code": "nl",
    #"name": "Netherlands"
    #},
    #{"code": "ph",
    #"name": "Philippines"
    #},
    #{"code": "pk",
    #"name": "Pakistan"
    #},
    #{"code": "pl",
    #"name": "Poland"
    #},
    #{"code": "pt",
    #"name": "Portugal"
    #},
    #{"code": "za",
    #"name": "South Africa"
    #},
    #{"code": "sr",
    #"name": "Suriname"
    #},
    #{"code": "th",
    #"name": "Thailand"
    #},
    #{"code": "tr",
    #"name": "Turkey"
    #},
    #{"code": "tw",
    #"name": "Taiwan"
    #},
    #{"code": "ve",
    #"name": "Venezuela"
    #}
]

# get all region codes as a list
#region_codes = [region["code"] for region in regions]
#print(f"region_codes : {region_codes}")

def change_market_cap_filter(driver):
    # Market Cap
    #elmt:menu;itc:1;sec:screener-filter;subsec:custom-screener;elm:expand;slk:Market%20Cap%20(Intraday)
    #link2-btn fin-size-small menuBtn hover:tw-bg-[var(--table-hover-emph)] rightAlign yf-1cfb8vd
    #WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="link2-btn fin-size-small menuBtn hover:tw-bg-[var(--table-hover-emph)] rightAlign yf-1cfb8vd"]'))).click()
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@data-ylk="elmt:menu;itc:1;sec:screener-filter;subsec:custom-screener;elm:expand;slk:Market%20Cap%20(Intraday)"]'))).click()
    #driver.find_element(By.XPATH, '//button[@class="link2-btn fin-size-small menuBtn hover:tw-bg-[var(--table-hover-emph)] rightAlign yf-1cfb8vd"]').click()
    print("market cap click success")

    # Custom CheckBox
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//input[@id="custom"]'))).click()
    #driver.find_element(By.XPATH, '//input[@id="custom"]').click()
    print("custom click success")
   #time.sleep(1200)

    #<button class="rounded yf-1cfb8vd" data-ylk="slk:Between;sec:screener-filter;subsec:custom-screener-menu;elm:operator;elmt:screener-filter;itc:1" data-rapid_p="1069" data-v9y="1"> Between</button>
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@data-ylk="slk:Between;sec:screener-filter;subsec:custom-screener-menu;elm:operator;elmt:screener-filter;itc:1"]'))).click()
    # Between operator
    #WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="rounded yf-1cfb8vd"]'))).click()
    #driver.find_element(By.XPATH, '//input[@id="custom"]').click()
    #time.sleep(2)

    print("between click success")

    time.sleep(10)

    # find both inputs
    inputs = driver.find_elements(By.XPATH, '//div[@class="left-content yf-i4lnim"]/input[@class="yf-i4lnim"]')
    left_input = inputs[0]
    right_input = inputs[1]
    time.sleep(2)

    # send keys to both inputs
    left_input.send_keys("300M")
    time.sleep(2)
    right_input.send_keys("2B")
    time.sleep(2)

    # Apply Button
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="primary-btn fin-size-small rounded yf-1cfb8vd"]'))).click()
    time.sleep(5)

def change_region_filter(driver, region, previous_region):
    time.sleep(10)
    # Open filter
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@data-ylk="elmt:menu;itc:1;sec:screener-filter;subsec:custom-screener;elm:expand;slk:Region"]'))).click()
    time.sleep(5)

    element = driver.find_element(By.XPATH, f'//input[@id="{previous_region}"]')
    actions = ActionChains(driver)
    actions.move_to_element(element).click().perform()
    print("success deselecting previous region")
    time.sleep(5)

    element = driver.find_element(By.XPATH, f'//input[@id="{region["code"]}"]')
    actions = ActionChains(driver)
    actions.move_to_element(element).click().perform()

    # Deselect previous region
    #WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, f'//input[@id="{previous_region}"]'))).click()
    #print("success deselecting previous region")
    #time.sleep(5)
    # Select new region
    #WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, f'//input[@id="{region["code"]}"]'))).click()
    print("success selecting new region")
    time.sleep(5)
    # Apply filter
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@data-ylk="sec:screener-filter;subsec:custom-screener;elm:intent-edit;elmt:screener-filter;slk:Region;itc:1"]'))).click()
        
    print(f"region {region["code"]} click success")
    

def scrape_all_tickers(driver):
    # save all tickers in a list
    tickers = []
    # loop until no more next page button
    while True:
        try: # see if next page button is present, scrape all tickers and then click next page button
            # get all tickers
            time.sleep(10)
            # get all tickers
            all_tickers = driver.find_elements(By.XPATH, '//a[@class="ticker x-small hover logo stacked yf-5ogvqh"]')
            for ticker in all_tickers:
                href = ticker.get_attribute('href')
                tickers.append(href)
            time.sleep(10)
            print(f"len of tickers : {len(tickers)}")
            # click next page button
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@data-ylk="sec:screener-table;subsec:custom-screener;elm:arrow;itc:1;slk:next"]'))).click()
        except:
            break
        print("page changed")
    return tickers

try :
    url = "https://finance.yahoo.com/research-hub/screener/equity/?start=0&count=100"
    print(f"url : {url}")
    driver.get(url)

    # decline all cookies if present
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="btn secondary reject-all"]'))).click()
    except:
        pass

    # arrive on the page us is selected
    time.sleep(10)
    
    # add market cap filter to small caps only have to set up once
    change_market_cap_filter(driver)

    # loop over page and scrape all tickers
    #scrape_all_tickers(driver)

    # store previous region
    previous_region = "us"
    # loop over possible regions
    for region in regions:
        print(f"previous_region : {previous_region}")
        print(f"region : {region['code']}")
        if region["code"] == previous_region:
            #tickers = scrape_all_tickers(driver)
            tickers = []
        else:
            time.sleep(10)
            driver.execute_script("window.scrollTo(0, 0);")
            actions.send_keys(Keys.HOME).perform()
            # change region filter
            change_region_filter(driver, region, previous_region)
            time.sleep(20)
            # scrape all tickers
            tickers = scrape_all_tickers(driver)
            previous_region = region["code"]
    
        # save tickers to a file
        print(f"length of tickers : {len(tickers)}")
        with open(f'fundamental_stock/small_caps_tickers_{region["code"]}.txt', 'w') as f:
            for ticker in tickers:
                f.write(ticker + '\n')

    # All tickers
    #all_tickers = driver.find_elements(By.XPATH, '//a[@class="ticker x-small hover logo stacked yf-5ogvqh"]')
    #for ticker in all_tickers:
    #    href = ticker.get_attribute('href')
    #    print(href)
    
    
    # next page button
    # <button class="icon-btn fin-size-medium rounded yf-1cfb8vd" data-ylk="sec:screener-table;subsec:custom-screener;elm:arrow;itc:1;slk:next" data-testid="next-page-button" aria-label="Goto next page" data-rapid_p="334" data-v9y="1"><div aria-hidden="true" class="icon fin-icon inherit-icn sz-medium yf-9qlxtu"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z"></path></svg></div> </button>
    #WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@data-ylk="sec:screener-table;subsec:custom-screener;elm:arrow;itc:1;slk:next"]'))).click()

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    if 'driver' in locals():
        driver.quit()