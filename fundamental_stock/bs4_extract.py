from bs4 import BeautifulSoup

import json
import pandas as pd
import re
from datetime import datetime
import requests

# get website
url = "https://finance.yahoo.com/quote/AAPL/balance-sheet/"

# fetch with bs4
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')
print(f"soup : {soup}")

# Load the HTML content
#with open("fundamental_stock/html_test.txt", "r", encoding="utf-8") as file:
#    html = file.read()

# Parse HTML
#soup = BeautifulSoup(html, 'html.parser')

# Find the "Breakdown" div
table_div = soup.find("div", class_="table yf-9ft13")

button_quarterly = soup.find("button", class_="tabBtn l1 yf-gfq5ju", id_="tab-quarterly")
#button_quarter.click()
if button_quarterly:
    content_quarterly = button_quarterly.get_text()
    print(f"content_quarterly : {content_quarterly}")
else:
    print("Quarterly button not found.")

button_annual = soup.find("button", class_="tabBtn l1 yf-gfq5ju", id_="tab-annual")
#button_annual.click()
if button_annual:
    content_annual = button_annual.get_text()
    print(f"content_annual : {content_annual}")
else:
    print("Annual button not found.")

# Get the parent or next sibling, depending on structure
if table_div:
    print(f"type(table_div) : {type(table_div)}")
    # get everything in the div which is a table
    next_content = table_div.get_text()
    
    # Split the content into lines and clean up
    first_line = next_content.split('      ')
    # parsing first line
    dates = first_line[0].split(' ')
    # convert to datetime if possible else keep as str
    def parse_date_or_keep(item):
        try:
            return datetime.strptime(item, "%m/%d/%Y").date()
        except ValueError:
            return item

    parsed_header = [parse_date_or_keep(i) for i in dates]
    print(f"dates : {parsed_header}")
    
    #parsed_header = []
    #for date in dates:
    #    try:
    #        date.append(datetime.strptime(date, '%m/%d/%Y').date())
    #    except ValueError:
    #        date.append(date)
    #print(f"parsed_header : {parsed_header}")
    
    data_lines = first_line[1].split('    ')
    # parsing data lines
    parsed_rows = []

    # match a label (text) followed by 4 numbers or placeholders
    pattern = re.compile(r'([A-Za-z \-]+?)\s+([\d\-,]+|--) ([\d\-,]+|--) ([\d\-,]+|--) ([\d\-,]+|--)')

    def clean_number(val):
        if val == '--':
            return None
        return int(val.replace(',', ''))

    for line in data_lines:
        matches = pattern.findall(line)
        for match in matches:
            label = match[0].strip()
            values = [clean_number(v.strip()) for v in match[1:]]
            parsed_rows.append([label] + values)

    print(parsed_rows)
    
    table_data = [parsed_header, *parsed_rows]
    #table_data.append(parsed_header)
    #table_data.append(parsed_rows)
    print(f"table_data : {table_data}")

    # extract to json
    with open('fundamental_stock/table_data.json', 'w') as f:
        json.dump(table_data, f)

    # create a dataframe with header
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    print(f"df : {df}")
else:
    print("Breakdown section not found.")