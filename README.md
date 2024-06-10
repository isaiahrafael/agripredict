python
Always show details

Copy code
# Combine the script and the explanation into one README.md file content

readme_content = """
# Commodity CSV Splitter

This script splits a CSV file containing commodity futures data into separate CSV files based on different commodity categories.

## Requirements

- Python 3.x
- pandas library

You can install pandas using pip:

```sh
pip install pandas
Usage
Place the CSV file named commodity_futures.csv in the same directory as the script.

Run the script:

sh
Always show details

Copy code
python portion_commodity_csv.py
The script will create separate CSV files for each commodity category in the same directory.
Script Explanation
Import Libraries
The script starts by importing the pandas library:

python
Always show details

Copy code
import pandas as pd
Load the CSV File
The script reads the CSV file commodity_futures.csv:

python
Always show details

Copy code
file_path = 'commodity_futures.csv'
data = pd.read_csv(file_path)
Define Categories
A dictionary containing commodity categories and their respective columns is created:

python
Always show details

Copy code
categories = {
    'Energy Commodities': ['NATURAL GAS', 'WTI CRUDE', 'BRENT CRUDE', 'LOW SULPHUR GAS OIL', 'ULS DIESEL', 'GASOLINE'],
    'Precious Metals': ['GOLD', 'SILVER'],
    'Base Metals': ['COPPER', 'ALUMINIUM', 'ZINC', 'NICKEL'],
    'Agricultural Commodities': ['SOYBEANS', 'CORN', 'SOYBEAN OIL', 'SOYBEAN MEAL', 'WHEAT', 'SUGAR', 'COFFEE', 'HRW WHEAT', 'COTTON'],
    'Livestock': ['LIVE CATTLE', 'LEAN HOGS'],
    'Date': ['DATE']
}
Save Data by Category
A function to save the data for each category to a separate CSV file:

python
Always show details

Copy code
def save_category_to_csv(category, columns):
    columns_with_date = ['Date'] + columns if 'Date' not in columns else columns
    subset = data[columns_with_date]
    subset.to_csv(f"{category.replace(' ', '_').lower()}_commodity.csv", index=False)
Loop Through Categories
The script loops through each category and calls the function to create the respective CSV files:

python
Always show details

Copy code
for category, columns in categories.items():
    save_category_to_csv(category, columns)
