# Commodity CSV Splitter

This script splits a CSV file containing commodity futures data into separate CSV files based on different commodity categories.

## Requirements

- Python 3.x
- pandas library

You can install pandas using pip:

pip install pandas

## Usage

1. Place the CSV file named `commodity_futures.csv` in the same directory as the script.

2. Run the script:

python portion_commodity_csv.py

3. The script will create separate CSV files for each commodity category in the same directory.

## Script Explanation

### Import Libraries

The script starts by importing the pandas library:

import pandas as pd

### Load the CSV File

The script reads the CSV file `commodity_futures.csv`:

file_path = 'commodity_futures.csv'
data = pd.read_csv(file_path)

### Define Categories

A dictionary containing commodity categories and their respective columns is created:

categories = {
    'Energy Commodities': ['NATURAL GAS', 'WTI CRUDE', 'BRENT CRUDE', 'LOW SULPHUR GAS OIL', 'ULS DIESEL', 'GASOLINE'],
    'Precious Metals': ['GOLD', 'SILVER'],
    'Base Metals': ['COPPER', 'ALUMINIUM', 'ZINC', 'NICKEL'],
    'Agricultural Commodities': ['SOYBEANS', 'CORN', 'SOYBEAN OIL', 'SOYBEAN MEAL', 'WHEAT', 'SUGAR', 'COFFEE', 'HRW WHEAT', 'COTTON'],
    'Livestock': ['LIVE CATTLE', 'LEAN HOGS'],
    'Date': ['DATE']
}

### Save Data by Category

A function to save the data for each category to a separate CSV file:

def save_category_to_csv(category, columns):
    columns_with_date = ['Date'] + columns if 'Date' not in columns else columns
    subset = data[columns_with_date]
    subset.to_csv(f"{category.replace(' ', '_').lower()}_commodity.csv", index=False)

### Loop Through Categories

The script loops through each category and calls the function to create the respective CSV files:

for category, columns in categories.items():
    save_category_to_csv(category, columns)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""
