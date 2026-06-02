"""
Task: Soil Fertility Assessment
Description: Calculate Soil Fertility Index (SFI) based on organic matter and NPK levels
"""

import csv
import os
import json
import math
from datetime import datetime

def calculate_sfi(row):
    try:
        organic_matter = float(row.get('Organic_Matter', 0))
        n = float(row.get('N', 0))
        p = float(row.get('P', 0))
        k = float(row.get('K', 0))
        sfi = (organic_matter * 0.2) + (n * 0.3) + (p * 0.2) + (k * 0.3)
        return sfi
    except Exception as e:
        print(f"Error calculating SFI: {e}")
        return None

def process_data(file_path):
    results = []
    dropped_rows = 0
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                sfi = calculate_sfi(row)
                if sfi is not None:
                    results.append({'Soil_Fertility_Index': sfi})
                else:
                    dropped_rows += 1
            except Exception as e:
                print(f"Error processing row: {e}")
                dropped_rows += 1
    print(f"Dropped {dropped_rows} rows")
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

def main():
    data_type = 'agri-data'
    file_path = f'data/{data_type}/raw_data.csv'
    if not os.path.exists(file_path):
        file_path = f'data/{data_type}/raw_data.txt'
    if not os.path.exists(file_path):
        print("Data file not found")
        return
    results = process_data(file_path)
    output_file = f'output/{data_type}/Soil_Fertility_Assessment_result.json'
    save_results(results, output_file)
    print("Soil Fertility Assessment completed")

if __name__ == '__main__':
    main()