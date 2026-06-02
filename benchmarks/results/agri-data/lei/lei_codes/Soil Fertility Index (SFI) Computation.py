"""
Task: Soil Fertility Index (SFI) Computation
Description: Calculate the Soil Fertility Index to assess composite soil fertility based on organic matter and NPK levels.
"""

import csv
import json
import os
import math

def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return None

def calculate_soil_fertility_index(row):
    n = safe_float_conversion(row.get('N'))
    p = safe_float_conversion(row.get('P'))
    k = safe_float_conversion(row.get('K'))
    organic_matter = safe_float_conversion(row.get('Organic_Matter'))
    if n is not None and p is not None and k is not None and organic_matter is not None:
        sfi = (n + p + k) / 3 * organic_matter / 100
        return sfi
    return None

def main():
    data_type = 'agri-data'
    filename = 'raw_data.csv'
    filepath = os.path.join('data', data_type, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join('data', data_type, 'raw_data.txt')
    if not os.path.exists(filepath):
        print(f"Error: File not found in {filepath}")
        return
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        sfi_values = []
        for row in reader:
            sfi = calculate_soil_fertility_index(row)
            if sfi is not None:
                sfi_values.append(sfi)
    result = {
        "task_name": "Soil Fertility Index (SFI) Computation",
        "description": "Soil Fertility Index values calculated",
        "result_summary": ["Min SFI: " + str(min(sfi_values)) if sfi_values else "No SFI values calculated",
                          "Max SFI: " + str(max(sfi_values)) if sfi_values else "",
                          "Mean SFI: " + str(sum(sfi_values) / len(sfi_values)) if sfi_values else ""]},
        "result_generated_at": "2023-12-01"
    }
    output_filename = 'Soil Fertility Index (SFI) Computation_result.json'
    with open(os.path.join('output', data_type, output_filename), 'w') as outfile:
        json.dump(result, outfile)
if __name__ == '__main__':
    main()