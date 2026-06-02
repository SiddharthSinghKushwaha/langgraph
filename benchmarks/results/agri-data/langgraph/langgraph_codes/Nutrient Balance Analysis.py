"""
Task: Nutrient Balance Analysis
Description: Calculate Nutrient Balance Ratio (NBR) for soil nutrient assessment
"""

import csv
import os
import json
import math

def calculate_nbr(row):
    try:
        n = float(row.get('N', '')) if row.get('N', '') else None
        p = float(row.get('P', '')) if row.get('P', '') else None
        k = float(row.get('K', '')) if row.get('K', '') else None
        if n is not None and p is not None and k is not None:
            nbr = (n + p + k) / 3
            return nbr
        else:
            return None
    except Exception as e:
        print(f"Error calculating NBR: {e}")
        return None

def process_data(file_path):
    nbr_values = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            nbr = calculate_nbr(row)
            if nbr is not None:
                nbr_values.append(nbr)
    return nbr_values

def main():
    data_type = 'agri-data'
    file_paths = [
        os.path.join('data', data_type, 'raw_data.csv'),
        os.path.join('data', data_type, 'raw_data.txt')
    ]
    file_path = next((path for path in file_paths if os.path.exists(path)), None)
    if file_path:
        nbr_values = process_data(file_path)
        result_summary = ["Nutrient Balance Ratio (NBR) values:"]
        result_summary.extend([str(nbr) for nbr in nbr_values])
        output = {
            "task_name": "Nutrient Balance Analysis",
            "description": "Calculate Nutrient Balance Ratio (NBR) for soil nutrient assessment",
            "result_summary": result_summary,
            "result_generated_at": "2023-12-01"
        }
        with open(os.path.join('output', data_type, 'Nutrient Balance Analysis_result.json'), 'w') as f:
            json.dump(output, f)
    else:
        print("No data file found.")

if __name__ == '__main__':
    main()