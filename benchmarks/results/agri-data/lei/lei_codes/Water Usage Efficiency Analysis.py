"""
Task: Water Usage Efficiency Analysis
Description: Analyze the efficiency of water usage per kilogram of crop yield to optimize irrigation strategies.
"""

import csv
import json
import os
import math
from datetime import datetime

def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return None

def main():
    data_type = 'agri-data'
    file_paths = [
        os.path.join('data', data_type, 'raw_data.csv'),
        os.path.join('data', data_type, 'raw_data.txt')
    ]
    file_path = None
    for path in file_paths:
        if os.path.exists(path):
            file_path = path
            break
    if file_path is None:
        print(f'No data file found for {data_type}')
        return
    water_usage_efficiency_values = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        dropped_rows = 0
        for row in reader:
            try:
                water_usage = safe_float_conversion(row.get('Water_Usage_Efficiency'))
                crop_density = safe_float_conversion(row.get('Crop_Density'))
                if water_usage is not None and crop_density is not None and crop_density != 0:
                    water_usage_efficiency = water_usage / crop_density
                    water_usage_efficiency_values.append(water_usage_efficiency)
                else:
                    dropped_rows += 1
            except Exception as e:
                print(f'Error processing row: {e}')
        print(f'Dropped {dropped_rows} rows due to missing or invalid data')
    if water_usage_efficiency_values:
        summary_stats = {
            'min': min(water_usage_efficiency_values),
            'max': max(water_usage_efficiency_values),
            'mean': sum(water_usage_efficiency_values) / len(water_usage_efficiency_values)
        }
        result = {
            'task_name': 'Water Usage Efficiency Analysis',
            'description': 'Analyzed water usage efficiency per kilogram of crop yield',
            'result_summary': summary_stats,
            'result_generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(f'output/{data_type}/Water_Usage_Efficiency_Analysis_result.json', 'w') as output_file:
            json.dump(result, output_file, indent=4)
    else:
        print('No valid data to process')
if __name__ == '__main__':
    main()