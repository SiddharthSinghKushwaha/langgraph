"""
Task: Crop Growth Stage Classification
Description: Classify crops into different growth stages (Seedling, Vegetative, Flowering) based on sensor data.
"""

import csv
import json
import os
import pathlib
import sys

def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return None

def classify_growth_stage(row):
    # Simple classification logic based on Growth_Stage
    growth_stage = row.get('Growth_Stage')
    if growth_stage == '':
        return None
    elif growth_stage == '2':
        return 'Vegetative'
    elif growth_stage == '3':
        return 'Flowering'
    else:
        return 'Seedling'

def main():
    data_type = 'agri-data'
    file_path_csv = pathlib.Path('data', data_type, 'raw_data.csv')
    file_path_txt = pathlib.Path('data', data_type, 'raw_data.txt')
    file_path = file_path_csv if file_path_csv.exists() else file_path_txt
    if not file_path.exists():
        print(f"Neither 'data/{data_type}/raw_data.csv' nor 'data/{data_type}/raw_data.txt' exists.")
        sys.exit(1)

    results = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            growth_stage = classify_growth_stage(row)
            if growth_stage is not None:
                results.append({
                    'Growth_Stage': growth_stage
                })

    output_path = pathlib.Path('output', data_type, 'Crop_Growth_Stage_Classification_result.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "task_name": "Crop Growth Stage Classification",
            "description": "Classify crops into different growth stages (Seedling, Vegetative, Flowering) based on sensor data.",
            "result_summary": results,
            "result_generated_at": "2023-12-01"
        }, f)
if __name__ == '__main__':
    main()