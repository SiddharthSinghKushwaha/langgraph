"""
Task: Photosynthesis Potential (PP) Estimation
Description: Estimate the Photosynthesis Potential based on sunlight exposure, CO2 concentration, and temperature.
"""

import csv
import os
import json
import math
from datetime import datetime

def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return None

def calculate_photosynthesis_potential(sunlight_exposure, co2_concentration, temperature):
    if sunlight_exposure is None or co2_concentration is None or temperature is None:
        return None
    pp = 0.5 * sunlight_exposure * co2_concentration * (temperature / 1000)
    return pp

def process_data(file_path):
    pp_values = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sunlight_exposure = safe_float_conversion(row.get('Sunlight_Exposure'))
            co2_concentration = safe_float_conversion(row.get('CO2_Concentration'))
            temperature = safe_float_conversion(row.get('Temperature'))
            pp = calculate_photosynthesis_potential(sunlight_exposure, co2_concentration, temperature)
            pp_values.append(pp)
    return pp_values

def main():
    data_type = 'agri-data'
    file_paths = [
        os.path.join('data', data_type, 'raw_data.csv'),
        os.path.join('data', data_type, 'raw_data.txt')
    ]
    file_path = next((path for path in file_paths if os.path.exists(path)), None)
    if file_path is None:
        print(f"Error: No data file found for {data_type}.")
        return
    pp_values = process_data(file_path)
    result = {
        "task_name": "Photosynthesis Potential (PP) Estimation",
        "description": "Estimate the Photosynthesis Potential based on sunlight exposure, CO2 concentration, and temperature.",
        "result_summary": pp_values,
        "result_generated_at": datetime.now().isoformat()
    }
    with open(os.path.join('output', data_type, 'Photosynthesis Potential (PP) Estimation_result.json'), 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    main()