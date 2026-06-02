"""
Task: Crop Health Classification
Description: Match NPK, temperature, and soil moisture levels to predict crop health
"""

import csv
import json
import os
import math
from datetime import datetime

def safe_float conversion(val):
 try: return float(val)
 except ValueError: return None

def load_data(data_type):
 file_path = f'data/{data_type}/raw_data.csv'
 if not os.path.exists(file_path):
 file_path = f'data/{data_type}/raw_data.txt'
 if not os.path.exists(file_path):
 print(f'Error: Neither CSV nor TXT file found for {data_type}')
 return []

 with open(file_path, 'r') as file:
 reader = csv.DictReader(file)
 data = []
 for row in reader:
 # Clean and convert data
 clean_row = {}
 for key, val in row.items():
 if val in ['', 'NA', 'N/A', 'null']:
 clean_row[key] = None
 else:
 if key in ['N', 'P', 'K']:
 clean_row[key] = safe_float(val)
 elif key in ['Temperature', 'Humidity', 'pH', 'Rainfall', 'Soil_Moisture', 'Sunlight_Exposure', 'Wind_Speed', 'CO2_Concentration', 'Organic_Matter']:
 clean_row[key] = safe_float(val)
 else:
 clean_row[key] = val
 data.append(clean_row)
 return data

def crop_health_classification(data):
 # Simple classification based on NPK, Temperature, and Soil Moisture
 classifications = []
 for row in data:
 if row['N'] is not None and row['P'] is not None and row['K'] is not None and row['Temperature'] is not None and row['Soil_Moisture'] is not None:
 # Basic threshold-based classification
 if row['N'] > 50 and row['P'] > 30 and row['K'] > 40 and row['Temperature'] > 15 and row['Soil_Moisture'] > 30:
 classifications.append('Healthy')
 else:
 classifications.append('Unhealthy')
 else:
 classifications.append('Insufficient Data')
 return classifications

def main():
 data_type = 'agri-data'
 data = load_data(data_type)
 classifications = crop_health_classification(data)

 # Save results to JSON
 result_summary = []
 for i, classification in enumerate(classifications):
 result_summary.append({"row_id": i, "classification": classification})
 output = {
 "task_name": "Crop Health Classification",
 "description": "Match NPK, temperature, and soil moisture levels to predict crop health",
 "result_summary": result_summary,
 "result_generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
 }
 with open(f'output/{data_type}/Crop Health Classification_result.json', 'w') as f:
 json.dump(output, f)

if __name__ == '__main__':
 main()