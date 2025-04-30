# src/data_pipeline.py

import requests
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

# Constants
NASA_FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov//api/area/csv/56fcf08b44536b0ea360de0e32a45991/VIIRS_SNPP_NRT/world/1/2025-04-19"
LOCAL_RAW_DATA_PATH = "data/raw_wildfire_data.csv"
LOCAL_PROCESSED_DATA_PATH = "data/processed_wildfire_data.csv"
API_KEY = "DEMO_KEY"  # <-- Replace with your NASA API Key if required

def fetch_wildfire_data():
    print("[INFO] Fetching wildfire data from NASA FIRMS API...")
    response = requests.get(NASA_FIRMS_URL)
    if response.status_code == 200:
        with open(LOCAL_RAW_DATA_PATH, 'wb') as f:
            f.write(response.content)
        print(f"[INFO] Raw data saved to {LOCAL_RAW_DATA_PATH}")
    else:
        print(f"[ERROR] Failed to fetch data. Status code: {response.status_code}")

def preprocess_data():
    print("[INFO] Preprocessing wildfire data...")
    df = pd.read_csv(LOCAL_RAW_DATA_PATH)

    # Keep only useful columns
    required_columns = [
    'latitude', 'longitude', 'brightness', 'scan', 'track',
    'acq_date', 'acq_time', 'confidence', 'version', 'instrument'
    ]

# Keep only the columns that exist in the file
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]

    print("[DEBUG] Final columns used:", df.columns.tolist())
    # Fill missing values
    df = df.fillna(method='ffill')

    # Feature engineering: Convert time
    df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')

    # Drop old columns
    df = df.drop(['acq_date', 'acq_time'], axis=1)

    df.to_csv(LOCAL_PROCESSED_DATA_PATH, index=False)
    print(f"[INFO] Processed data saved to {LOCAL_PROCESSED_DATA_PATH}")

def visualize_data():
    print("[INFO] Visualizing wildfire hotspots...")
    df = pd.read_csv(LOCAL_PROCESSED_DATA_PATH)

    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude'], df['latitude'], s=1, c='red')
    plt.title('Wildfire Hotspots Detected')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig('data/wildfire_hotspots_map.png')
    print("[INFO] Wildfire hotspots map saved to data/wildfire_hotspots_map.png")
    plt.show()

def main():
    fetch_wildfire_data()
    preprocess_data()
    visualize_data()

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    main()
