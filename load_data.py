import os
import pandas as pd
from datetime import datetime

def get_latest_csv(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        return None
    
    # Get the latest file by modified time
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
    return os.path.join(folder_path, latest_file)

def load_top100_data():
    base_path = os.path.join(os.path.dirname(__file__), "data")

    data_paths = {
        "crypto": os.path.join(base_path, "crypto"),
        "etf": os.path.join(base_path, "etf"),
        "stocks": os.path.join(base_path, "stocks")
    }

    datasets = {}

    for category, path in data_paths.items():
        latest_csv = get_latest_csv(path)
        if latest_csv:
            df = pd.read_csv(latest_csv)
            datasets[category] = df
        else:
            datasets[category] = pd.DataFrame()

    return datasets
