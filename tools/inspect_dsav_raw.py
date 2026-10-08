import pandas as pd
import numpy as np
import os

def inspect(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"Error: {label} not found at {csv_path}")
        return
    
    # Read first 1000 rows to get a good sample
    df = pd.read_csv(csv_path, nrows=1000)
    print(f"\n--- {label} ---")
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    print("Time range in sample:", df['t'].min(), "to", df['t'].max())
    print("U range in sample:", df['u'].min(), "to", df['u'].max())
    print("V range in sample:", df['v'].min(), "to", df['v'].max())

if __name__ == "__main__":
    # Hardcoded paths based on project structure
    head_csv = "../dsav360_extracted/head_data/head_data/head_video_0001.csv"
    gaze_csv = "../dsav360_extracted/gaze_data/gaze_video_0001.csv"
    
    inspect(head_csv, "HEAD DATA")
    inspect(gaze_csv, "GAZE DATA")
