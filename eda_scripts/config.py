import pandas as pd
import os
import sys

# Define Path to CSV
# Assuming this script is run from the 'eda_scripts' folder, we go up one level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "Resume", "Resume.csv")

def load_data():
    """Loads the Resume.csv file."""
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} records from Resume.csv")
    return df
