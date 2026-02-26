import pandas as pd
import os

def load_local_data(file_path="data/tsla_data.csv"):
    """Loads the Tesla CSV from your local folder."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Ensure the Date column is actually treated as a date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        print(f"✅ Successfully loaded {len(df)} rows of Tesla data.")
        return df
    except Exception as e:
        print(f"❌ An error occurred while loading: {e}")
        return None