import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_and_scale(df):
    """Cleans the data and scales it for the AI model."""
    # 1. Drop any missing values (NaN)
    df_clean = df.dropna().copy()
    
    # 2. Select the features your model uses (matching your Market Input)
    # We use these 4 because they are in your dashboard
    features = ['Open', 'High', 'Low', 'Volume']
    data_to_scale = df_clean[features]
    
    # 3. Scaling (The most important step for safe predictions)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)
    
    # We return the scaled data AND the scaler 
    # (You need the scaler later to 'un-scale' the prediction back to dollars)
    return scaled_data, scaler

def prepare_input_for_model(open_p, high, low, volume, scaler):
    """Converts user input from the dashboard into a format the model understands."""
    user_input = np.array([[open_p, high, low, volume]])
    scaled_input = scaler.transform(user_input)
    return scaled_input