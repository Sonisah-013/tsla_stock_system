import numpy as np

def prepare_features(open_price: float, high: float, low: float, volume: float):
    """
    Converts raw stock data into a format the model understands.
    """
    # Create the 2D array needed for joblib/sklearn
    features = np.array([[open_price, high, low, volume]])
    return features

def get_trend_message(predicted_close, current_open):
    """
    Compares the prediction to the open price to give a 'Bull' or 'Bear' signal.
    """
    if predicted_close > current_open:
        return "🚀 Bullish (Price expected to rise)"
    else:
        return "📉 Bearish (Price expected to drop)"
