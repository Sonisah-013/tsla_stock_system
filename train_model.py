import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# ---------------------------
# Step 1: Load dataset
# ---------------------------
DATA_PATH = "data/data_tsla.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully ✅")

# ---------------------------
# Step 2: Define features & target
# ---------------------------
features = ["open", "high", "low", "volume"]
target = "close"

X = df[features]
y = df[target]

# ---------------------------
# Step 3: Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 4: Train model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully ✅")

# ---------------------------
# Step 5: Evaluate model
# ---------------------------
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ---------------------------
# Step 6: Save model
# ---------------------------
os.makedirs("saved_models", exist_ok=True)
MODEL_PATH = "saved_models/tsla_model.pkl"
joblib.dump(model, MODEL_PATH)

print(f"Model saved at {MODEL_PATH} ✅")
