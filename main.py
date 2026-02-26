import sys
import os

# --- 1. THE CRITICAL PATH FIX ---
# This finds the 'TSLA_Stock_System' folder (the parent of 'model_api')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Add the root directory to sys.path so we can find data_loader and preprocess
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- 2. NOW IMPORTS WORK ---
from data_loader import load_local_data
from preprocess import clean_and_scale, prepare_input_for_model

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
from datetime import datetime

app = FastAPI(title="TSLA Stock Predictor")

# Define path constants for the rest of the script
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models/tsla_model.pkl")
CSV_PATH = os.path.join(ROOT_DIR, "data/tsla_data.csv")
STATIC_PATH = os.path.join(ROOT_DIR, "dashboard/static")
TEMPLATES_PATH = os.path.join(ROOT_DIR, "dashboard/templates")

# 3. Database Initialization
def init_db():
    # Store DB in the root folder so it's easy to find
    db_path = os.path.join(ROOT_DIR, 'predictions.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (timestamp TEXT, open REAL, high REAL, low REAL, volume REAL, prediction REAL)''')
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {db_path}")

init_db()

# 4. Mount static files and templates
if os.path.exists(STATIC_PATH):
    app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=TEMPLATES_PATH)

# 5. Global state
model = None
scaler = None

def startup_logic():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    
    df = load_local_data(CSV_PATH)
    if df is not None:
        _, scaler = clean_and_scale(df)
        print("✅ Scaler initialized from CSV.")

startup_logic()

class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float

# 6. Routes
@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "model_loaded": model is not None and scaler is not None
    })

@app.post("/predict")
async def predict_stock(data: StockInput):
    if model is None or scaler is None:
        return {"status": "error", "message": "Model or Scaler not ready."}
    try:
        scaled_input = prepare_input_for_model(data.open, data.high, data.low, data.volume, scaler)
        prediction_raw = model.predict(scaled_input)[0]
        prediction = round(float(prediction_raw), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(os.path.join(ROOT_DIR, 'predictions.db'))
        c = conn.cursor()
        c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, data.open, data.high, data.low, data.volume, prediction))
        conn.commit()
        conn.close()

        return {"status": "success", "predicted_close": prediction, "timestamp": timestamp}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)