import os
import sys

# --- THE FIX: Tell the bossy library to ignore the keyboard ---
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "saved_models", "tsla_model.pkl")
TEMPLATES_PATH = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_PATH)

# Load Model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Success! Model is ready.")
except:
    print("❌ Model not found! Check your saved_models folder.")

class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: StockInput):
    features = np.array([[data.open, data.high, data.low, data.volume]])
    prediction = model.predict(features)[0]
    return {"predicted_close": round(float(prediction), 2)}

if __name__ == "__main__":
    import uvicorn
    # We run without 'reload' to prevent the Fortran crash
    uvicorn.run(app, host="127.0.0.1", port=8081)