from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="EcoSentinel Wildfire API")

model = joblib.load("models/wildfire_rf_model.pkl")
label_map = {0: "Low", 1: "Nominal", 2: "High"}

class FireInput(BaseModel):
    scan: float
    track: float
    latitude: float
    longitude: float
@app.get("/")
def home():
    return {"status": "EcoSentinel Wildfire API is running."}
@app.post("/predict")
def predict_fire(data: FireInput):
    features = np.array([[data.scan, data.track, data.latitude, data.longitude]])
    prediction = model.predict(features)[0]
    return {"predicted_confidence": label_map[prediction]}
import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
