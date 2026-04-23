from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load model and scaler
model = joblib.load("traffic_predictor.pkl")
scaler = joblib.load("traffic_scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body using Pydantic
class TrafficInput(BaseModel):
    hour: int
    dayofweek: int
    is_weekend: int
    temperature: float
    humidity: float
    rain: float
    wind_speed: float
    vehicle_count_last_5min: int
    weather_condition: int

@app.post("/predict")
def predict_traffic(data: TrafficInput):
    # Prepare features
    features = [
        data.hour,
        data.dayofweek,
        data.is_weekend,
        data.temperature,
        data.humidity,
        data.rain,
        data.wind_speed,
        data.vehicle_count_last_5min,
        data.weather_condition
    ]
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    predicted_vehicle_count = model.predict(X_scaled)[0]

    return {"predicted_vehicle_count": round(float(predicted_vehicle_count), 2)}


if __name__ == "__main__":
    uvicorn.run("predict_traffic:app", host="0.0.0.0", port=8090, reload=False)