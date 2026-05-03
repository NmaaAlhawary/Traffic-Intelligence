"""
Traffic Flow Forecasting & Signal Optimization Support API (Phase 3 §8.3)
Serves multi-horizon predictions (15 / 30 / 60 min) and signal timing
recommendations.  Uses LSTM when available, falls back to RandomForest.

Endpoints:
  POST /predict                – original single-horizon (backward compat)
  POST /predict/multihorizon   – 15 / 30 / 60 min forecasts per approach
  GET  /recommendations        – current signal timing recommendations
  GET  /health                 – model status
"""

import math
import time
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path(__file__).resolve().parents[1] / "model"
LSTM_PATH   = MODEL_DIR / "lstm_multihorizon.pt"
RF_PATH     = Path(__file__).with_name("traffic_predictor.pkl")
SCALER_PATH = Path(__file__).with_name("traffic_scaler.pkl")

# ── Lazy-load models ──────────────────────────────────────────────────────────
_rf_model  = None
_rf_scaler = None
_lstm_pkg  = None


def _load_rf():
    global _rf_model, _rf_scaler
    if _rf_model is None and RF_PATH.exists():
        _rf_model  = joblib.load(RF_PATH)
        _rf_scaler = joblib.load(SCALER_PATH)
    return _rf_model, _rf_scaler


def _load_lstm():
    global _lstm_pkg
    if _lstm_pkg is None and LSTM_PATH.exists():
        try:
            import torch
            from train_multihorizon import TrafficLSTM
            pkg = torch.load(str(LSTM_PATH), map_location="cpu", weights_only=False)
            m   = TrafficLSTM(input_size=pkg["input_size"], hidden=pkg["hidden"])
            m.load_state_dict(pkg["model_state"])
            m.eval()
            _lstm_pkg = {"model": m, "pkg": pkg}
        except Exception as exc:
            print(f"LSTM load skipped ({exc}); using RF fallback")
    return _lstm_pkg


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Forecasting & Signal Optimization API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

APPROACHES = ["North", "South", "East", "West"]
APPROACH_PHASE   = {"North": 1, "South": 1, "East": 3, "West": 3}
DEFAULT_GREEN    = {"North": 55, "South": 55, "East": 40, "West": 40}


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


class MultiHorizonInput(BaseModel):
    hour: float
    dayofweek: int
    is_weekend: int
    temperature: float
    rain: float
    vehicle_count_last_15min: int
    dominant_phase: int = 0
    # Per-approach counts (optional; used for per-approach forecasts)
    north_count: int = -1
    south_count: int = -1
    east_count:  int = -1
    west_count:  int = -1


# ── Single-horizon (backward compat) ─────────────────────────────────────────
@app.post("/predict")
def predict_traffic(data: TrafficInput):
    model, scaler = _load_rf()
    if model is None:
        return {"predicted_vehicle_count": 0.0, "model": "unavailable"}
    features = [data.hour, data.dayofweek, data.is_weekend, data.temperature,
                data.humidity, data.rain, data.wind_speed,
                data.vehicle_count_last_5min, data.weather_condition]
    X = np.array(features).reshape(1, -1)
    pred = model.predict(scaler.transform(X))[0]
    return {"predicted_vehicle_count": round(float(pred), 2), "model": "random_forest"}


# ── Multi-horizon prediction ──────────────────────────────────────────────────
def _rf_multihorizon(data: MultiHorizonInput) -> dict:
    """Generate 15/30/60-min forecasts using the RF model for offset times."""
    model, scaler = _load_rf()
    if model is None:
        base = data.vehicle_count_last_15min
        return {"horizon_15": base, "horizon_30": base, "horizon_60": base, "model": "fallback"}

    results = {}
    # Model was trained with vehicle_count_last_5min = total_15min / 3
    count_5min = max(1, data.vehicle_count_last_15min // 3)
    # Scale factor: model predicts total across 22 detectors; dashboard uses camera counts
    det_scale = 5.5
    offsets = {"horizon_15": 0.25, "horizon_30": 0.50, "horizon_60": 1.00}
    for key, offset in offsets.items():
        h = (data.hour + offset) % 24
        is_we = int(h < 6 or h > 22) | data.is_weekend
        feats = [h, data.dayofweek, is_we, data.temperature, 60.0,
                 data.rain, 8.0, count_5min, int(data.rain)]
        X = np.array(feats).reshape(1, -1)
        pred = model.predict(scaler.transform(X))[0]
        results[key] = max(0.0, round(float(pred) / det_scale, 1))
    results["model"] = "random_forest"
    return results


def _lstm_multihorizon(data: MultiHorizonInput) -> dict | None:
    pkg_data = _load_lstm()
    if pkg_data is None:
        return None
    try:
        import torch
        pkg   = pkg_data["pkg"]
        model = pkg_data["model"]
        mean  = np.array(pkg["norm_mean"], dtype=np.float32)
        std   = np.array(pkg["norm_std"],  dtype=np.float32)
        # Build a synthetic 12-step sequence ending at current time
        steps = []
        for lag in range(12, 0, -1):
            h_lag  = (data.hour - lag * 0.25) % 24
            h_sin  = math.sin(2 * math.pi * h_lag / 24)
            h_cos  = math.cos(2 * math.pi * h_lag / 24)
            d_sin  = math.sin(2 * math.pi * data.dayofweek / 7)
            d_cos  = math.cos(2 * math.pi * data.dayofweek / 7)
            p_sin  = math.sin(2 * math.pi * data.dominant_phase / 4)
            p_cos  = math.cos(2 * math.pi * data.dominant_phase / 4)
            # approximate past count by blending current with a lag decay
            lag_count = data.vehicle_count_last_15min * max(0.4, 1 - lag * 0.04)
            steps.append([lag_count, data.rain, data.temperature,
                           h_sin, h_cos, d_sin, d_cos, float(data.is_weekend), p_sin, p_cos])
        arr  = (np.array(steps, dtype=np.float32) - mean) / std
        x    = torch.from_numpy(arr).unsqueeze(0)   # (1, 12, 10)
        with torch.no_grad():
            out = model(x).squeeze().numpy()
        # LSTM trained on total of 22 detectors; scale to match camera-count units
        # (camera count ≈ vehicles visible in frame, roughly total/avg_detectors_per_approach)
        det_scale = 5.5   # 22 detectors / 4 approaches, vehicles visible per lane-set
        h15 = float(out[0]) / det_scale
        h30 = float(out[1]) / det_scale
        h60 = float(out[2]) / det_scale
        return {
            "horizon_15": max(0.0, round(h15, 1)),
            "horizon_30": max(0.0, round(h30, 1)),
            "horizon_60": max(0.0, round(h60, 1)),
            "model": "lstm",
        }
    except Exception:
        return None


@app.post("/predict/multihorizon")
def predict_multihorizon(data: MultiHorizonInput):
    result = _lstm_multihorizon(data) or _rf_multihorizon(data)

    # Per-approach forecasts (scale total by approach share if individual counts not given)
    approach_counts = {
        "North": data.north_count if data.north_count >= 0 else int(data.vehicle_count_last_15min * 0.30),
        "South": data.south_count if data.south_count >= 0 else int(data.vehicle_count_last_15min * 0.28),
        "East":  data.east_count  if data.east_count  >= 0 else int(data.vehicle_count_last_15min * 0.22),
        "West":  data.west_count  if data.west_count  >= 0 else int(data.vehicle_count_last_15min * 0.20),
    }

    approach_scale = {"North": 0.30, "South": 0.28, "East": 0.22, "West": 0.20}
    per_approach = {}
    for ap, share in approach_scale.items():
        per_approach[ap] = {
            "current":    approach_counts[ap],
            "horizon_15": max(0.0, round(result["horizon_15"] * share, 1)),
            "horizon_30": max(0.0, round(result["horizon_30"] * share, 1)),
            "horizon_60": max(0.0, round(result["horizon_60"] * share, 1)),
        }

    recs = _generate_recommendations(per_approach)

    return {
        "total":        result,
        "per_approach": per_approach,
        "recommendations": recs,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── Signal recommendations ────────────────────────────────────────────────────
def _generate_recommendations(per_approach: dict) -> list:
    recs = []
    for ap, vals in per_approach.items():
        current  = vals["current"]
        pred_15  = vals["horizon_15"]
        default_g = DEFAULT_GREEN.get(ap, 45)

        if current < 5:
            continue   # too sparse to make meaningful recommendations

        ratio = pred_15 / max(current, 1)

        if ratio >= 1.40:
            delta   = min(20, int((ratio - 1.0) * 25))
            new_g   = min(90, default_g + delta)
            urgency = "HIGH" if ratio >= 1.70 else "MEDIUM"
            recs.append({
                "approach":             ap,
                "action":               "EXTEND_GREEN",
                "current_green_s":      default_g,
                "recommended_green_s":  new_g,
                "delta_s":              delta,
                "urgency":              urgency,
                "reason":               f"Demand forecast +{int((ratio-1)*100)}% in 15 min "
                                        f"({current:.0f} → {pred_15:.0f} veh)",
            })
        elif ratio <= 0.65:
            delta   = min(15, int((1.0 - ratio) * 20))
            new_g   = max(15, default_g - delta)
            recs.append({
                "approach":             ap,
                "action":               "REDUCE_GREEN",
                "current_green_s":      default_g,
                "recommended_green_s":  new_g,
                "delta_s":              -delta,
                "urgency":              "LOW",
                "reason":               f"Demand forecast -{int((1-ratio)*100)}% in 15 min "
                                        f"({current:.0f} → {pred_15:.0f} veh)",
            })
        else:
            recs.append({
                "approach":             ap,
                "action":               "MAINTAIN",
                "current_green_s":      default_g,
                "recommended_green_s":  default_g,
                "delta_s":              0,
                "urgency":              "NONE",
                "reason":               f"Demand stable ({current:.0f} → {pred_15:.0f} veh)",
            })

    return recs


@app.get("/recommendations")
def get_recommendations():
    """Quick recommendations based on current hour demand estimate."""
    import datetime
    now   = datetime.datetime.now()
    hour  = now.hour + now.minute / 60.0
    dow   = now.weekday()
    is_we = int(dow >= 5)

    # Rough current demand estimate using RF
    model, scaler = _load_rf()
    base_count = 40
    if model is not None:
        feats = [hour, dow, is_we, 28.0, 55.0, 0.0, 8.0, 40, 0]
        X     = np.array(feats).reshape(1, -1)
        base_count = max(10, int(model.predict(scaler.transform(X))[0]))

    inp = MultiHorizonInput(
        hour=hour, dayofweek=dow, is_weekend=is_we,
        temperature=28.0, rain=0.0,
        vehicle_count_last_15min=base_count,
    )
    result = predict_multihorizon(inp)
    return result["recommendations"]


@app.get("/health")
def health():
    rf_ok   = RF_PATH.exists()
    lstm_ok = LSTM_PATH.exists()
    return {
        "ok":            True,
        "rf_model":      rf_ok,
        "lstm_model":    lstm_ok,
        "active_model":  "lstm" if lstm_ok else ("random_forest" if rf_ok else "none"),
        "horizons_min":  [15, 30, 60],
    }


if __name__ == "__main__":
    uvicorn.run("predict_traffic:app", host="0.0.0.0", port=8090, reload=False)
