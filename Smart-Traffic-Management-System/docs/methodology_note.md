# Methodology Note – Data Sandbox & AI Models

## Phase 1 Data Sandbox

### Video input simulation
Live CCTV streams are simulated by looping recorded `.mov` files through OpenCV.
Each file is served as an MJPEG stream over HTTP (equivalent to RTSP for local use).
Frame size is processed at 1280×720; the inference engine accepts any input resolution.
The sandbox files (`sense01–04.mov`) are representative clips from the WTS dataset
(AI City Challenge Track 2), which provides overhead-view pedestrian/vehicle footage
comparable to Amman intersection camera placement.

**Limitation:** The real site uses H.264 RTSP streams at 1920×1080.
The sandbox downscales to 1280×720 to reduce CPU load on laptop hardware.
In production, replace `cv2.VideoCapture(file)` with `cv2.VideoCapture(rtsp_url)`.

### Traffic detector dataset
Generated synthetically by `generate_sandbox.py` using a demand model calibrated to:
- Published Amman peak-hour traffic profiles (Jordan Ministry of Public Works, 2022)
- Wadi Saqra intersection geometry (4 approaches, 22 detectors, 6+6+5+5 lane layout)
- Jordan working week (Sunday–Thursday business days; Friday = religious day with reduced midday traffic)
- Seasonal temperature range for April in Amman (22–34°C)

The demand curve uses sinusoidal blending between known Amman peak times
(7:30–9:30, 13:00–14:30, 16:30–19:30).  Noise is Gaussian (σ = 18% of mean).

**Limitation:** No real loop-detector exports were available.
The synthetic data matches known aggregate patterns but cannot reproduce micro-level
lane imbalances specific to Wadi Saqra.  Replace with actual detector exports when available.

### Signal timing log
Generated from a 4-phase cycle (167 s) with adaptive green extension (±12 s) based
on the same demand model.  Phase timings are based on publicly observable signal cycles
at comparable West Amman intersections.

**Limitation:** Actual GAM signal controller logs may use different phase numbering
and event formats.  The log format was designed to be easily remapped.

---

## Real-Time Incident Detection

### Model
YOLOv12s (Ultralytics) pre-trained on COCO.  No fine-tuning was performed on Amman-specific data.
Fine-tuning on annotated Wadi Saqra clips would improve precision for:
- Jordanian vehicle types (older sedans, pickups, school buses)
- Overhead camera angle at 6 m height
- Night and low-light conditions

### Tracking
ByteTrack-like tracker with IoU-based greedy matching.
Speed calculated as pixel displacement per second, calibrated to 8 px/s stationarity threshold.

### Event thresholds
All event thresholds are configurable in `camera_thresholds.json` and the
`camera_config` defaults in `http_stream.py`.  Conservative defaults are set to
minimise false positives during a live demo.

### Accident risk score
Heuristic pairwise analysis of active vehicle tracks:
- Bounding-box IoU > 0.05 → collision probability (up to 100%)
- Centroid distance < 3× average vehicle diameter → proximity risk (up to 55%)
- Closing velocity (dot product of displacement vectors) → speed risk (up to +35%)
- Scenario multiplier: heavy rain ×1.45, night ×1.30, rush hour ×1.20

**Limitation:** True collision physics require 3D reconstruction and known vehicle dimensions.
The heuristic risk score is an indicator, not a certified collision predictor.

---

## Traffic Flow Forecasting

### RandomForest model (baseline)
Trained on 10 days of synthetic detector data at 5-minute resolution.
Features: hour, day-of-week, is_weekend, temperature, humidity, rain, wind_speed,
vehicle_count_last_5min, weather_condition.
Single-horizon output.  Multi-horizon produced by querying the model at +0.25, +0.5, +1.0 hour offsets.

### LSTM model (primary, if trained)
PyTorch LSTM with 2 layers, hidden=64.
Input: 12-step sliding window (3 hours) × 10 features.
Output: [count_+15min, count_+30min, count_+60min] simultaneously.
Trained on the generated 14-day detector dataset.
Validation: 15% chronological holdout.

### Signal recommendations
Rule-based logic applied to per-approach 15-minute forecasts:
- Predicted demand > current × 1.40 → EXTEND_GREEN (up to +20 s)
- Predicted demand < current × 0.65 → REDUCE_GREEN (up to −15 s)
- Otherwise → MAINTAIN

**These recommendations are strictly for human operator evaluation.
No automated signal control commands are issued.**

---

## Limitations Summary

| Area | Limitation | Production Fix |
|---|---|---|
| Video input | HTTP MJPEG instead of RTSP H.264 | Replace VideoCapture source with RTSP URL |
| Resolution | 1280×720 instead of 1920×1080 | Remove downscale or increase to 1920×1080 |
| Detector data | Synthetic (model-based) | Replace with real GAM detector exports |
| Signal logs | Synthetic (estimated timings) | Connect to GAM signal controller log feed |
| Accident detection | Heuristic proximity model | Train dedicated collision detection model |
| Forecasting | No live weather input | Integrate OpenWeatherMap or local weather station |
| Authentication | Simple shared secret | Replace with LDAP/SSO for enterprise deployment |
| Storage | SQLite single-file | Migrate to PostgreSQL for multi-site deployment |
