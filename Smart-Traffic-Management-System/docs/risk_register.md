# Risk Register – Wadi Saqra Traffic Intelligence System

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R01 | CCTV stream drops or becomes unavailable | Medium | High | High | Auto-reconnect with 5 s retry; dashboard shows OFFLINE badge; events continue from last known state | Detection layer |
| R02 | YOLO model misses or mis-classifies vehicles in low-light or rain | Medium | Medium | Medium | Confidence threshold tunable per camera; `camera_thresholds.json` allows per-camera tuning; risk score uses conservative thresholds | AI module |
| R03 | Accident risk score triggers false positive alert | Medium | Medium | Medium | 6-second cooldown on event emission; risk threshold set at 80% (conservative); demo mode for testing without real video | Detection layer |
| R04 | LSTM model not trained before demo (no `lstm_multihorizon.pt`) | Low | Low | Low | System automatically falls back to RandomForest multi-horizon predictions; graceful degradation built in | Forecasting module |
| R05 | Forecasting API (port 8090) not running | Low | Medium | Medium | Dashboard forecast panel shows "Loading" but all other panels continue to function; no system-wide failure | Dashboard |
| R06 | SQLite DB file corrupted | Very Low | Medium | Low | DB is supplementary; all events also stored in memory deques; artifacts saved as flat files on disk | Storage layer |
| R07 | Dashboard port 8000 conflict with another service | Low | Medium | Medium | Port configurable via `--port` argument; all camera ports also configurable | Deployment |
| R08 | Video files (sense0*.mov) missing or unreadable | Low | High | High | `run_wadi_saqra_streams.py` raises `FileNotFoundError` with clear path message before any process starts | Orchestration |
| R09 | GPU memory exhaustion during YOLOv12 inference | Low | Medium | Medium | `--device cpu` fallback; `--infer-every` skips frames to reduce load; image size configurable via `--imgsz` | AI module |
| R10 | Multiple browser clients overloading WebSocket broadcast | Low | Low | Low | Dead-client cleanup on every broadcast cycle; WS connections capped implicitly by OS TCP limit | Dashboard |
| R11 | Unauthorized access to dashboard | Medium | High | High | Session-cookie auth required; sessions expire after 24 hours; credentials never sent in URL | Auth layer |
| R12 | Forecasting uses weather=0/temp=28 defaults (no live weather feed) | High | Low | Low | Document assumption; replace with OpenWeatherMap API call in production | Forecasting |
| R13 | Signal recommendations sent without operator review | Low | High | High | Recommendations are display-only; no connection to signal controllers; read-only isolation enforced by architecture | System design |
| R14 | Accident detection based only on bounding-box proximity (not true physics) | Medium | Medium | Medium | Documented as heuristic; supplemented by speed/trajectory events; confidence scores indicate uncertainty | AI module |
| R15 | Sandbox data patterns may not match real Wadi Saqra traffic | Medium | Medium | Medium | Demand model uses published Amman peak-hour profiles; retrain LSTM with real detector exports when available | Data layer |
