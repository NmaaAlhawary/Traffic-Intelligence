# Test Cases & Validation – Wadi Saqra Traffic Intelligence System

## TC-01 – Video stream ingestion and reconnection

**Module:** Data Acquisition Layer (`http_stream.py`)
**Steps:**
1. Start a camera stream pointing to a valid video file.
2. Wait for the stream to loop (video ends).
3. Confirm `reconnect_count` increments in `/stats`.
4. Confirm `/video_feed` continues serving frames without interruption.
**Pass:** `reconnect_count >= 1`, no gap in MJPEG output.

---

## TC-02 – Vehicle detection accuracy

**Module:** Real-Time Incident Detection (`http_stream.py`)
**Steps:**
1. Run YOLOv12s on `sense01.mov` with `confidence=0.18`.
2. Inspect `/stats` counts for `car`, `truck`, `bus`, `person`.
3. Manually review 10 annotated frames from `ground_truth_annotations.json`.
**Pass:** Precision > 0.70, Recall > 0.65 on the 10 sample frames.

---

## TC-03 – Stalled vehicle detection

**Module:** Incident detection (`_analyze_events`)
**Steps:**
1. Use video `sense02.mov` with demo accident mode enabled.
2. Wait 30 s.
3. Call `GET /events`.
4. Check for `event_type = "stalled_vehicle"` in response.
**Pass:** At least one `stalled_vehicle` event appears within 60 s of stream start.

---

## TC-04 – Accident risk scoring

**Module:** `_compute_accident_risk()`
**Steps:**
1. Enable `--demo-accident` flag on CAM-02.
2. Wait for the demo accident window (seconds 4–12 of loop).
3. Call `GET /accident_risk`.
4. Observe `accident_risk` value.
**Pass:** `accident_risk >= 70` during demo accident window.

---

## TC-05 – Accident alert banner

**Module:** Dashboard (`dashboard.py`)
**Steps:**
1. Open `http://localhost:8000/` (logged in).
2. Trigger demo accident on CAM-02 (or wait for the loop).
3. Observe browser UI.
**Pass:** Red alert banner appears at top of page; alert sound plays; camera card shows red border.

---

## TC-06 – Multi-horizon traffic forecast

**Module:** Forecasting API (`predict_traffic.py`)
**Steps:**
1. POST to `http://localhost:8090/predict/multihorizon` with `vehicle_count_last_15min=40, hour=8.5, dayofweek=2`.
2. Inspect response.
**Pass:** Response contains `total.horizon_15`, `total.horizon_30`, `total.horizon_60` all > 0 and `model` field is either `"lstm"` or `"random_forest"`.

---

## TC-07 – Signal timing recommendations

**Module:** Forecasting API + Dashboard
**Steps:**
1. GET `http://localhost:8090/recommendations`.
2. Inspect `recommendations` array.
**Pass:** Response contains 4 entries (one per approach) with fields: `approach`, `action`, `current_green_s`, `recommended_green_s`, `urgency`, `reason`.

---

## TC-08 – Dashboard authentication

**Module:** Dashboard auth middleware
**Steps:**
1. Open `http://localhost:8000/` in a fresh browser (no cookie).
2. Confirm redirect to `/login`.
3. Enter username=`admin`, password=`wrong`.
4. Confirm error message shown.
5. Enter correct credentials (`admin` / `traffic2024`).
6. Confirm redirect to main dashboard.
**Pass:** Unauthenticated request → `/login`; wrong password → error; correct password → dashboard access.

---

## TC-09 – Event persistence to SQLite

**Module:** Storage layer (`storage.py`)
**Steps:**
1. Start system and wait 2 minutes.
2. Check `detection/data/traffic_events.db` exists.
3. GET `http://localhost:8000/api/events/db`.
4. Confirm events returned from DB match in-memory events.
**Pass:** DB file exists; `events/db` endpoint returns ≥ 1 event; event fields match in-memory structure.

---

## TC-10 – Prometheus metrics endpoint

**Module:** `http_stream.py /metrics`
**Steps:**
1. GET `http://localhost:8011/metrics` (CAM-01 NORTH).
2. Inspect response body.
**Pass:** Response is text/plain, contains `traffic_frames_processed_total`, `traffic_accident_risk`, `traffic_vehicle_count` labels.

---

## TC-11 – Sandbox data generation

**Module:** `generate_sandbox.py`
**Steps:**
1. Run `python detection/src/data/generate_sandbox.py`.
2. Confirm all 5 output files are created.
3. Load `detector_dataset.csv` and verify: 22 unique `detector_id` values, 15-min timestamp resolution, ≥ 10,000 rows.
4. Load `signal_timing_log.csv` and verify: all 4 phases present, states = {GREEN_ON, YELLOW_ON, ALL_RED}.
**Pass:** All files present; detector CSV has exactly 22 detector IDs; signal CSV covers all 4 phases.

---

## TC-12 – LSTM training (optional / if GPU available)

**Module:** `train_multihorizon.py`
**Steps:**
1. Run `python detection/src/train_multihorizon.py --epochs 20`.
2. Confirm `model/lstm_multihorizon.pt` is created.
3. Re-run TC-06 and confirm `model` field = `"lstm"`.
**Pass:** Model file created; validation RMSE < 30 vehicles; forecasting API uses LSTM.

---

## TC-13 – Multi-camera health aggregation

**Module:** `incident_detector.py`
**Steps:**
1. With all 4 camera streams running, GET `http://localhost:5002/health`.
2. Inspect `active_tracks` per camera.
**Pass:** Response contains tracks for all 4 cameras; `frames_processed` > 0 for each.

---

## Benchmark Targets

| Metric | Target | Notes |
|---|---|---|
| Frame decode latency | < 100 ms/frame | At 1280×720, CPU inference |
| Event detection latency | < 2 s after trigger | From occurrence to event in `/events` |
| Forecast response time | < 500 ms | `/predict/multihorizon` cold call |
| Dashboard load time | < 3 s | First paint after login |
| Stream reconnect time | < 10 s | After video loop |
| Frames dropped rate | < 5% | Over 10-minute run |
| Dashboard WebSocket lag | < 1.5 s | From stat change to browser update |
