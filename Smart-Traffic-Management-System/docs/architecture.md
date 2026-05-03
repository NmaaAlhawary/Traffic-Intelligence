# System Architecture – Wadi Saqra Traffic Intelligence System

## Overview
Full-stack traffic intelligence system for the Wadi Saqra intersection, Amman.
Built as the repeatable first-site blueprint for the 9XAI Hackathon.

---

## Module Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OPERATOR INTERFACES                                 │
│   Browser Dashboard (port 8000)  ·  Desktop App (Kotlin/Compose)        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP / WebSocket
┌──────────────────────────────▼──────────────────────────────────────────┐
│              VISUALIZATION & DECISION-SUPPORT LAYER                     │
│   dashboard.py  (aiohttp, port 8000)                                    │
│   • Auth middleware (session cookies)                                   │
│   • MJPEG feed proxy  /feed/{camera}                                    │
│   • WebSocket push    /ws                                               │
│   • Forecast API      /api/forecast  → proxies port 8090               │
│   • Historical API    /api/historical → SQLite                          │
│   • Scenario control  /proxy_scenario/{camera}                          │
└───────┬──────────────────┬────────────────────┬────────────────────────┘
        │                  │                    │
        ▼                  ▼                    ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│ INCIDENT     │  │ FORECASTING &    │  │ DATA STORAGE & LOGGING       │
│ DETECTION    │  │ SIGNAL OPT.      │  │ storage.py  (SQLite)         │
│ LAYER        │  │ predict_traffic  │  │ • events table               │
│              │  │ .py (FastAPI     │  │ • forecasts table            │
│ http_stream  │  │  port 8090)      │  │ • performance_log table      │
│ .py (aiohttp │  │                  │  │ • signal_recommendations     │
│  per camera) │  │ • RF model       │  │ • detector_counts table      │
│              │  │ • LSTM model     │  │                              │
│ • YOLOv12s   │  │ • /predict       │  │ DB path:                     │
│ • ByteTrack  │  │ • /predict/      │  │ detection/data/              │
│ • 6 event    │  │   multihorizon   │  │   traffic_events.db          │
│   types      │  │ • /recommendations│  └──────────────────────────────┘
│ • Accident   │  └──────────────────┘
│   risk score │
│ • Prometheus │  ┌──────────────────────────────────────────────────────┐
│   /metrics   │  │ DATA ACQUISITION LAYER                               │
│              │  │ run_wadi_saqra_streams.py (orchestrator)             │
│ incident_    │  │ • Launches 4 camera http_stream.py processes         │
│ detector.py  │  │ • Launches incident_detector.py (aggregator :5002)  │
│ (aggregator  │  │ • Launches dashboard.py (:8000)                      │
│  port 5002)  │  │ • Launches predict_traffic.py (:8090)               │
└──────┬───────┘  │ • Reconnect logic (5–10 s retry)                    │
       │          │ • Frame drop tracking                                │
       ▼          │ • Stream uptime monitoring                           │
┌──────────────┐  └──────────────────────────────────────────────────────┘
│ VIDEO        │
│ STREAMS      │  ┌──────────────────────────────────────────────────────┐
│              │  │ PHASE 1 DATA SANDBOX                                 │
│ sense01.mov  │  │ detection/src/data/                                  │
│ sense02.mov  │  │ • detector_dataset.csv   (22 det × 14 days × 15min) │
│ sense03.mov  │  │ • signal_timing_log.csv  (phase event log)           │
│ sense04.mov  │  │ • intersection_metadata.json                         │
│ (loop replay │  │ • ground_truth_annotations.json (20 events)          │
│  simulating  │  │ • data_dictionary.md                                 │
│  live CCTV)  │  └──────────────────────────────────────────────────────┘
└──────────────┘
```

---

## Data Flow

```
Video files (sense0*.mov)
  └─► http_stream.py  [YOLOv12s inference @ 10 FPS]
        ├─► /video_feed  (MJPEG stream)  ──► dashboard /feed/{cam}  ──► browser
        ├─► /stats       (JSON)          ──► dashboard /api/stats   ──► browser
        ├─► /events      (JSON)          ──► dashboard /api/events  ──► browser
        ├─► /metrics     (Prometheus)    ──► Prometheus scrape
        ├─► /accident_risk  (JSON)
        └─► storage.save_event()         ──► SQLite events table

detector_dataset.csv + signal_timing_log.csv
  └─► train_multihorizon.py  [PyTorch LSTM training]
        └─► model/lstm_multihorizon.pt

predict_traffic.py  [FastAPI @ :8090]
  ├─► POST /predict/multihorizon  ──► dashboard /api/forecast  ──► browser
  └─► GET  /recommendations       ──► dashboard               ──► browser

SQLite (traffic_events.db)
  └─► GET /api/historical  ──► dashboard historical panel  ──► browser
```

---

## Port Map

| Service | Port | Description |
|---|---|---|
| Dashboard | 8000 | Main web UI (auth required) |
| CAM-02 SOUTH | 8010 | YOLOv12 MJPEG stream |
| CAM-01 NORTH | 8011 | YOLOv12 MJPEG stream |
| CAM-02 SOUTH DEMO | 8012 | YOLOv12 MJPEG + accident demo |
| CAM-03 EAST | 8013 | YOLOv12 MJPEG stream |
| CAM-04 WEST | 8014 | YOLOv12 MJPEG stream |
| Incident Aggregator | 5002 | Multi-camera event merge |
| Forecasting API | 8090 | FastAPI multi-horizon predictions |
| Ktor Backend | 8080 | Kotlin REST + WebSocket server |

---

## Security & Isolation

- Dashboard requires login (session cookie, 24-hour expiry)
- System operates in **read-only** mode — no writes to operational traffic infrastructure
- All outputs are for analysis and human decision support only
- No connection to live GAM traffic signal controllers
- Prometheus metrics exposed locally only; restrict `/metrics` in production firewall

---

## Scale Path

To add a second site:
1. Deploy another set of `http_stream.py` instances pointing to new camera feeds
2. Add new camera entries to `dashboard.py` CAMERAS dict
3. Add new site entry to `intersection_metadata.json`
4. The SQLite schema supports multiple `intersection_id` values
5. The forecasting API is site-agnostic; supply different detector data to retrain LSTM
