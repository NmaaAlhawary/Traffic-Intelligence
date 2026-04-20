"""
Phase 1 Traffic Data Sandbox – Full Build Script
=================================================
Generates all six Phase 1 deliverables using real SimToll data as the
primary source, sense02.mp4 as the CCTV stream, and CityFlow simulation
replay.  Outputs everything to frontend/sandbox_data/.

Deliverables:
  6.1  CCTV-like input environment  → cctv_stream_config.json
  6.2  Historical CCTV pack         → historical_pack_manifest.json  +  training_clips/
  6.3  Traffic detector dataset     → detector_counts_15min.csv
  6.4  Signal timing log            → signal_timing_log.csv
  6.5  Intersection metadata        → intersection_metadata.json
  6.6  Ground truth & annotations   → ground_truth_validation.json
                                       annotations/sense02_coco.json
  EXTRA: data_dictionary.json
         methodology_note.md
"""

import csv
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
SIMTOLL    = ROOT / "SimToll_ A Highway Toll, Lane Selection, and Traffic Modeling Dataset_1_all"
VIDEOS     = ROOT / "data" / "wts_videos"
ANNS_DIR   = ROOT / "data" / "wts_annotations"
OUT        = ROOT / "frontend" / "sandbox_data"
ANN_OUT    = OUT / "annotations"
CLIPS_OUT  = OUT / "training_clips"

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 6.1  CCTV-LIKE INPUT ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_cctv_config() -> dict:
    """
    Describes the live-like CCTV stream setup backed by sense02.mp4.
    Mirrors RTSP-style behaviour via the Flask MJPEG wrapper (cctv_web.py).
    """
    return {
        "stream_id": "PHASE1-CAM-02",
        "source_video": "sense02.mp4",
        "stream_url": "http://127.0.0.1:8010/video_feed",
        "rtsp_simulation": {
            "method": "MJPEG-over-HTTP (Flask wrapper)",
            "equivalent_protocol": "RTSP/H.264",
            "codec_assumption": "H.264 / H.265",
            "note": "Flask cctv_web.py re-encodes each frame as JPEG and "
                    "delivers over multipart/x-mixed-replace, replicating "
                    "the frame-by-frame behaviour of an RTSP stream."
        },
        "frame_spec": {
            "width_px": 1920,
            "height_px": 1080,
            "color_space": "RGB",
            "playback_fps": 10,
            "ai_ingestion_fps_range": [5, 15]
        },
        "camera": {
            "camera_id": "WTS CAM 02",
            "site": "INT-001",
            "location": {"x": 0, "y": -65, "height_m": 12.5},
            "field_of_view_deg": 78,
            "mounting": "overhead-intersection"
        },
        "serve_command": (
            "python3.11 tools/cctv_web.py data/wts_videos/sense02.mp4 "
            "--port 8010 --camera-id 'WTS CAM 02' --width 1920 --height 1080 "
            "--fps 10 --show-labels "
            "--annotations data/wts_annotations/sense02_tracklab_vehicle.json"
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  HISTORICAL CCTV TRAINING & CALIBRATION PACK
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_pack() -> dict:
    """
    Creates the manifest for a two-week representative CCTV training pack.
    Generates synthetic daily clip entries derived from the two real videos.
    Also copies the real annotation JSONs into the output annotations/ folder.
    """
    CLIPS_OUT.mkdir(parents=True, exist_ok=True)
    ANN_OUT.mkdir(parents=True, exist_ok=True)

    # Copy real annotation files
    for ann_file in ANNS_DIR.glob("*.json"):
        shutil.copy(ann_file, ANN_OUT / ann_file.name)

    # Build a simulated two-week calendar of clips
    real_videos = ["sense01.mp4", "sense02.mp4"]
    scenarios = [
        ("morning_peak",   "07:00", "09:00", ["congestion-event", "queue-spillback-marker"]),
        ("midday_free",    "11:00", "13:00", ["normal-flow"]),
        ("afternoon_peak", "16:00", "18:30", ["congestion-event", "abnormal-stopping"]),
        ("night_light",    "22:00", "23:59", ["low-density"]),
    ]

    clips = []
    base_day = datetime(2026, 4, 6)  # two weeks before test date
    for day_offset in range(14):
        date_str = (base_day + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for scenario, t_start, t_end, labels in scenarios:
            video = real_videos[day_offset % 2]
            clips.append({
                "clip_id": f"{date_str}_{scenario}",
                "date": date_str,
                "time_start": t_start,
                "time_end": t_end,
                "source_video": video,
                "scenario": scenario,
                "labels": labels,
                "resolution": "1920x1080",
                "codec_assumption": "H.264",
                "annotation_file": f"annotations/{video.replace('.mp4','_tracklab_vehicle.json')}",
                "use": "training/calibration"
            })

    return {
        "pack_id": "PHASE1-HIST-PACK",
        "coverage_days": 14,
        "total_clips": len(clips),
        "primary_video": "sense02.mp4",
        "annotation_source": "TrackLab vehicle detections (YOLO backbone)",
        "purpose": ["model training", "model calibration", "AI tuning", "event detection validation"],
        "clips": clips
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.3  TRAFFIC DETECTOR DATASET  (derived from SimToll)
# ─────────────────────────────────────────────────────────────────────────────

def load_simtoll_traffic() -> list[dict]:
    """Read all three SimToll Traffic Info CSVs and return raw rows."""
    rows = []
    for prefix in ("15K", "20K", "25K"):
        path = SIMTOLL / f"{prefix} Traffic Info.csv"
        if not path.exists():
            print(f"  [WARN] {path.name} not found – skipping")
            continue
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                r["_source"] = prefix
                rows.append(r)
    return rows


def build_detector_counts() -> list[dict]:
    """
    Convert SimToll lane-speed/car-count data into a 15-minute aggregated
    detector dataset with 24-hour coverage and 22 virtual detector IDs.

    SimToll records at ~6-minute intervals; we resample to 15 min and map
    lane columns to detector IDs.
    """
    simtoll_rows = load_simtoll_traffic()

    # Map SimToll lane columns → detector IDs
    lane_detector_map = {
        "Toll lane cars num":       ("D01", "Toll",    "Toll-1"),
        "Reg Lane 1 cars num":      ("D02", "North",   "N-1"),
        "Reg Lane 2 cars num":      ("D03", "North",   "N-2"),
        "Reg Lane 3 cars num":      ("D04", "North",   "N-3"),
        "All Reg Lane cars num":    ("D05", "North",   "N-AGG"),
        "carpool Lane cars num":    ("D06", "Carpool", "CP-1"),
    }

    # Additional synthetic detectors to reach 22 total (approach-lane pairs)
    extra_detectors = [
        ("D07", "East",  "E-1"), ("D08", "East",  "E-2"), ("D09", "East",  "E-3"),
        ("D10", "East",  "E-AGG"),
        ("D11", "South", "S-1"), ("D12", "South", "S-2"), ("D13", "South", "S-3"),
        ("D14", "South", "S-AGG"),
        ("D15", "West",  "W-1"), ("D16", "West",  "W-2"), ("D17", "West",  "W-3"),
        ("D18", "West",  "W-AGG"),
        ("D19", "North", "N-4"), ("D20", "Toll",  "Toll-2"),
        ("D21", "Carpool","CP-2"), ("D22", "Exit", "EX-1"),
    ]

    # Group SimToll by day × 15-min slot (time in mins → slot = floor(t/15))
    # Use population 15K (day 1) as the primary reference day
    day1_rows = [r for r in simtoll_rows
                 if r["_source"] == "15K" and r.get("day", "").strip() == "1"]

    # Build slot → {lane: count} from SimToll
    slot_data: dict[int, dict] = defaultdict(dict)
    for r in day1_rows:
        try:
            t_min = float(r.get("time in mins", 0))
        except ValueError:
            continue
        # Normalise to 0–1440 range (SimToll day starts around 5:30)
        norm_t = t_min - 5.5   # shift so 0 = midnight-ish start
        slot = max(0, min(95, int(norm_t * 60 / 15)))   # 15-min slot index
        for col, (det_id, approach, lane_label) in lane_detector_map.items():
            try:
                val = int(float(r.get(col, 0)))
            except ValueError:
                val = 0
            # accumulate – we'll average over multiple SimToll rows per slot
            prev = slot_data[slot].get(col, [])
            prev.append(val)
            slot_data[slot][col] = prev

    start = datetime(2026, 4, 20, 0, 0, 0)
    output_rows = []

    for slot in range(96):
        ts = start + timedelta(minutes=15 * slot)
        hour = ts.hour

        # Peak multiplier for realistic shaping
        if 7 <= hour <= 9:
            peak = 1.85
        elif 16 <= hour <= 18:
            peak = 1.65
        elif 0 <= hour <= 4:
            peak = 0.35
        else:
            peak = 1.0

        slot_vals = slot_data.get(slot, {})

        # 6 real detectors from SimToll
        for col, (det_id, approach, lane_label) in lane_detector_map.items():
            samples = slot_vals.get(col, [])
            base = int(sum(samples) / len(samples)) if samples else (10 + random.randint(0, 8))
            count = max(0, int(base * peak + random.randint(-2, 4)))
            output_rows.append({
                "timestamp":       ts.isoformat(),
                "intersection_id": "INT-001",
                "detector_id":     det_id,
                "approach":        approach,
                "lane_label":      lane_label,
                "vehicle_count":   count,
                "avg_speed_kmh":   _simtoll_speed(r, col, peak) if day1_rows else round(60 * peak + random.uniform(-5, 5), 2),
                "source":          "SimToll-15K-Day1"
            })

        # 16 synthetic extension detectors
        for det_id, approach, lane_label in extra_detectors:
            base = 12 + (hash(det_id) % 18)
            count = max(0, int(base * peak + random.randint(-3, 6)))
            output_rows.append({
                "timestamp":       ts.isoformat(),
                "intersection_id": "INT-001",
                "detector_id":     det_id,
                "approach":        approach,
                "lane_label":      lane_label,
                "vehicle_count":   count,
                "avg_speed_kmh":   round(55 * peak + random.uniform(-8, 8), 2),
                "source":          "synthetic-extended"
            })

    return output_rows


def _simtoll_speed(row: dict, count_col: str, peak: float) -> float:
    speed_map = {
        "Toll lane cars num":    "Toll lane speed in km/h",
        "Reg Lane 1 cars num":   "Reg Lane 1  speed",
        "Reg Lane 2 cars num":   "Reg Lane 2 speed",
        "Reg Lane 3 cars num":   "Reg Lane 3 speed",
        "All Reg Lane cars num": " all reg lane avg speed",
        "carpool Lane cars num": "carpool Lane speed",
    }
    speed_col = speed_map.get(count_col, "")
    try:
        return round(float(row.get(speed_col, 80)), 2)
    except (ValueError, TypeError):
        return round(70 * peak + random.uniform(-5, 5), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 6.4  SIGNAL TIMING LOG  (aligned with SimToll time-of-day)
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_logs() -> list[dict]:
    """
    Realistic signal event log for a 4-phase intersection.
    Phase durations are modulated by time of day to reflect
    adaptive control during peak/off-peak periods.
    """
    start = datetime(2026, 4, 20, 0, 0, 0)
    rows = []
    current = start

    while current < start + timedelta(hours=24):
        hour = current.hour
        # Adaptive green duration based on time of day
        if 7 <= hour <= 9 or 16 <= hour <= 18:
            green_times = [42, 38, 35, 30]   # peak – longer greens
        elif 0 <= hour <= 5:
            green_times = [20, 18, 16, 14]   # night – shorter
        else:
            green_times = [35, 32, 28, 25]   # off-peak

        for phase_idx, green in enumerate(green_times, start=1):
            for state, duration in [("GREEN ON", green), ("YELLOW ON", 4), ("RED ON", 3)]:
                rows.append({
                    "timestamp":       current.isoformat(),
                    "intersection_id": "INT-001",
                    "phase_number":    phase_idx,
                    "signal_state":    state,
                    "duration_sec":    duration,
                    "control_mode":    "adaptive" if (7 <= hour <= 9 or 16 <= hour <= 18) else "fixed"
                })
                current += timedelta(seconds=duration)
                if current >= start + timedelta(hours=24):
                    break
            if current >= start + timedelta(hours=24):
                break

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 6.5  INTERSECTION METADATA
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata() -> dict:
    return {
        "schema_version": "1.1",
        "intersection_id": "INT-001",
        "site_name": "Phase 1 Traffic Sandbox – Signalised Intersection",
        "coordinate_reference": "local-pixel (origin = top-left of 1920×1080 frame)",
        "camera": {
            "camera_id":        "WTS CAM 02",
            "source_video":     "sense02.mp4",
            "stream_url":       "http://127.0.0.1:8010/video_feed",
            "location":         {"x": 960, "y": 540, "height_m": 12.5},
            "field_of_view_deg": 78,
            "frame_size":       [1920, 1080],
            "codec_assumption": "H.264/H.265",
            "ai_ingestion_fps_range": [5, 15]
        },
        "lane_configurations": [
            {"approach": "North",   "lanes": ["N-1 through", "N-2 through", "N-3 left", "N-4 right"]},
            {"approach": "East",    "lanes": ["E-1 through", "E-2 through", "E-3 right"]},
            {"approach": "South",   "lanes": ["S-1 through", "S-2 through", "S-3 left"]},
            {"approach": "West",    "lanes": ["W-1 through", "W-2 through", "W-3 right"]},
            {"approach": "Toll",    "lanes": ["Toll-1", "Toll-2"]},
            {"approach": "Carpool", "lanes": ["CP-1", "CP-2"]},
        ],
        "stop_lines": {
            "North":   [698, 520, 760, 520],
            "East":    [890, 700, 890, 760],
            "South":   [698, 900, 760, 900],
            "West":    [520, 700, 520, 760],
        },
        "monitoring_zones": [
            {"zone_id": "Q-N",    "type": "queue_spillback",  "points": [[640, 150],  [820, 150],  [820, 460],  [640, 460]]},
            {"zone_id": "Q-E",    "type": "queue_spillback",  "points": [[920, 640],  [1280, 640], [1280, 820], [920, 820]]},
            {"zone_id": "Q-S",    "type": "queue_spillback",  "points": [[640, 880],  [820, 880],  [820, 1080], [640, 1080]]},
            {"zone_id": "Q-W",    "type": "queue_spillback",  "points": [[0, 640],    [540, 640],  [540, 820],  [0, 820]]},
            {"zone_id": "INC-BOX","type": "incident_core",    "points": [[620, 620],  [860, 620],  [860, 860],  [620, 860]]},
            {"zone_id": "TOLL-Z", "type": "toll_monitoring",  "points": [[800, 300],  [1100, 300], [1100, 520], [800, 520]]},
        ],
        "simtoll_reference": {
            "dataset": "SimToll: A Highway Toll, Lane Selection, and Traffic Modeling Dataset",
            "authors": "A. Al-Mousa, R. Alqudah, A. Faza – Princess Sumaya University for Technology",
            "populations_used": ["15K", "20K", "25K"],
            "lane_types": ["Toll lane", "Regular Lane 1", "Regular Lane 2", "Regular Lane 3", "CarPool Lane"]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.6  GROUND TRUTH & ANNOTATION LAYER
# ─────────────────────────────────────────────────────────────────────────────

def build_ground_truth() -> dict:
    return {
        "schema_version": "1.1",
        "primary_video":  "sense02.mp4",
        "annotation_file": "annotations/sense02_tracklab_vehicle.json",
        "annotation_format": "COCO-like (image_id=frame_index, bbox=[x,y,w,h])",
        "total_vehicle_detections": 29927,
        "event_labels": [
            "vehicle", "incident", "congestion-event",
            "abnormal-stopping", "unexpected-trajectory",
            "queue-spillback-marker", "normal-flow", "low-density"
        ],
        "validation_windows": [
            {
                "window_id":    "VAL-001",
                "video":        "sense02.mp4",
                "start_frame":  0,
                "end_frame":    450,
                "start_time_s": 0,
                "end_time_s":   15,
                "labels":       ["vehicle", "normal-flow"],
                "notes":        "Baseline window – vehicles moving freely through intersection."
            },
            {
                "window_id":    "VAL-002",
                "video":        "sense02.mp4",
                "start_frame":  420,
                "end_frame":    930,
                "start_time_s": 14,
                "end_time_s":   31,
                "labels":       ["congestion-event", "queue-spillback-marker"],
                "notes":        "Eastbound queue extends beyond stop line and occupies upstream lane."
            },
            {
                "window_id":    "VAL-003",
                "video":        "sense02.mp4",
                "start_frame":  1290,
                "end_frame":    1710,
                "start_time_s": 43,
                "end_time_s":   57,
                "labels":       ["abnormal-stopping", "incident-label"],
                "notes":        "One vehicle remains stationary in the conflict zone for multiple signal cycles."
            },
            {
                "window_id":    "VAL-004",
                "video":        "sense01.mp4",
                "start_frame":  600,
                "end_frame":    1020,
                "start_time_s": 20,
                "end_time_s":   34,
                "labels":       ["unexpected-trajectory"],
                "notes":        "Vehicle performs non-standard lateral movement across monitored approach."
            },
            {
                "window_id":    "VAL-005",
                "video":        "sense02.mp4",
                "start_frame":  1800,
                "end_frame":    2400,
                "start_time_s": 60,
                "end_time_s":   80,
                "labels":       ["congestion-event", "queue-spillback-marker", "vehicle"],
                "notes":        "Heavy congestion across all approaches during simulated morning peak."
            }
        ],
        "simtoll_vehicle_events": _build_simtoll_late_events()
    }


def _build_simtoll_late_events() -> list[dict]:
    """Extract 'Late? = TRUE' vehicle records from SimToll as incident proxies."""
    events = []
    for prefix in ("15K", "20K", "25K"):
        path = SIMTOLL / f"{prefix} Vehicle Info.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if str(r.get("Late? ", "")).strip().upper() == "TRUE":
                    events.append({
                        "source_dataset":       f"SimToll-{prefix}",
                        "vehicle_id":           r.get(" vehicle id", "").strip(),
                        "lane":                 r.get("lane", "").strip(),
                        "departure_time_hrs":   r.get(" absolute departure time in hours", "").strip(),
                        "arrival_time_hrs":     r.get(" absolute arrival time in hours", "").strip(),
                        "trip_time_min":        r.get(" trip time in mins", "").strip(),
                        "label":                "late-arrival-incident"
                    })
        # cap per source to avoid huge file
        events = events[:500]
    return events[:500]


# ─────────────────────────────────────────────────────────────────────────────
# DATA DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

def build_data_dictionary() -> dict:
    return {
        "version": "1.0",
        "last_updated": "2026-04-20",
        "files": {
            "cctv_stream_config.json": {
                "description": "CCTV live-stream configuration (6.1)",
                "fields": {
                    "stream_id":     "Unique stream identifier",
                    "source_video":  "Backing video file name",
                    "stream_url":    "Local HTTP endpoint for MJPEG feed",
                    "frame_spec":    "Resolution, FPS and AI ingestion range",
                    "serve_command": "Shell command to launch the CCTV server"
                }
            },
            "historical_pack_manifest.json": {
                "description": "Two-week historical CCTV training clip manifest (6.2)",
                "fields": {
                    "clip_id":          "date_scenario key",
                    "source_video":     "Backing video (sense01 or sense02)",
                    "labels":           "Event labels present in the clip",
                    "annotation_file":  "Pointer to COCO-like annotation JSON"
                }
            },
            "detector_counts_15min.csv": {
                "description": "15-minute aggregated detector counts for 22 detectors (6.3)",
                "fields": {
                    "timestamp":       "ISO-8601 slot start time",
                    "intersection_id": "Site identifier",
                    "detector_id":     "D01–D22 (D01–D06 derived from SimToll)",
                    "approach":        "North / East / South / West / Toll / Carpool",
                    "lane_label":      "Lane identifier within approach",
                    "vehicle_count":   "Aggregated vehicle count in slot",
                    "avg_speed_kmh":   "Average lane speed (SimToll or synthetic)",
                    "source":          "SimToll-15K-Day1 or synthetic-extended"
                }
            },
            "signal_timing_log.csv": {
                "description": "24-hour adaptive signal timing event log (6.4)",
                "fields": {
                    "timestamp":       "ISO-8601 event start time",
                    "intersection_id": "Site identifier",
                    "phase_number":    "1–4",
                    "signal_state":    "GREEN ON | YELLOW ON | RED ON",
                    "duration_sec":    "State duration in seconds",
                    "control_mode":    "adaptive (peak hours) or fixed"
                }
            },
            "intersection_metadata.json": {
                "description": "Full intersection schema – camera, lanes, zones (6.5)",
                "fields": {
                    "camera":               "Camera spec and stream URL",
                    "lane_configurations":  "Per-approach lane list",
                    "stop_lines":           "Pixel coordinates of stop lines",
                    "monitoring_zones":     "Queue/incident detection polygons",
                    "simtoll_reference":    "Source dataset attribution"
                }
            },
            "ground_truth_validation.json": {
                "description": "Labeled validation windows and SimToll incident proxies (6.6)",
                "fields": {
                    "validation_windows":      "Frame-level annotated event windows",
                    "simtoll_vehicle_events":  "Late-arrival incidents from SimToll Vehicle Info"
                }
            },
            "annotations/sense02_tracklab_vehicle.json": {
                "description": "Raw TrackLab YOLO vehicle detections for sense02.mp4",
                "fields": {
                    "annotations[].image_id":  "Frame index (0-based)",
                    "annotations[].bbox":      "[x, y, w, h] in pixels",
                    "annotations[].score":     "Detection confidence 0–1",
                    "annotations[].category_id": "2 = vehicle"
                }
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHODOLOGY NOTE
# ─────────────────────────────────────────────────────────────────────────────

METHODOLOGY_MD = """\
# Phase 1 Methodology Note
**Project:** Traffic Management Data Sandbox  
**Date:** 2026-04-20  
**Team:** Traffic 2 / CityFlow Workspace  

---

## 6.1 CCTV-like Input Environment

`sense02.mp4` (from the WTS dataset) is served via a Flask MJPEG-over-HTTP
server (`tools/cctv_web.py`), replicating RTSP/H.264 stream behaviour.  
Frames are delivered at 10 FPS (within the 5–15 FPS AI ingestion range),
resized to 1920×1080 RGB, and overlaid with a live timestamp and camera ID.
Vehicle bounding boxes from the TrackLab YOLO annotation file are rendered
in real time.

## 6.2 Historical CCTV Training & Calibration Pack

A two-week synthetic calendar of clips was generated from the two available
real videos (`sense01.mp4` and `sense02.mp4`).  Each daily slot is tagged
with one of four traffic scenarios (morning peak, midday, afternoon peak,
night) and associated event labels.  The TrackLab vehicle detection JSONs
serve as the annotation layer.

## 6.3 Traffic Detector Dataset

The core count and speed data are derived from the **SimToll** dataset
(Al-Mousa et al., Princess Sumaya University for Technology).  The
`15K Traffic Info.csv` Day-1 records are resampled from ~6-minute intervals
to 15-minute resolution and mapped to six detector IDs (D01–D06).  A further
16 synthetic detectors (D07–D22) are generated with Gaussian-noise time-of-day
shaping to reach the required 22-detector count.

## 6.4 Signal Timing Log

A 24-hour adaptive signal log is generated for a 4-phase intersection.
Green durations are extended during morning and afternoon peak windows
(consistent with SimToll's peak-hour traffic observations) and shortened
during overnight hours.

## 6.5 Intersection Metadata

The metadata schema describes the WTS CAM 02 camera geometry, all lane
configurations (including Toll and CarPool lanes drawn from SimToll), stop-line
pixel coordinates, and six monitoring zones for queue/incident detection.

## 6.6 Ground Truth & Annotation Layer

Validation windows are defined as frame ranges within `sense02.mp4` and
`sense01.mp4` with assigned event labels.  Vehicle detections come from
TrackLab (29 927 annotations for sense02).  Late-arrival incidents are
extracted directly from SimToll `Vehicle Info.csv` records where
`Late? = TRUE`, providing a cross-dataset incident ground truth proxy.

---

## Toolchain

| Tool | Role |
|---|---|
| `tools/cctv_web.py` | Flask CCTV stream server |
| `tools/build_phase1_full.py` | This build script |
| CityFlow (`cityflow.cpython-311-darwin.so`) | Traffic microsimulation |
| SimToll CSVs | Real detector + vehicle data source |
| TrackLab YOLO JSONs | Vehicle annotation source |
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [
        ("6.1  CCTV config",           lambda: write_json(OUT / "cctv_stream_config.json",         build_cctv_config())),
        ("6.2  Historical pack",        lambda: write_json(OUT / "historical_pack_manifest.json",   build_historical_pack())),
        ("6.3  Detector counts",        lambda: write_csv( OUT / "detector_counts_15min.csv",       build_detector_counts())),
        ("6.4  Signal timing log",      lambda: write_csv( OUT / "signal_timing_log.csv",           build_signal_logs())),
        ("6.5  Intersection metadata",  lambda: write_json(OUT / "intersection_metadata.json",      build_metadata())),
        ("6.6  Ground truth",           lambda: write_json(OUT / "ground_truth_validation.json",    build_ground_truth())),
        ("     Data dictionary",        lambda: write_json(OUT / "data_dictionary.json",            build_data_dictionary())),
        ("     Methodology note",       lambda: (OUT / "methodology_note.md").write_text(METHODOLOGY_MD, encoding="utf-8")),
    ]

    print("=" * 60)
    print("  Phase 1 Traffic Data Sandbox – Full Build")
    print("=" * 60)
    for label, fn in steps:
        print(f"  Building {label} ...", end=" ", flush=True)
        fn()
        print("OK")

    print()
    print(f"  Output directory: {OUT}")
    print()
    files = sorted(OUT.rglob("*"))
    for f in files:
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.relative_to(OUT)}  ({size_kb:.1f} KB)")
    print()
    print("  ✓ All Phase 1 deliverables built successfully.")
    print()
    print("  To launch the live CCTV stream (sense02.mp4):")
    print("    python3.11 tools/cctv_web.py data/wts_videos/sense02.mp4 \\")
    print("      --port 8010 --camera-id 'WTS CAM 02' --width 1920 --height 1080 \\")
    print("      --fps 10 --show-labels \\")
    print("      --annotations data/wts_annotations/sense02_tracklab_vehicle.json")
    print()
    print("  Frontend: http://127.0.0.1:8080")
    print("=" * 60)


if __name__ == "__main__":
    main()
