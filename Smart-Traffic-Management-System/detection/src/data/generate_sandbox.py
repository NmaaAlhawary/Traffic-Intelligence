"""
Phase 1 – Traffic Data Sandbox Generator
Generates all required Phase 1 datasets for the Wadi Saqra intersection, Amman:
  - detector_dataset.csv       : 22 detectors × 14 days × 15-min resolution
  - signal_timing_log.csv      : full signal phase event log
  - intersection_metadata.json : site schema
  - ground_truth_annotations.json : labeled event windows
  - data_dictionary.md         : field definitions
Run: python generate_sandbox.py
"""

import json
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent
random.seed(42)

# ── Site constants ─────────────────────────────────────────────────────────────
INTERSECTION_ID = "WADI_SAQRA_01"
START_DATE = datetime(2026, 4, 13)          # 14 days ending day before hackathon
END_DATE   = datetime(2026, 4, 26, 23, 45)  # inclusive last 15-min slot

DETECTORS = [
    # North approach (↓ southbound)
    {"id": "DET-N01", "approach": "North", "lane": 1, "movement": "left_only"},
    {"id": "DET-N02", "approach": "North", "lane": 2, "movement": "left_straight"},
    {"id": "DET-N03", "approach": "North", "lane": 3, "movement": "straight"},
    {"id": "DET-N04", "approach": "North", "lane": 4, "movement": "straight"},
    {"id": "DET-N05", "approach": "North", "lane": 5, "movement": "straight_right"},
    {"id": "DET-N06", "approach": "North", "lane": 6, "movement": "right_only"},
    # South approach (↑ northbound)
    {"id": "DET-S01", "approach": "South", "lane": 1, "movement": "left_only"},
    {"id": "DET-S02", "approach": "South", "lane": 2, "movement": "left_straight"},
    {"id": "DET-S03", "approach": "South", "lane": 3, "movement": "straight"},
    {"id": "DET-S04", "approach": "South", "lane": 4, "movement": "straight"},
    {"id": "DET-S05", "approach": "South", "lane": 5, "movement": "straight_right"},
    {"id": "DET-S06", "approach": "South", "lane": 6, "movement": "right_only"},
    # East approach (← westbound)
    {"id": "DET-E01", "approach": "East", "lane": 1, "movement": "left_straight"},
    {"id": "DET-E02", "approach": "East", "lane": 2, "movement": "straight"},
    {"id": "DET-E03", "approach": "East", "lane": 3, "movement": "straight_right"},
    {"id": "DET-E04", "approach": "East", "lane": 4, "movement": "right_only"},
    {"id": "DET-E05", "approach": "East", "lane": 5, "movement": "u_turn"},
    # West approach (→ eastbound)
    {"id": "DET-W01", "approach": "West", "lane": 1, "movement": "left_straight"},
    {"id": "DET-W02", "approach": "West", "lane": 2, "movement": "straight"},
    {"id": "DET-W03", "approach": "West", "lane": 3, "movement": "straight_right"},
    {"id": "DET-W04", "approach": "West", "lane": 4, "movement": "right_only"},
    {"id": "DET-W05", "approach": "West", "lane": 5, "movement": "u_turn"},
]
assert len(DETECTORS) == 22

# Phase timings (seconds): phase_id → (green_s, yellow_s, all_red_s, description)
PHASES = {
    1: (55, 5, 3, "N-S straight+right"),
    2: (25, 4, 3, "N-S left turns"),
    3: (40, 5, 3, "E-W straight+right"),
    4: (20, 4, 3, "E-W left turns + U-turns"),
}
CYCLE_S = sum(g + y + r for g, y, r, _ in PHASES.values())  # 167 s

# Approach green phases
APPROACH_PHASE = {"North": [1, 2], "South": [1, 2], "East": [3, 4], "West": [3, 4]}


def demand_factor(hour: float, dow: int, is_ramadan: bool = False) -> float:
    """Return 0-1 traffic demand multiplier for Amman conditions."""
    # Base time-of-day curve
    if 0 <= hour < 5:
        base = 0.04
    elif 5 <= hour < 6:
        base = 0.10
    elif 6 <= hour < 7:
        base = 0.35
    elif 7 <= hour < 8:
        base = 0.72
    elif 8 <= hour < 9:
        base = 0.90
    elif 9 <= hour < 10:
        base = 0.78
    elif 10 <= hour < 12:
        base = 0.65
    elif 12 <= hour < 13:
        base = 0.70
    elif 13 <= hour < 14:
        base = 0.82
    elif 14 <= hour < 15:
        base = 0.88
    elif 15 <= hour < 16:
        base = 0.75
    elif 16 <= hour < 17:
        base = 0.85
    elif 17 <= hour < 18:
        base = 1.00
    elif 18 <= hour < 19:
        base = 0.92
    elif 19 <= hour < 20:
        base = 0.80
    elif 20 <= hour < 21:
        base = 0.68
    elif 21 <= hour < 22:
        base = 0.52
    else:
        base = 0.28

    # Day-of-week adjustment (0=Monday … 6=Sunday; Jordan: Fri/Sat = weekend)
    dow_mult = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.70, 6: 0.85}[dow % 7]
    # Friday prayer dip (11:30-13:00)
    if dow % 7 == 5 and 11.5 <= hour < 13.0:
        dow_mult *= 0.30

    # Ramadan: night traffic surges, day dips
    if is_ramadan:
        if 3 <= hour < 4:
            base *= 1.8   # suhoor
        elif 18 <= hour < 19:
            base *= 0.40  # iftar prep
        elif 19 <= hour < 21:
            base *= 0.30  # iftar
        elif 21 <= hour < 23:
            base *= 1.6   # after-iftar surge
        else:
            base *= 0.7

    return base * dow_mult


def peak_for_approach(approach: str, hour: float) -> float:
    """N-S is the main artery (higher volume), E-W is cross street."""
    scale = {"North": 1.15, "South": 1.10, "East": 0.90, "West": 0.85}
    return scale.get(approach, 1.0)


def lane_share(movement: str) -> float:
    """Approximate lane flow share relative to straight lane."""
    m = {"left_only": 0.6, "left_straight": 0.8, "straight": 1.0, "straight_right": 0.9,
         "right_only": 0.5, "u_turn": 0.2}
    return m.get(movement, 1.0)


def generate_detector_dataset() -> pd.DataFrame:
    rows = []
    ts = START_DATE
    while ts <= END_DATE:
        hour   = ts.hour + ts.minute / 60.0
        dow    = ts.weekday()
        factor = demand_factor(hour, dow)
        rain   = random.choices([0.0, 1.0], weights=[0.82, 0.18])[0]
        temp   = round(random.uniform(14, 32) if ts.month in (1,2,11,12) else random.uniform(22, 38), 1)

        for det in DETECTORS:
            peak_base = 72 * peak_for_approach(det["approach"], hour) * lane_share(det["movement"])
            mean_count = peak_base * factor * (0.85 if rain else 1.0)
            count = max(0, int(random.gauss(mean_count, mean_count * 0.18 + 1)))
            # occupancy pct correlates with count
            occ = min(99.0, round(count * 0.95 + random.gauss(0, 2), 1))
            speed = max(5.0, round(random.gauss(55 - factor * 20, 8), 1))

            rows.append({
                "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
                "intersection_id":  INTERSECTION_ID,
                "detector_id":      det["id"],
                "approach":         det["approach"],
                "lane":             det["lane"],
                "movement":         det["movement"],
                "vehicle_count":    count,
                "occupancy_pct":    occ,
                "speed_avg_kmh":    speed,
                "rain":             rain,
                "temp_c":           temp,
            })

        ts += timedelta(minutes=15)

    return pd.DataFrame(rows)


def generate_signal_timing_log() -> pd.DataFrame:
    rows = []
    ts   = START_DATE
    # Track current phase cycle position
    phase_order = [1, 2, 3, 4]
    phase_idx   = 0

    while ts <= END_DATE:
        phase_id = phase_order[phase_idx % len(phase_order)]
        green_s, yellow_s, all_red_s, desc = PHASES[phase_id]

        # Adaptive timing: extend green ±15% based on rough time-of-day demand
        hour = ts.hour + ts.minute / 60.0
        dow  = ts.weekday()
        f    = demand_factor(hour, dow)
        adj  = int((f - 0.5) * 12)     # ±6 s at extremes
        green_s_adj = max(15, green_s + adj)

        for state, dur in [("GREEN_ON", green_s_adj), ("YELLOW_ON", yellow_s), ("ALL_RED", all_red_s)]:
            rows.append({
                "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
                "intersection_id":  INTERSECTION_ID,
                "phase_number":     phase_id,
                "signal_state":     state,
                "duration_s":       dur,
                "phase_desc":       desc,
                "cycle_time_s":     CYCLE_S,
            })
            ts += timedelta(seconds=dur)

        phase_idx += 1
        if ts > END_DATE:
            break

    return pd.DataFrame(rows)


def generate_intersection_metadata() -> dict:
    return {
        "intersection_id":  INTERSECTION_ID,
        "name":             "Wadi Saqra Intersection",
        "city":             "Amman",
        "country":          "Jordan",
        "coordinates":      {"lat": 31.9739, "lon": 35.8894},
        "description":      "Major 4-way signalised intersection on Wadi Saqra Street near 3rd Circle, West Amman.",
        "camera": {
            "id":           "CAM-WS-01",
            "location":     "NE corner pole, 6 m height",
            "field_of_view_deg": 95,
            "frame_size":   "1920x1080",
            "fps_target":   10,
            "codec":        "H.264",
            "stream_type":  "RTSP (simulated via HTTP MJPEG for sandbox)",
        },
        "approaches": [
            {"id": "North", "street": "Wadi Saqra St (northbound exit)", "lanes": 6,
             "stop_line_px": {"x1": 320, "y1": 580, "x2": 760, "y2": 582},
             "queue_zone": {"x1": 0, "y1": 0, "x2": 1280, "y2": 250}},
            {"id": "South", "street": "Wadi Saqra St (southbound exit)", "lanes": 6,
             "stop_line_px": {"x1": 520, "y1": 140, "x2": 960, "y2": 142},
             "queue_zone": {"x1": 0, "y1": 470, "x2": 1280, "y2": 720}},
            {"id": "East",  "street": "3rd Circle approach",              "lanes": 5,
             "stop_line_px": {"x1": 980, "y1": 200, "x2": 982, "y2": 520},
             "queue_zone": {"x1": 900, "y1": 0, "x2": 1280, "y2": 720}},
            {"id": "West",  "street": "Rainbow St / University approach", "lanes": 5,
             "stop_line_px": {"x1": 298, "y1": 200, "x2": 300, "y2": 520},
             "queue_zone": {"x1": 0, "y1": 0, "x2": 380, "y2": 720}},
        ],
        "detectors": DETECTORS,
        "signal_phases": [
            {"phase_number": k, "green_s": v[0], "yellow_s": v[1],
             "all_red_s": v[2], "description": v[3]}
            for k, v in PHASES.items()
        ],
        "monitoring_zones": {
            "queue_spillback_threshold_vehicles": 4,
            "stalled_vehicle_threshold_s":        12,
            "congestion_speed_threshold_kmh":     15,
        },
        "coordinate_system": "pixel coordinates on 1280×720 processed frame",
    }


def generate_ground_truth_annotations() -> dict:
    """20 hand-labeled event windows drawn from the sandbox video set."""
    base = datetime(2026, 4, 15, 8, 0, 0)

    def mkev(etype, category, cam, track_id, ts_offset_min, frames, confidence, notes, queue_len=None):
        ts = base + timedelta(minutes=ts_offset_min)
        return {
            "event_type":     etype,
            "category":       category,
            "camera_id":      cam,
            "track_id":       track_id,
            "timestamp":      ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_start":    frames[0],
            "frame_end":      frames[1],
            "confidence":     confidence,
            "queue_length":   queue_len,
            "notes":          notes,
            "bounding_boxes": [],  # populate with actual coords when real video is annotated
        }

    events = [
        mkev("stalled_vehicle",       "possible traffic incident",   "CAM-01 NORTH", 12, 2,   [580, 680],   0.91, "Sedan stopped in right lane after apparent breakdown"),
        mkev("stalled_vehicle",       "possible traffic incident",   "CAM-02 SOUTH", 8,  18,  [1200, 1340], 0.87, "Taxi double-parked, blocking lane 3"),
        mkev("stalled_vehicle",       "possible traffic incident",   "CAM-03 EAST",  5,  34,  [420, 530],   0.83, "Motorcycle stalled, partially on shoulder"),
        mkev("stalled_vehicle",       "possible traffic incident",   "CAM-04 WEST",  19, 55,  [900, 1050],  0.89, "Bus stalled at stop line, lane 1"),
        mkev("stalled_vehicle",       "possible traffic incident",   "CAM-01 NORTH", 3,  71,  [2100, 2200], 0.78, "Pickup truck stopped midblock during green phase"),
        mkev("abnormal_stopping",     "abnormal traffic condition",  "CAM-02 SOUTH", 11, 6,   [310, 390],   0.85, "Car braked sharply from 60 km/h to 0 within 2 s"),
        mkev("abnormal_stopping",     "abnormal traffic condition",  "CAM-03 EAST",  7,  22,  [740, 820],   0.80, "SUV emergency stop, rear-end near-miss avoided"),
        mkev("abnormal_stopping",     "abnormal traffic condition",  "CAM-04 WEST",  14, 40,  [1560, 1640], 0.77, "Minibus sudden stop due to pedestrian crossing"),
        mkev("unexpected_trajectory", "abnormal traffic condition",  "CAM-01 NORTH", 21, 14,  [840, 940],   0.82, "Car turning right from straight-only lane"),
        mkev("unexpected_trajectory", "abnormal traffic condition",  "CAM-03 EAST",  9,  29,  [1100, 1200], 0.79, "Motorcycle weaving across 3 lanes"),
        mkev("unexpected_trajectory", "abnormal traffic condition",  "CAM-02 SOUTH", 15, 48,  [1900, 2000], 0.74, "Truck reversing at intersection after wrong turn"),
        mkev("queue_spillback",       "congestion event",            "CAM-01 NORTH", None, 3, [60,  200],   0.93, "North approach queue extending 120 m past stop zone", queue_len=9),
        mkev("queue_spillback",       "congestion event",            "CAM-02 SOUTH", None, 9, [380, 520],   0.91, "South approach queue 8 vehicles in zone", queue_len=8),
        mkev("queue_spillback",       "congestion event",            "CAM-03 EAST",  None, 16,[640, 780],   0.88, "East approach queue blocking adjacent intersection", queue_len=7),
        mkev("queue_spillback",       "congestion event",            "CAM-04 WEST",  None, 27,[1080,1220],  0.86, "West approach queue spilling onto Rainbow St", queue_len=11),
        mkev("sudden_congestion",     "congestion event",            "CAM-01 NORTH", None, 5, [200, 400],   0.90, "Total count rose from 6 to 18 vehicles in 30 s; avg speed < 10 km/h", queue_len=18),
        mkev("sudden_congestion",     "congestion event",            "CAM-02 SOUTH", None, 21,[840, 1040],  0.87, "Phase-change congestion surge on S approach"),
        mkev("sudden_congestion",     "congestion event",            "CAM-04 WEST",  None, 38,[1520,1720],  0.84, "Secondary incident caused downstream congestion"),
        mkev("accident_detected",     "accident alert",              "CAM-02 SOUTH", 6,  11,  [440, 560],   0.96, "Side-swipe collision between car and motorcycle at junction entry; debris visible"),
        mkev("accident_detected",     "accident alert",              "CAM-03 EAST",  4,  44,  [1760,1880],  0.94, "Rear-end collision; two vehicles stationary post-impact"),
    ]

    return {
        "intersection_id": INTERSECTION_ID,
        "annotation_version": "1.0",
        "annotated_by": "sandbox_generator",
        "total_events": len(events),
        "event_type_summary": {
            "stalled_vehicle":        5,
            "abnormal_stopping":      3,
            "unexpected_trajectory":  3,
            "queue_spillback":        4,
            "sudden_congestion":      3,
            "accident_detected":      2,
        },
        "events": events,
    }


DATA_DICTIONARY_MD = """# Data Dictionary – Wadi Saqra Traffic Intelligence Sandbox

## detector_dataset.csv
| Field | Type | Unit | Description |
|---|---|---|---|
| timestamp | datetime | YYYY-MM-DD HH:MM:SS | Start of the 15-minute aggregation window |
| intersection_id | string | – | Site identifier (WADI_SAQRA_01) |
| detector_id | string | – | Unique detector ID (e.g. DET-N01) |
| approach | string | – | Cardinal approach: North / South / East / West |
| lane | int | – | Lane number within the approach (1 = leftmost) |
| movement | string | – | Permitted movement: left_only / left_straight / straight / straight_right / right_only / u_turn |
| vehicle_count | int | vehicles / 15 min | Aggregated vehicle count passing the detector in the window |
| occupancy_pct | float | % | Percentage of time the detector loop was occupied |
| speed_avg_kmh | float | km/h | Average vehicle speed through the detector |
| rain | int | 0/1 | Whether it was raining during this interval |
| temp_c | float | °C | Ambient temperature |

## signal_timing_log.csv
| Field | Type | Unit | Description |
|---|---|---|---|
| timestamp | datetime | YYYY-MM-DD HH:MM:SS | Exact moment the state began |
| intersection_id | string | – | Site identifier |
| phase_number | int | – | Signal phase (1–4) |
| signal_state | string | – | GREEN_ON / YELLOW_ON / ALL_RED |
| duration_s | int | seconds | How long this state lasted |
| phase_desc | string | – | Plain-language phase description |
| cycle_time_s | int | seconds | Nominal cycle length for reference |

## intersection_metadata.json
| Field | Description |
|---|---|
| intersection_id | Unique site key |
| coordinates | WGS-84 lat/lon of intersection centre |
| camera | Camera hardware specs and stream parameters |
| approaches | Per-approach geometry: lanes, stop line pixel coords, queue zone rect |
| detectors | All 22 detector specs linked to approach + lane |
| signal_phases | Phase timing configuration |
| monitoring_zones | Algorithm sensitivity thresholds |

## ground_truth_annotations.json
| Field | Description |
|---|---|
| event_type | Machine-readable event class (matches detection module output) |
| category | Human-readable event grouping |
| camera_id | Which camera captured the event |
| track_id | Vehicle tracker ID (null for area events) |
| timestamp | ISO-8601 event timestamp |
| frame_start / frame_end | Video frame range for the event window |
| confidence | Annotator confidence score 0–1 |
| queue_length | Number of vehicles in queue (congestion events only) |
| bounding_boxes | List of XYXY bounding boxes per frame (empty until real-video annotation) |
| notes | Free-text annotation note |
"""


def main():
    print("Generating Phase 1 traffic data sandbox for Wadi Saqra intersection…")

    print("  [1/5] Detector dataset (22 detectors × 14 days × 15-min)…", end=" ", flush=True)
    df_det = generate_detector_dataset()
    df_det.to_csv(OUT / "detector_dataset.csv", index=False)
    print(f"done  ({len(df_det):,} rows)")

    print("  [2/5] Signal timing log…", end=" ", flush=True)
    df_sig = generate_signal_timing_log()
    df_sig.to_csv(OUT / "signal_timing_log.csv", index=False)
    print(f"done  ({len(df_sig):,} rows)")

    print("  [3/5] Intersection metadata…", end=" ", flush=True)
    meta = generate_intersection_metadata()
    (OUT / "intersection_metadata.json").write_text(json.dumps(meta, indent=2))
    print("done")

    print("  [4/5] Ground truth annotations…", end=" ", flush=True)
    gt = generate_ground_truth_annotations()
    (OUT / "ground_truth_annotations.json").write_text(json.dumps(gt, indent=2))
    print(f"done  ({gt['total_events']} labeled events)")

    print("  [5/5] Data dictionary…", end=" ", flush=True)
    (OUT / "data_dictionary.md").write_text(DATA_DICTIONARY_MD.strip())
    print("done")

    print(f"\nAll sandbox files written to: {OUT.resolve()}")
    print("\nSummary:")
    print(f"  detector_dataset.csv          {len(df_det):>8,} rows")
    print(f"  signal_timing_log.csv         {len(df_sig):>8,} rows")
    print(f"  intersection_metadata.json    22 detectors, 4 approaches, 4 phases")
    print(f"  ground_truth_annotations.json {gt['total_events']} labeled events")
    print(f"  data_dictionary.md            all field definitions")


if __name__ == "__main__":
    main()
