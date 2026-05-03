# Data Dictionary – Wadi Saqra Traffic Intelligence Sandbox

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