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
