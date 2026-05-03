import argparse
import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path
from threading import Lock, Thread

import cv2
from aiohttp import web
from ultralytics import YOLO

from tracker import ByteTrackLikeTracker, bbox_iou

try:
    from storage import save_event as _db_save_event, save_performance as _db_save_perf
    _STORAGE_OK = True
except ImportError:
    _STORAGE_OK = False
    def _db_save_event(_e): pass
    def _db_save_perf(_s):  pass


STREET_LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
}

TRACKED_LABELS = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
VEHICLE_EVENT_LABELS = {"car", "motorcycle", "bus", "truck"}
COUNT_KEYS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic_light",
    "traffic_sign",
    "street_fixture",
    "other",
    "total",
)
COUNT_KEY_BY_LABEL = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "stop sign": "traffic_sign",
    "fire hydrant": "street_fixture",
    "parking meter": "street_fixture",
}

BOX_COLORS = {
    "person": (248, 113, 113),
    "bicycle": (251, 191, 36),
    "car": (62, 207, 255),
    "motorcycle": (255, 181, 46),
    "bus": (34, 217, 122),
    "truck": (245, 110, 95),
    "traffic light": (167, 139, 250),
    "stop sign": (244, 63, 94),
    "fire hydrant": (251, 146, 60),
    "parking meter": (148, 163, 184),
}

CAMERA_LOCATIONS = {
    "CAM-01 NORTH": "North approach",
    "CAM-02 SOUTH": "South approach",
    "CAM-03 EAST": "East approach",
    "CAM-04 WEST": "West approach",
    "WTS CAM 05": "Primary junction overview",
}

# Focus modes: each camera has a specialist role
FOCUS_META = {
    "traffic_flow": {
        "label":      "FLOW MONITOR",
        "header_bgr": (120, 60, 10),   # dark blue
        "header_txt": (255, 220, 100),
    },
    "stop_stall": {
        "label":      "STOP & STALL",
        "header_bgr": (20, 100, 180),  # dark orange
        "header_txt": (80, 220, 255),
    },
    "general": {
        "label":      "FULL MONITOR",
        "header_bgr": (20, 80, 20),    # dark green
        "header_txt": (100, 255, 160),
    },
    "accident": {
        "label":      "ACCIDENT DETECTION",
        "header_bgr": (10, 10, 100),   # dark red
        "header_txt": (80, 80, 255),
    },
}

CONFIG_PATH = Path(__file__).with_name("camera_thresholds.json")


def default_model_path():
    detection_dir = Path(__file__).resolve().parents[1]
    for candidate in (
        detection_dir / "model" / "yolov12s.pt",
        detection_dir / "model" / "yolov12n.pt",
        detection_dir / "model" / "yolov8m.pt",
        detection_dir / "model" / "yolov8s.pt",
        detection_dir / "yolov8n.pt",
        Path(__file__).with_name("yolov8n.pt"),
    ):
        if candidate.exists():
            return str(candidate)
    return str(Path(__file__).with_name("yolov8n.pt"))


def empty_counts():
    return {key: 0 for key in COUNT_KEYS}


def count_key_for_label(label):
    return COUNT_KEY_BY_LABEL.get(label, "other")


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_camera_config(camera_id):
    defaults = {
        "tracker": {
            "high_thresh": 0.4,
            "low_thresh": 0.15,
            "match_iou_thresh": 0.28,
            "low_match_iou_thresh": 0.18,
            "new_track_thresh": 0.52,
            "track_buffer_frames": 28,
            "stationary_speed_px_s": 8.0,
        },
        "events": {
            "stalled_age_s": 5.0,
            "stalled_stationary_s": 5.0,
            "abnormal_prev_speed_px_s": 28.0,
            "abnormal_current_speed_px_s": 8.0,
            "abnormal_stationary_s": 1.5,
            "trajectory_path_px": 90.0,
            "trajectory_angle_deg": 115.0,
            "queue_speed_px_s": 8.0,
            "queue_min_tracks": 4,
            "queue_cooldown_s": 6.0,
            "congestion_history_samples": 6,
            "congestion_min_total": 8,
            "congestion_delta": 5,
            "congestion_avg_speed_px_s": 18.0,
        },
        "zones": {
            "queue": {
                "mode": "band",
                "axis": "y",
                "side": "min",
                "fraction": 0.35,
            }
        },
    }
    try:
        payload = json.loads(CONFIG_PATH.read_text())
    except Exception:
        return defaults
    merged = _deep_merge(defaults, payload.get("defaults", {}))
    return _deep_merge(merged, payload.get(camera_id, {}))


def angle_between(vec_a, vec_b):
    mag_a = math.hypot(*vec_a)
    mag_b = math.hypot(*vec_b)
    if mag_a < 1e-6 or mag_b < 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]) / (mag_a * mag_b)))
    return math.degrees(math.acos(dot))


class YoloVideoStream:
    def __init__(
        self,
        video_path: str,
        model_path: str,
        width: int,
        height: int,
        fps: float,
        confidence: float,
        infer_every: int,
        camera_id: str,
        imgsz: int,
        iou: float,
        max_det: int,
        device,
        street_only: bool,
        demo_accident: bool,
        demo_accident_start_s: float,
        demo_accident_duration_s: float,
        camera_focus: str = "general",
        disable_accident_detection: bool = False,
    ) -> None:
        self.video_path = Path(video_path)
        self.model_path = str(model_path)
        self.model = YOLO(model_path)
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.confidence = confidence
        self.infer_every = max(1, infer_every)
        self.imgsz = max(320, int(imgsz))
        self.iou = iou
        self.max_det = max(1, int(max_det))
        self.device = device or None
        self.street_only = street_only
        self.disable_accident_detection = disable_accident_detection
        self.demo_accident = demo_accident and not disable_accident_detection
        self.demo_accident_start_s = max(0.0, demo_accident_start_s)
        self.demo_accident_duration_s = max(1.0, demo_accident_duration_s)
        self._demo_accident_emitted = False
        self.camera_id = camera_id
        self.source_name = self.video_path.name
        self.location_name = CAMERA_LOCATIONS.get(camera_id, camera_id)
        self.camera_config = load_camera_config(camera_id)
        tracker_cfg = self.camera_config["tracker"]
        tracker_cfg["track_buffer_frames"] = max(6, int(tracker_cfg["track_buffer_frames"]))
        self.tracker = ByteTrackLikeTracker(**tracker_cfg)
        self.event_cfg = self.camera_config["events"]
        self.zone_cfg = self.camera_config["zones"]
        self.camera_focus = camera_focus

        # ── Per-focus threshold overrides ────────────────────────────────────
        if camera_focus == "traffic_flow":
            # More sensitive to congestion and queue events
            self.event_cfg.update({
                "queue_min_tracks":          3,
                "congestion_min_total":      5,
                "congestion_delta":          3,
                "congestion_avg_speed_px_s": 22.0,
                "queue_cooldown_s":          4.0,
            })
            self.street_only = True

        elif camera_focus == "stop_stall":
            # Faster stall and abnormal-stop detection
            self.event_cfg.update({
                "stalled_age_s":               3.0,
                "stalled_stationary_s":        3.0,
                "abnormal_prev_speed_px_s":    20.0,
                "abnormal_current_speed_px_s": 4.0,
                "abnormal_stationary_s":       1.0,
                "queue_cooldown_s":            3.0,
            })
            self.street_only = True

        elif camera_focus == "accident":
            # Fastest event emission; all classes visible
            self.event_cfg["queue_cooldown_s"] = 2.0
            self.street_only = False   # detect everything (pedestrians matter too)

        # Accident-camera-specific alert threshold (lower = more sensitive)
        self._accident_threshold = 55.0 if camera_focus == "accident" else 80.0
        # Track IDs involved in detected collision for red-box highlight
        self._collision_track_ids: set = set()

        self.start_time = time.time()

        self.artifact_dir = Path(__file__).resolve().parents[1] / "logs" / "events" / self.camera_id.lower().replace(" ", "_")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._jpeg = b""
        self._stats = {
            "source_video": self.source_name,
            "camera_id": self.camera_id,
            "frame_index": 0,
            "counts": empty_counts(),
            "fps": self.fps,
            "resolution": {"width": self.width, "height": self.height},
            "updated_at": time.time(),
            "frames_processed": 0,
            "frames_dropped": 0,
            "stream_uptime_s": 0,
            "reconnect_count": 0,
            "active_tracks_total": 0,
            "model": Path(self.model_path).name,
            "confidence": self.confidence,
            "imgsz": self.imgsz,
            "street_only": self.street_only,
            "demo_accident_enabled": self.demo_accident,
            "demo_accident_active": False,
            "camera_focus": self.camera_focus,
            "accident_detection_enabled": not self.disable_accident_detection,
        }
        self._last_detections = []
        self._events = deque(maxlen=100)
        self._event_cooldowns = {}
        self._recent_frames = deque(maxlen=max(12, int(self.fps * 4)))
        self._count_history = deque(maxlen=max(12, int(self.fps * 8)))
        self._accident_risk = 0.0
        self._accident_alert = False
        self._scenario = "normal"
        self._running = True
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def latest_stats(self) -> dict:
        with self._lock:
            return {
                "source_video": self._stats["source_video"],
                "camera_id": self._stats["camera_id"],
                "frame_index": self._stats["frame_index"],
                "counts": dict(self._stats["counts"]),
                "fps": self._stats["fps"],
                "resolution": dict(self._stats["resolution"]),
                "updated_at": self._stats["updated_at"],
                "frames_processed": self._stats["frames_processed"],
                "frames_dropped": self._stats["frames_dropped"],
                "stream_uptime_s": self._stats["stream_uptime_s"],
                "reconnect_count": self._stats["reconnect_count"],
                "active_tracks_total": self._stats["active_tracks_total"],
                "model": self._stats["model"],
                "confidence": self._stats["confidence"],
                "imgsz": self._stats["imgsz"],
                "street_only": self._stats["street_only"],
                "demo_accident_enabled": self._stats["demo_accident_enabled"],
                "demo_accident_active": self._stats["demo_accident_active"],
                "accident_risk": self._accident_risk,
                "accident_alert": self._accident_alert,
                "scenario": self._scenario,
            }

    def latest_events(self):
        with self._lock:
            return list(self._events)

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")
        return capture

    def _run_yolo(self, frame):
        predict_kwargs = {
            "verbose": False,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
        }
        if self.device:
            predict_kwargs["device"] = self.device
        results = self.model(frame, **predict_kwargs)[0]
        detections = []
        counts = empty_counts()
        model_names = getattr(self.model, "names", {}) or {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if self.street_only and cls_id not in STREET_LABELS:
                continue

            label = STREET_LABELS.get(cls_id, str(model_names.get(cls_id, f"class {cls_id}")).lower())
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]
            detections.append(
                {
                    "class_id": cls_id,
                    "label": label,
                    "score": score,
                    "bbox": bbox,
                    "track_id": None,
                    "trackable": label in TRACKED_LABELS,
                }
            )
            counts[count_key_for_label(label)] += 1
            counts["total"] += 1

        return detections, counts

    def _queue_zone_name(self):
        name = self.camera_id.upper()
        if "NORTH" in name:
            return "north ingress"
        if "SOUTH" in name:
            return "south ingress"
        if "EAST" in name:
            return "east ingress"
        if "WEST" in name:
            return "west ingress"
        return "primary ingress"

    def _in_queue_zone(self, point):
        queue_cfg = self.zone_cfg.get("queue", {})
        if queue_cfg.get("mode") == "band":
            axis = queue_cfg.get("axis", "y")
            side = queue_cfg.get("side", "min")
            fraction = float(queue_cfg.get("fraction", 0.35))
            coord = point[0] if axis == "x" else point[1]
            extent = self.width if axis == "x" else self.height
            boundary = extent * fraction
            return coord <= boundary if side == "min" else coord >= boundary

        x, y = point
        name = self.camera_id.upper()
        if "NORTH" in name:
            return y < self.height * 0.35
        if "SOUTH" in name:
            return y > self.height * 0.65
        if "EAST" in name:
            return x > self.width * 0.68
        if "WEST" in name:
            return x < self.width * 0.32
        return y < self.height * 0.35

    def _location_for_track(self, track):
        x, y = track.centroid
        h_band = "center"
        v_band = "middle"
        if x < self.width * 0.33:
            h_band = "west"
        elif x > self.width * 0.66:
            h_band = "east"
        if y < self.height * 0.33:
            v_band = "north"
        elif y > self.height * 0.66:
            v_band = "south"
        return f"{self.location_name} · {v_band}-{h_band}"

    def _capture_artifacts(self, event_key, frame):
        safe_key = event_key.replace(":", "-").replace(" ", "_")
        snapshot_path = self.artifact_dir / f"{safe_key}.jpg"
        clip_path = self.artifact_dir / f"{safe_key}.mp4"

        cv2.imwrite(str(snapshot_path), frame)

        try:
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            for _, buffered_frame in list(self._recent_frames):
                writer.write(buffered_frame)
            writer.release()
        except Exception:
            clip_path = None

        return str(snapshot_path), (str(clip_path) if clip_path else None)

    def _emit_event(self, event_type, category, location, confidence, frame, track=None, queue_length=None):
        now = time.time()
        track_id = track.track_id if track else None
        cooldown_key = f"{event_type}:{track_id or 'global'}"
        cooldown_seconds = float(self.event_cfg.get("queue_cooldown_s", 6.0))
        if now - self._event_cooldowns.get(cooldown_key, 0) < cooldown_seconds:
            return
        self._event_cooldowns[cooldown_key] = now

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        artifact_key = f"{timestamp}_{event_type}_{track_id or 'site'}"
        snapshot_path, clip_path = self._capture_artifacts(artifact_key, frame)

        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "category": category,
            "location": location,
            "confidence": round(float(confidence), 3),
            "camera_id": self.camera_id,
            "source_video": self.source_name,
            "track_id": track_id,
            "snapshot_image": snapshot_path,
            "clip_path": clip_path,
            "queue_length": queue_length,
        }

        with self._lock:
            self._events.appendleft(event)

        # Persist to SQLite
        _db_save_event(event)

    def _track_direction_change(self, track):
        points = [point for _, point in track.history]
        if len(points) < 4:
            return 0.0, 0.0
        vec_a = (points[-3][0] - points[-4][0], points[-3][1] - points[-4][1])
        vec_b = (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
        return angle_between(vec_a, vec_b), math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])

    def _analyze_events(self, frame, now_ts, counts):
        self._count_history.append((now_ts, counts["total"]))

        active_tracks = [
            track for track in self.tracker.active_tracks()
            if now_ts - track.last_seen_ts <= (2.0 / max(1.0, self.fps)) + 1.0
            and track.label in VEHICLE_EVENT_LABELS
        ]

        for track in active_tracks:
            age = now_ts - track.created_ts
            location = self._location_for_track(track)
            speeds = list(track.speed_history)
            current_speed = speeds[-1] if speeds else 0.0

            if (
                age >= float(self.event_cfg["stalled_age_s"])
                and track.stationary_seconds >= float(self.event_cfg["stalled_stationary_s"])
                and "stalled_vehicle" not in track.flags
            ):
                self._emit_event(
                    "stalled_vehicle",
                    "possible traffic incident",
                    location,
                    min(0.99, 0.55 + track.stationary_seconds / 10.0),
                    frame,
                    track=track,
                )
                track.flags.add("stalled_vehicle")

            if len(speeds) >= 4:
                previous_speed = sum(speeds[-4:-1]) / 3.0
                if (
                    previous_speed > float(self.event_cfg["abnormal_prev_speed_px_s"])
                    and current_speed < float(self.event_cfg["abnormal_current_speed_px_s"])
                    and track.stationary_seconds >= float(self.event_cfg["abnormal_stationary_s"])
                    and "abnormal_stopping" not in track.flags
                ):
                    self._emit_event(
                        "abnormal_stopping",
                        "abnormal traffic condition",
                        location,
                        min(0.97, 0.52 + (previous_speed - current_speed) / 60.0),
                        frame,
                        track=track,
                    )
                    track.flags.add("abnormal_stopping")

            turn_angle, path_length = self._track_direction_change(track)
            if (
                path_length > float(self.event_cfg["trajectory_path_px"])
                and turn_angle > float(self.event_cfg["trajectory_angle_deg"])
                and "unexpected_trajectory" not in track.flags
            ):
                self._emit_event(
                    "unexpected_trajectory",
                    "abnormal traffic condition",
                    location,
                    min(0.96, 0.5 + turn_angle / 220.0),
                        frame,
                        track=track,
                    )
                track.flags.add("unexpected_trajectory")

        queued_tracks = [
            track for track in active_tracks
            if self._in_queue_zone(track.centroid)
            and track.latest_speed < float(self.event_cfg["queue_speed_px_s"])
        ]
        if len(queued_tracks) >= int(self.event_cfg["queue_min_tracks"]):
            queue_conf = min(0.98, 0.52 + len(queued_tracks) / 12.0)
            self._emit_event(
                "queue_spillback",
                "congestion event",
                f"{self.location_name} · {self._queue_zone_name()}",
                queue_conf,
                frame,
                queue_length=len(queued_tracks),
            )

        if len(self._count_history) >= int(self.event_cfg["congestion_history_samples"]):
            recent_total = counts["total"]
            earlier = [item[1] for item in list(self._count_history)[:-1]]
            baseline = sum(earlier) / max(1, len(earlier))
            if recent_total >= max(float(self.event_cfg["congestion_min_total"]), baseline + float(self.event_cfg["congestion_delta"])):
                avg_speed = 0.0
                speed_samples = []
                for track in active_tracks:
                    if track.speed_history:
                        speed_samples.append(track.latest_speed)
                if speed_samples:
                    avg_speed = sum(speed_samples) / len(speed_samples)
                if avg_speed < float(self.event_cfg["congestion_avg_speed_px_s"]):
                    self._emit_event(
                        "sudden_congestion",
                        "congestion event",
                        self.location_name,
                        min(0.97, 0.5 + (recent_total - baseline) / 20.0),
                        frame,
                        queue_length=recent_total,
                    )

        if self.disable_accident_detection:
            with self._lock:
                self._accident_risk = 0.0
                self._accident_alert = False
                self._collision_track_ids = set()
        else:
            # ── Accident risk scoring (all cameras) ───────────────────────
            risk = self._compute_accident_risk(active_tracks)
            alert = risk >= self._accident_threshold

            with self._lock:
                self._accident_risk = risk
                self._accident_alert = alert

            if alert:
                self._emit_event(
                    "accident_detected",
                    "accident alert",
                    self.location_name,
                    min(0.99, 0.50 + risk / 200.0),
                    frame,
                )

            # ── Physics-based real accident detection (accident camera only) ──
            if self.camera_focus == "accident":
                self._detect_real_accident(active_tracks, frame)

    def _compute_accident_risk(self, active_tracks):
        """
        Returns 0-100 risk score by analysing pairwise vehicle proximity
        and closing velocity.  Factors:
          - Bounding-box overlap (IoU > 0 → near-certain collision)
          - Centroid proximity relative to vehicle size
          - Relative closing speed (dot product of displacement vectors)
          - Scenario multiplier (rain / night / rush_hour raise baseline)
        """
        vehicle_tracks = [t for t in active_tracks if t.label in VEHICLE_EVENT_LABELS]
        if len(vehicle_tracks) < 2:
            return 0.0

        scenario_mult = {
            "heavy_rain": 1.45,
            "night": 1.30,
            "rush_hour": 1.20,
            "accident_zone": 1.60,
        }.get(self._scenario, 1.0)

        max_risk = 0.0

        for i, ta in enumerate(vehicle_tracks):
            for j, tb in enumerate(vehicle_tracks):
                if i >= j:
                    continue

                iou = bbox_iou(ta.bbox, tb.bbox)
                if iou > 0.05:
                    risk = min(100.0, 65.0 + iou * 350.0)
                    max_risk = max(max_risk, risk)
                    continue

                cx_a, cy_a = ta.centroid
                cx_b, cy_b = tb.centroid
                dist = math.hypot(cx_a - cx_b, cy_a - cy_b)

                w_a = ta.bbox[2] - ta.bbox[0]
                h_a = ta.bbox[3] - ta.bbox[1]
                w_b = tb.bbox[2] - tb.bbox[0]
                h_b = tb.bbox[3] - tb.bbox[1]
                avg_size = (math.hypot(w_a, h_a) + math.hypot(w_b, h_b)) / 2.0

                if avg_size < 1.0:
                    continue
                norm_dist = dist / avg_size

                if norm_dist > 3.0:
                    continue

                proximity_risk = max(0.0, (3.0 - norm_dist) / 3.0) * 55.0

                history_a = [p for _, p in ta.history]
                history_b = [p for _, p in tb.history]
                closing_bonus = 0.0
                if len(history_a) >= 2 and len(history_b) >= 2:
                    vel_a = (history_a[-1][0] - history_a[-2][0], history_a[-1][1] - history_a[-2][1])
                    vel_b = (history_b[-1][0] - history_b[-2][0], history_b[-1][1] - history_b[-2][1])
                    ab = (cx_b - cx_a, cy_b - cy_a)
                    denom = math.hypot(*ab)
                    if denom > 0:
                        closing_a = (vel_a[0] * ab[0] + vel_a[1] * ab[1]) / denom
                        closing_b = -(vel_b[0] * ab[0] + vel_b[1] * ab[1]) / denom
                        closing_speed = max(0.0, closing_a) + max(0.0, closing_b)
                        closing_bonus = min(35.0, closing_speed * 1.2)

                raw = proximity_risk + closing_bonus
                max_risk = max(max_risk, min(100.0, raw * scenario_mult))

        return round(max_risk, 1)

    def _detect_real_accident(self, active_tracks, frame):
        """
        Physics-based collision detection for the accident-focus camera.
        Three independent tests:
          1. Bounding-box overlap (IoU > 0.02) — vehicles physically touching.
          2. High-speed approach — closing velocity > 18 px/s while within
             1.5× average vehicle diameter.
          3. Impact cluster — vehicle that was fast (>20 px/s) is now stopped
             AND is surrounded by ≥2 other stopped vehicles within 2.5× its size.
        Emits 'accident_detected' events and updates _collision_track_ids so
        _draw_frame() can highlight the involved vehicles with red cross-boxes.
        """
        vehicle_tracks = [t for t in active_tracks if t.label in VEHICLE_EVENT_LABELS]
        if len(vehicle_tracks) < 2:
            with self._lock:
                self._collision_track_ids = set()
            return

        collision_ids: set = set()

        for i, ta in enumerate(vehicle_tracks):
            for j, tb in enumerate(vehicle_tracks):
                if i >= j:
                    continue

                # ── Test 1: direct bounding-box overlap ───────────────────
                iou = bbox_iou(ta.bbox, tb.bbox)
                if iou > 0.02:
                    collision_ids.update([ta.track_id, tb.track_id])
                    self._emit_event(
                        "accident_detected",
                        "accident alert",
                        self._location_for_track(ta),
                        min(0.99, 0.72 + iou * 1.4),
                        frame,
                        track=ta,
                    )
                    continue

                # ── Test 2: high-speed approach ───────────────────────────
                cx_a, cy_a = ta.centroid
                cx_b, cy_b = tb.centroid
                dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
                w_a = ta.bbox[2] - ta.bbox[0]
                h_a = ta.bbox[3] - ta.bbox[1]
                w_b = tb.bbox[2] - tb.bbox[0]
                h_b = tb.bbox[3] - tb.bbox[1]
                avg_size = (math.hypot(w_a, h_a) + math.hypot(w_b, h_b)) / 2.0
                if avg_size < 1.0:
                    continue

                hist_a = [p for _, p in ta.history]
                hist_b = [p for _, p in tb.history]
                if len(hist_a) >= 2 and len(hist_b) >= 2:
                    vel_a = (hist_a[-1][0] - hist_a[-2][0], hist_a[-1][1] - hist_a[-2][1])
                    vel_b = (hist_b[-1][0] - hist_b[-2][0], hist_b[-1][1] - hist_b[-2][1])
                    ab = (cx_b - cx_a, cy_b - cy_a)
                    denom = math.hypot(*ab)
                    if denom > 0:
                        closing_a = (vel_a[0] * ab[0] + vel_a[1] * ab[1]) / denom
                        closing_b = -(vel_b[0] * ab[0] + vel_b[1] * ab[1]) / denom
                        closing_speed = max(0.0, closing_a) + max(0.0, closing_b)
                        if closing_speed > 18.0 and dist < avg_size * 1.5:
                            collision_ids.update([ta.track_id, tb.track_id])
                            self._emit_event(
                                "accident_detected",
                                "accident alert",
                                self._location_for_track(ta),
                                min(0.97, 0.56 + closing_speed / 80.0),
                                frame,
                                track=ta,
                            )

        # ── Test 3: post-impact cluster ───────────────────────────────────
        for ta in vehicle_tracks:
            speeds = list(ta.speed_history)
            if len(speeds) < 5:
                continue
            prev_speed = max(speeds[max(0, len(speeds) - 6):len(speeds) - 1])
            curr_speed = ta.latest_speed
            if prev_speed < 20.0 or curr_speed > 6.0:
                continue
            cx_a, cy_a = ta.centroid
            w_a = ta.bbox[2] - ta.bbox[0]
            h_a = ta.bbox[3] - ta.bbox[1]
            size_a = math.hypot(w_a, h_a)
            nearby_stopped = [
                tb for tb in vehicle_tracks
                if tb.track_id != ta.track_id
                and tb.latest_speed < 6.0
                and math.hypot(tb.centroid[0] - cx_a, tb.centroid[1] - cy_a) < size_a * 2.5
            ]
            if len(nearby_stopped) >= 2:
                collision_ids.add(ta.track_id)
                for tb in nearby_stopped:
                    collision_ids.add(tb.track_id)
                self._emit_event(
                    "accident_detected",
                    "accident alert",
                    self._location_for_track(ta),
                    min(0.96, 0.65 + len(nearby_stopped) / 10.0),
                    frame,
                    track=ta,
                )

        with self._lock:
            self._collision_track_ids = collision_ids
            if collision_ids:
                self._scenario = "accident_zone"

    def set_scenario(self, scenario: str):
        self._scenario = scenario

    def _demo_accident_active(self, frame_index):
        if not self.demo_accident:
            return False
        elapsed = time.time() - self.start_time
        period = max(20.0, self.demo_accident_start_s + self.demo_accident_duration_s + 12.0)
        position = elapsed % period
        return self.demo_accident_start_s <= position < self.demo_accident_start_s + self.demo_accident_duration_s

    def _demo_accident_targets(self, detections):
        vehicle_detections = [
            item for item in detections
            if item.get("label") in VEHICLE_EVENT_LABELS
        ]
        return sorted(
            vehicle_detections,
            key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]),
            reverse=True,
        )[:2]

    def _draw_demo_accident_overlay(self, display, detections):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 58), (12, 18, 28), -1)
        cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)
        cv2.putText(
            display,
            "ACCIDENT EXAMPLE  |  SIMULATED COLLISION ALERT",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )

        targets = self._demo_accident_targets(detections)
        if targets:
            x1 = min(item["bbox"][0] for item in targets)
            y1 = min(item["bbox"][1] for item in targets)
            x2 = max(item["bbox"][2] for item in targets)
            y2 = max(item["bbox"][3] for item in targets)
        else:
            zone_w = int(self.width * 0.24)
            zone_h = int(self.height * 0.18)
            x1 = int(self.width * 0.50 - zone_w / 2)
            y1 = int(self.height * 0.56 - zone_h / 2)
            x2 = x1 + zone_w
            y2 = y1 + zone_h

        pad = 12
        x1 = max(0, x1 - pad)
        y1 = max(60, y1 - pad)
        x2 = min(self.width - 1, x2 + pad)
        y2 = min(self.height - 1, y2 + pad)
        cv2.rectangle(display, (x1, y1), (x2, y2), (40, 40, 255), 4)
        cv2.line(display, (x1, y1), (x2, y2), (40, 40, 255), 2)
        cv2.line(display, (x2, y1), (x1, y2), (40, 40, 255), 2)
        cv2.rectangle(display, (x1, max(60, y1 - 32)), (min(self.width - 1, x1 + 270), y1), (40, 40, 255), -1)
        cv2.putText(
            display,
            "DEMO ACCIDENT",
            (x1 + 8, max(82, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def _maybe_emit_demo_accident(self, frame, frame_index, detections):
        if not self._demo_accident_active(frame_index) or self._demo_accident_emitted:
            return
        snapshot = self._draw_demo_accident_overlay(frame.copy(), detections)
        self._emit_event(
            "demo_accident",
            "accident example",
            f"{self.location_name} · simulated collision zone",
            0.96,
            snapshot,
        )
        self._demo_accident_emitted = True

    def _draw_frame(self, frame, frame_index: int, detections, counts, demo_accident_active=False):
        display = frame.copy()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # ── Focus-mode coloured header ─────────────────────────────────
        fm = FOCUS_META.get(self.camera_focus, FOCUS_META["general"])
        hdr_bgr = fm["header_bgr"]
        hdr_txt = fm["header_txt"]
        focus_label = fm["label"]
        cv2.rectangle(display, (14, 14), (480, 82), hdr_bgr, -1)
        cv2.putText(
            display,
            f"{self.camera_id}  [{focus_label}]",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            hdr_txt,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            timestamp,
            (24, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        signals = counts["traffic_light"] + counts["traffic_sign"]
        stats_line_1 = (
            f"People {counts['person']}   Bike {counts['bicycle']}   "
            f"Cars {counts['car']}   Moto {counts['motorcycle']}"
        )
        stats_line_2 = (
            f"Bus {counts['bus']}   Truck {counts['truck']}   "
            f"Signals {signals}   Other {counts['other']}   Total {counts['total']}"
        )
        cv2.rectangle(display, (14, 92), (650, 150), (0, 0, 0), -1)
        cv2.putText(
            display,
            stats_line_1,
            (24, 113),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 201, 64),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            stats_line_2,
            (24, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 201, 64),
            1,
            cv2.LINE_AA,
        )

        collision_ids = set() if self.disable_accident_detection else self._collision_track_ids
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            color = BOX_COLORS.get(item["label"], (255, 138, 101))
            label = f"{item['label']} {item['score']:.2f}"
            track_id = item.get("track_id")
            if track_id is None:
                best_iou = 0.0
                for track in self.tracker.active_tracks():
                    if track.label != item["label"]:
                        continue
                    overlap = bbox_iou(track.bbox, item["bbox"])
                    if overlap > best_iou:
                        best_iou = overlap
                        track_id = track.track_id
            if track_id is not None:
                label = f"T{track_id} {label}"

            # ── Red cross-box for collision vehicles (accident camera) ──
            is_collision = (
                self.camera_focus == "accident"
                and track_id is not None
                and track_id in collision_ids
            )
            if is_collision:
                cv2.rectangle(display, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
                cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(display, (x2, y1), (x1, y2), (0, 0, 255), 2)
                color = (0, 0, 255)
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(display, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                display,
                label,
                (x1 + 5, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (3, 16, 24),
                2,
                cv2.LINE_AA,
            )

        if demo_accident_active:
            self._draw_demo_accident_overlay(display, detections)

        if not self.disable_accident_detection:
            # ── Accident risk HUD ──────────────────────────────────────────
            risk = self._accident_risk
            alert = self._accident_alert
            bar_x, bar_y = self.width - 230, 14
            bar_w, bar_h = 210, 28
            fill_w = int(bar_w * risk / 100.0)
            if risk >= 80:
                risk_color = (0, 0, 230)
            elif risk >= 50:
                risk_color = (0, 140, 255)
            else:
                risk_color = (0, 200, 80)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
            if fill_w > 0:
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), risk_color, -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            cv2.putText(
                display,
                f"RISK {risk:.0f}%",
                (bar_x + 8, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if alert:
                pulse = int(time.time() * 4) % 2 == 0
                if pulse:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)
                cv2.putText(
                    display,
                    "ACCIDENT DETECTED",
                    (self.width // 2 - 200, self.height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        return display

    def _run(self) -> None:
        capture = self._open_capture()
        frame_index = 0
        _last_perf_save = time.time()

        while self._running:
            ok, frame = capture.read()
            if not ok:
                capture.release()
                with self._lock:
                    self._stats["reconnect_count"] += 1
                capture = self._open_capture()
                frame_index = 0
                continue

            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            now_ts = time.time()
            self._recent_frames.append((now_ts, frame.copy()))

            if frame_index % self.infer_every == 0:
                self._last_detections, counts = self._run_yolo(frame)
                trackable_detections = [
                    item for item in self._last_detections
                    if item.get("trackable")
                ]
                self.tracker.update(trackable_detections, frame_index, now_ts)
                self._analyze_events(frame.copy(), now_ts, counts)
            else:
                counts = empty_counts()
                for item in self._last_detections:
                    counts[count_key_for_label(item["label"])] += 1
                    counts["total"] += 1

            demo_accident_active = self._demo_accident_active(frame_index)
            self._maybe_emit_demo_accident(frame.copy(), frame_index, self._last_detections)
            annotated = self._draw_frame(
                frame,
                frame_index,
                self._last_detections,
                counts,
                demo_accident_active=demo_accident_active,
            )
            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            with self._lock:
                self._stats["frames_processed"] += 1
                self._stats["stream_uptime_s"] = int(now_ts - self.start_time)
                self._stats["active_tracks_total"] = len(self.tracker.active_tracks())
                self._stats["demo_accident_active"] = demo_accident_active
            if ok:
                with self._lock:
                    self._jpeg = encoded.tobytes()
                    self._stats.update(
                        {
                            "source_video": self.source_name,
                            "camera_id": self.camera_id,
                            "frame_index": frame_index,
                            "counts": counts,
                            "fps": self.fps,
                            "resolution": {"width": self.width, "height": self.height},
                            "updated_at": now_ts,
                            "model": Path(self.model_path).name,
                            "confidence": self.confidence,
                            "imgsz": self.imgsz,
                            "street_only": self.street_only,
                            "demo_accident_enabled": self.demo_accident,
                            "demo_accident_active": demo_accident_active,
                        }
                    )
            else:
                with self._lock:
                    self._stats["frames_dropped"] += 1

            frame_index += 1
            # Persist performance metrics every 60 s
            now_perf = time.time()
            if now_perf - _last_perf_save >= 60.0:
                _db_save_perf(self.latest_stats())
                _last_perf_save = now_perf

            time.sleep(1.0 / self.fps)

        capture.release()


def build_app(stream: YoloVideoStream) -> web.Application:
    @web.middleware
    async def cors_middleware(request, handler):
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    async def options_handler(_request):
        return web.Response(status=204)

    async def index(_request):
        stats = stream.latest_stats()
        html = f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>{stats['camera_id']} YOLO Feed</title>
          <style>
            body {{
              margin: 0;
              min-height: 100vh;
              display: grid;
              place-items: center;
              background: #06090d;
              color: #eef7ff;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            main {{
              width: min(96vw, 1440px);
            }}
            img {{
              display: block;
              width: 100%;
              border-radius: 20px;
              border: 1px solid rgba(255,255,255,.12);
              box-shadow: 0 24px 80px rgba(0,0,0,.45);
              background: #000;
            }}
          </style>
        </head>
        <body>
          <main>
            <img src="/video_feed" alt="YOLO CCTV stream">
          </main>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    async def health(_request):
        return web.json_response({"ok": True, **stream.latest_stats()})

    async def stats(_request):
        return web.json_response(stream.latest_stats())

    async def events(request):
        base_url = f"{request.scheme}://{request.host}"
        payload = []
        for event in stream.latest_events():
            item = dict(event)
            snapshot_path = item.get("snapshot_image")
            clip_path = item.get("clip_path")
            if snapshot_path:
                item["snapshot_url"] = base_url + "/artifacts/snapshots/" + Path(snapshot_path).name
            if clip_path:
                item["clip_url"] = base_url + "/artifacts/clips/" + Path(clip_path).name
            payload.append(item)
        return web.json_response(payload)

    async def snapshot_artifact(request):
        filename = request.match_info["filename"]
        path = stream.artifact_dir / filename
        if not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def clip_artifact(request):
        filename = request.match_info["filename"]
        path = stream.artifact_dir / filename
        if not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def accident_risk(_request):
        stats = stream.latest_stats()
        return web.json_response({
            "camera_id": stats["camera_id"],
            "accident_risk": stats["accident_risk"],
            "accident_alert": stats["accident_alert"],
            "scenario": stats["scenario"],
        })

    async def set_scenario(request):
        body = await request.json()
        scenario = body.get("scenario", "normal")
        stream.set_scenario(scenario)
        return web.json_response({"ok": True, "scenario": scenario})

    async def video_feed(_request):
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )
        await response.prepare(_request)
        try:
            while True:
                frame = stream.latest_jpeg()
                if frame:
                    await response.write(b"--frame\r\n")
                    await response.write(b"Content-Type: image/jpeg\r\n\r\n")
                    await response.write(frame)
                    await response.write(b"\r\n")
                await asyncio.sleep(1.0 / stream.fps)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        return response

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_route("OPTIONS", "/{tail:.*}", options_handler)
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/stats", stats)
    app.router.add_get("/events", events)
    app.router.add_get("/artifacts/snapshots/{filename}", snapshot_artifact)
    app.router.add_get("/artifacts/clips/{filename}", clip_artifact)
    app.router.add_get("/video_feed", video_feed)
    app.router.add_get("/accident_risk", accident_risk)
    app.router.add_post("/scenario", set_scenario)

    async def prometheus_metrics(_request):
        s = stream.latest_stats()
        cam = s.get("camera_id", "unknown").replace(" ", "_")
        lines = [
            f'# HELP traffic_frames_processed_total Total frames processed',
            f'# TYPE traffic_frames_processed_total counter',
            f'traffic_frames_processed_total{{camera="{cam}"}} {s.get("frames_processed", 0)}',
            f'# HELP traffic_frames_dropped_total Total frames dropped',
            f'# TYPE traffic_frames_dropped_total counter',
            f'traffic_frames_dropped_total{{camera="{cam}"}} {s.get("frames_dropped", 0)}',
            f'# HELP traffic_active_tracks Current tracked objects',
            f'# TYPE traffic_active_tracks gauge',
            f'traffic_active_tracks{{camera="{cam}"}} {s.get("active_tracks_total", 0)}',
            f'# HELP traffic_vehicle_count Current vehicles in frame',
            f'# TYPE traffic_vehicle_count gauge',
            f'traffic_vehicle_count{{camera="{cam}",class="car"}} {s.get("counts", {}).get("car", 0)}',
            f'traffic_vehicle_count{{camera="{cam}",class="truck"}} {s.get("counts", {}).get("truck", 0)}',
            f'traffic_vehicle_count{{camera="{cam}",class="bus"}} {s.get("counts", {}).get("bus", 0)}',
            f'traffic_vehicle_count{{camera="{cam}",class="person"}} {s.get("counts", {}).get("person", 0)}',
            f'traffic_vehicle_count{{camera="{cam}",class="total"}} {s.get("counts", {}).get("total", 0)}',
            f'# HELP traffic_accident_risk Accident risk score 0-100',
            f'# TYPE traffic_accident_risk gauge',
            f'traffic_accident_risk{{camera="{cam}"}} {s.get("accident_risk", 0.0)}',
            f'# HELP traffic_stream_uptime_seconds Stream uptime',
            f'# TYPE traffic_stream_uptime_seconds gauge',
            f'traffic_stream_uptime_seconds{{camera="{cam}"}} {s.get("stream_uptime_s", 0)}',
            f'# HELP traffic_reconnect_total Stream reconnect count',
            f'# TYPE traffic_reconnect_total counter',
            f'traffic_reconnect_total{{camera="{cam}"}} {s.get("reconnect_count", 0)}',
        ]
        return web.Response(text="\n".join(lines) + "\n",
                            content_type="text/plain; version=0.0.4")

    app.router.add_get("/metrics", prometheus_metrics)
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a looping YOLO-annotated CCTV video over HTTP.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8010, help="Bind port")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--fps", type=float, default=10.0, help="Output playback fps")
    parser.add_argument("--confidence", type=float, default=0.18, help="Detection confidence threshold")
    parser.add_argument("--infer-every", type=int, default=1, help="Run YOLO every Nth frame")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame")
    parser.add_argument("--device", default="", help="Optional Ultralytics device value, for example mps, cpu, or 0")
    parser.add_argument(
        "--all-coco-classes",
        action="store_true",
        help="Draw every COCO class instead of the street-object subset.",
    )
    parser.add_argument(
        "--demo-accident",
        action="store_true",
        help="Overlay and emit one simulated accident event for demonstration.",
    )
    parser.add_argument("--demo-accident-start-s", type=float, default=4.0, help="Demo accident start time in the loop")
    parser.add_argument("--demo-accident-duration-s", type=float, default=8.0, help="Demo accident display duration")
    parser.add_argument("--camera-id", default="CAM-01 NORTH", help="Displayed camera id")
    parser.add_argument(
        "--camera-focus",
        default="general",
        choices=["traffic_flow", "stop_stall", "general", "accident"],
        help="Camera specialization mode: traffic_flow | stop_stall | general | accident",
    )
    parser.add_argument(
        "--disable-accident-detection",
        action="store_true",
        help="Disable accident risk scoring, accident alerts, and accident HUD overlays.",
    )
    parser.add_argument(
        "--model",
        default=default_model_path(),
        help="Path to YOLO model weights. Defaults to the largest local YOLOv8 model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stream = YoloVideoStream(
        video_path=args.video,
        model_path=args.model,
        width=args.width,
        height=args.height,
        fps=args.fps,
        confidence=args.confidence,
        infer_every=args.infer_every,
        camera_id=args.camera_id,
        imgsz=args.imgsz,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        street_only=not args.all_coco_classes,
        demo_accident=args.demo_accident,
        demo_accident_start_s=args.demo_accident_start_s,
        demo_accident_duration_s=args.demo_accident_duration_s,
        camera_focus=args.camera_focus,
        disable_accident_detection=args.disable_accident_detection,
    )
    stream.start()
    app = build_app(stream)
    try:
        web.run_app(app, host=args.host, port=args.port)
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
