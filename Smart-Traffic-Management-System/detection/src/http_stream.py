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


VEHICLE_LABELS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

BOX_COLORS = {
    "car": (62, 207, 255),
    "motorcycle": (255, 181, 46),
    "bus": (34, 217, 122),
    "truck": (245, 110, 95),
}

CAMERA_LOCATIONS = {
    "CAM-01 NORTH": "North approach",
    "CAM-02 SOUTH": "South approach",
    "CAM-03 EAST": "East approach",
    "CAM-04 WEST": "West approach",
    "WTS CAM 05": "Primary junction overview",
}

CONFIG_PATH = Path(__file__).with_name("camera_thresholds.json")


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
    ) -> None:
        self.video_path = Path(video_path)
        self.model = YOLO(model_path)
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.confidence = confidence
        self.infer_every = max(1, infer_every)
        self.camera_id = camera_id
        self.source_name = self.video_path.name
        self.location_name = CAMERA_LOCATIONS.get(camera_id, camera_id)
        self.camera_config = load_camera_config(camera_id)
        tracker_cfg = self.camera_config["tracker"]
        tracker_cfg["track_buffer_frames"] = max(6, int(tracker_cfg["track_buffer_frames"]))
        self.tracker = ByteTrackLikeTracker(**tracker_cfg)
        self.event_cfg = self.camera_config["events"]
        self.zone_cfg = self.camera_config["zones"]
        self.start_time = time.time()

        self.artifact_dir = Path(__file__).resolve().parents[1] / "logs" / "events" / self.camera_id.lower().replace(" ", "_")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._jpeg = b""
        self._stats = {
            "source_video": self.source_name,
            "camera_id": self.camera_id,
            "frame_index": 0,
            "counts": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0},
            "fps": self.fps,
            "resolution": {"width": self.width, "height": self.height},
            "updated_at": time.time(),
            "frames_processed": 0,
            "frames_dropped": 0,
            "stream_uptime_s": 0,
            "reconnect_count": 0,
            "active_tracks_total": 0,
        }
        self._last_detections = []
        self._events = deque(maxlen=100)
        self._event_cooldowns = {}
        self._recent_frames = deque(maxlen=max(12, int(self.fps * 4)))
        self._count_history = deque(maxlen=max(12, int(self.fps * 8)))
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
        results = self.model(frame, verbose=False)[0]
        detections = []
        counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = VEHICLE_LABELS.get(cls_id)
            if not label:
                continue

            score = float(box.conf[0])
            if score < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]
            detections.append(
                {
                    "label": label,
                    "score": score,
                    "bbox": bbox,
                    "track_id": None,
                }
            )
            counts[label] += 1
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

    def _draw_frame(self, frame, frame_index: int, detections, counts):
        display = frame.copy()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        cv2.rectangle(display, (14, 14), (345, 82), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"{self.camera_id}  YOLO LIVE",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
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

        stats_line = (
            f"Cars {counts['car']}   Moto {counts['motorcycle']}   "
            f"Bus {counts['bus']}   Truck {counts['truck']}   Total {counts['total']}"
        )
        cv2.rectangle(display, (14, 92), (520, 122), (0, 0, 0), -1)
        cv2.putText(
            display,
            stats_line,
            (24, 113),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 201, 64),
            1,
            cv2.LINE_AA,
        )

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

        return display

    def _run(self) -> None:
        capture = self._open_capture()
        frame_index = 0

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
                self.tracker.update(self._last_detections, frame_index, now_ts)
                self._analyze_events(frame.copy(), now_ts, counts)
            else:
                counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}
                for item in self._last_detections:
                    counts[item["label"]] += 1
                    counts["total"] += 1

            annotated = self._draw_frame(frame, frame_index, self._last_detections, counts)
            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            with self._lock:
                self._stats["frames_processed"] += 1
                self._stats["stream_uptime_s"] = int(now_ts - self.start_time)
                self._stats["active_tracks_total"] = len(self.tracker.active_tracks())
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
                        }
                    )
            else:
                with self._lock:
                    self._stats["frames_dropped"] += 1

            frame_index += 1
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
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a looping YOLO-annotated CCTV video over HTTP.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8010, help="Bind port")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--fps", type=float, default=10.0, help="Output playback fps")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--infer-every", type=int, default=2, help="Run YOLO every Nth frame")
    parser.add_argument("--camera-id", default="CAM-01 NORTH", help="Displayed camera id")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).with_name("yolov8n.pt")),
        help="Path to YOLO model weights",
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
    )
    stream.start()
    app = build_app(stream)
    try:
        web.run_app(app, host=args.host, port=args.port)
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
