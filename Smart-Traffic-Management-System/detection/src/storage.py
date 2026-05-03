"""
Data Storage and Event Logging Layer (Phase 3 §8.5)
SQLite-backed persistent store for events, forecasts, performance metrics,
and detector counts.  Thread-safe; all public functions can be called from
any thread or async context.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "traffic_events.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not getattr(_local, "conn", None):
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return _local.conn


def _init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        event_type      TEXT    NOT NULL,
        category        TEXT,
        camera_id       TEXT,
        location        TEXT,
        confidence      REAL,
        track_id        INTEGER,
        snapshot_image  TEXT,
        clip_path       TEXT,
        queue_length    INTEGER,
        source_video    TEXT,
        raw_json        TEXT,
        inserted_at     REAL    DEFAULT (unixepoch('now', 'subsec'))
    );
    CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_type     ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_events_camera   ON events(camera_id);

    CREATE TABLE IF NOT EXISTS forecasts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        camera_id       TEXT,
        horizon_15_min  REAL,
        horizon_30_min  REAL,
        horizon_60_min  REAL,
        model_used      TEXT,
        input_json      TEXT,
        inserted_at     REAL    DEFAULT (unixepoch('now', 'subsec'))
    );
    CREATE INDEX IF NOT EXISTS idx_forecasts_ts ON forecasts(timestamp);

    CREATE TABLE IF NOT EXISTS performance_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT    NOT NULL,
        camera_id           TEXT,
        frames_processed    INTEGER,
        frames_dropped      INTEGER,
        stream_uptime_s     REAL,
        reconnect_count     INTEGER,
        active_tracks       INTEGER,
        accident_risk       REAL,
        inserted_at         REAL    DEFAULT (unixepoch('now', 'subsec'))
    );
    CREATE INDEX IF NOT EXISTS idx_perf_ts ON performance_log(timestamp);

    CREATE TABLE IF NOT EXISTS detector_counts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        intersection_id TEXT    NOT NULL,
        detector_id     TEXT    NOT NULL,
        approach        TEXT,
        lane            INTEGER,
        vehicle_count   INTEGER,
        occupancy_pct   REAL,
        speed_avg_kmh   REAL,
        inserted_at     REAL    DEFAULT (unixepoch('now', 'subsec'))
    );
    CREATE INDEX IF NOT EXISTS idx_det_ts  ON detector_counts(timestamp);
    CREATE INDEX IF NOT EXISTS idx_det_did ON detector_counts(detector_id);

    CREATE TABLE IF NOT EXISTS signal_recommendations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        intersection_id TEXT,
        approach        TEXT,
        action          TEXT,
        current_green_s INTEGER,
        recommended_green_s INTEGER,
        delta_s         INTEGER,
        urgency         TEXT,
        reason          TEXT,
        inserted_at     REAL    DEFAULT (unixepoch('now', 'subsec'))
    );
    CREATE INDEX IF NOT EXISTS idx_recs_ts ON signal_recommendations(timestamp);
    """)
    c.commit()


def save_event(event: dict):
    try:
        c = _conn()
        c.execute(
            """INSERT INTO events
               (timestamp, event_type, category, camera_id, location, confidence,
                track_id, snapshot_image, clip_path, queue_length, source_video, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.get("timestamp"),
                event.get("event_type"),
                event.get("category"),
                event.get("camera_id"),
                event.get("location"),
                event.get("confidence"),
                event.get("track_id"),
                event.get("snapshot_image"),
                event.get("clip_path"),
                event.get("queue_length"),
                event.get("source_video"),
                json.dumps(event),
            ),
        )
        c.commit()
    except Exception:
        pass


def save_forecast(camera_id: str, h15: float, h30: float, h60: float,
                  model_used: str = "lstm", input_data: dict = None):
    try:
        c = _conn()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        c.execute(
            """INSERT INTO forecasts
               (timestamp, camera_id, horizon_15_min, horizon_30_min, horizon_60_min,
                model_used, input_json)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, camera_id, h15, h30, h60, model_used, json.dumps(input_data or {})),
        )
        c.commit()
    except Exception:
        pass


def save_performance(stats: dict):
    try:
        c = _conn()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        c.execute(
            """INSERT INTO performance_log
               (timestamp, camera_id, frames_processed, frames_dropped,
                stream_uptime_s, reconnect_count, active_tracks, accident_risk)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                ts,
                stats.get("camera_id"),
                stats.get("frames_processed", 0),
                stats.get("frames_dropped", 0),
                stats.get("stream_uptime_s", 0),
                stats.get("reconnect_count", 0),
                stats.get("active_tracks_total", 0),
                stats.get("accident_risk", 0.0),
            ),
        )
        c.commit()
    except Exception:
        pass


def save_recommendation(rec: dict):
    try:
        c = _conn()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        c.execute(
            """INSERT INTO signal_recommendations
               (timestamp, intersection_id, approach, action,
                current_green_s, recommended_green_s, delta_s, urgency, reason)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                rec.get("intersection_id", "WADI_SAQRA_01"),
                rec.get("approach"),
                rec.get("action"),
                rec.get("current_green_s"),
                rec.get("recommended_green_s"),
                rec.get("delta_s"),
                rec.get("urgency"),
                rec.get("reason"),
            ),
        )
        c.commit()
    except Exception:
        pass


def get_recent_events(limit: int = 100, event_type: str = None) -> list:
    try:
        c = _conn()
        if event_type:
            rows = c.execute(
                "SELECT raw_json FROM events WHERE event_type=? ORDER BY inserted_at DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT raw_json FROM events ORDER BY inserted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["raw_json"]) for r in rows]
    except Exception:
        return []


def get_recent_forecasts(limit: int = 50) -> list:
    try:
        c = _conn()
        rows = c.execute(
            """SELECT timestamp, camera_id, horizon_15_min, horizon_30_min, horizon_60_min, model_used
               FROM forecasts ORDER BY inserted_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_performance_history(camera_id: str, last_n: int = 60) -> list:
    try:
        c = _conn()
        rows = c.execute(
            """SELECT timestamp, frames_processed, frames_dropped,
                      active_tracks, accident_risk
               FROM performance_log
               WHERE camera_id=?
               ORDER BY inserted_at DESC LIMIT ?""",
            (camera_id, last_n),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    except Exception:
        return []


def get_event_counts_by_type(hours: int = 24) -> dict:
    try:
        c = _conn()
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%S",
                               time.localtime(time.time() - hours * 3600))
        rows = c.execute(
            """SELECT event_type, COUNT(*) as cnt FROM events
               WHERE timestamp >= ? GROUP BY event_type ORDER BY cnt DESC""",
            (cutoff,),
        ).fetchall()
        return {r["event_type"]: r["cnt"] for r in rows}
    except Exception:
        return {}


def get_recent_recommendations(limit: int = 20) -> list:
    try:
        c = _conn()
        rows = c.execute(
            """SELECT * FROM signal_recommendations ORDER BY inserted_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


# Initialise DB on import
_init_db()
