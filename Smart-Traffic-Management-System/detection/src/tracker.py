from collections import deque
from dataclasses import dataclass, field


def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass
class Track:
    track_id: int
    label: str
    bbox: list
    score: float
    created_ts: float
    last_seen_ts: float
    last_seen_frame: int
    state: str = "tracked"
    lost_frames: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=24))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=16))
    stationary_seconds: float = 0.0
    flags: set = field(default_factory=set)

    @property
    def centroid(self):
        return bbox_centroid(self.bbox)

    @property
    def latest_speed(self):
        return self.speed_history[-1] if self.speed_history else 0.0


class ByteTrackLikeTracker:
    def __init__(
        self,
        high_thresh=0.45,
        low_thresh=0.15,
        match_iou_thresh=0.3,
        low_match_iou_thresh=0.2,
        new_track_thresh=0.55,
        track_buffer_frames=30,
        stationary_speed_px_s=8.0,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou_thresh = match_iou_thresh
        self.low_match_iou_thresh = low_match_iou_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer_frames = track_buffer_frames
        self.stationary_speed_px_s = stationary_speed_px_s
        self._tracks = {}
        self._next_id = 1

    def active_tracks(self):
        return [
            track
            for track in self._tracks.values()
            if track.state == "tracked"
        ]

    def all_tracks(self):
        return list(self._tracks.values())

    def update(self, detections, frame_index, now_ts):
        high = [det for det in detections if det["score"] >= self.high_thresh]
        low = [det for det in detections if self.low_thresh <= det["score"] < self.high_thresh]

        candidate_tracks = [
            track
            for track in self._tracks.values()
            if frame_index - track.last_seen_frame <= self.track_buffer_frames
        ]

        high_matches = self._match(candidate_tracks, high, self.match_iou_thresh)
        matched_track_ids = {track_id for track_id, _ in high_matches}
        matched_high_ids = {det_idx for _, det_idx in high_matches}
        self._apply_matches(high_matches, high, frame_index, now_ts)

        unmatched_tracks = [
            track for track in candidate_tracks
            if track.track_id not in matched_track_ids
        ]
        low_matches = self._match(unmatched_tracks, low, self.low_match_iou_thresh)
        matched_track_ids |= {track_id for track_id, _ in low_matches}
        matched_low_ids = {det_idx for _, det_idx in low_matches}
        self._apply_matches(low_matches, low, frame_index, now_ts)

        for track in candidate_tracks:
            if track.track_id in matched_track_ids:
                continue
            track.lost_frames += 1
            track.state = "lost"

        for det_idx, det in enumerate(high):
            if det_idx in matched_high_ids or det["score"] < self.new_track_thresh:
                continue
            self._start_track(det, frame_index, now_ts)

        for det_idx, det in enumerate(low):
            if det_idx in matched_low_ids:
                continue
            det["track_id"] = None

        self._remove_stale_tracks(frame_index)
        return self.active_tracks()

    def _match(self, tracks, detections, iou_thresh):
        pairs = []
        for track in tracks:
            for det_idx, det in enumerate(detections):
                if track.label != det["label"]:
                    continue
                iou = bbox_iou(track.bbox, det["bbox"])
                if iou < iou_thresh:
                    continue
                centroid_dist = _distance(track.centroid, bbox_centroid(det["bbox"]))
                pairs.append((iou, -centroid_dist, track.track_id, det_idx))

        pairs.sort(reverse=True)
        matches = []
        used_tracks = set()
        used_detections = set()
        for _, _, track_id, det_idx in pairs:
            if track_id in used_tracks or det_idx in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(det_idx)
            matches.append((track_id, det_idx))
        return matches

    def _apply_matches(self, matches, detections, frame_index, now_ts):
        for track_id, det_idx in matches:
            track = self._tracks[track_id]
            det = detections[det_idx]
            previous_centroid = track.centroid
            previous_ts = track.last_seen_ts

            track.bbox = det["bbox"]
            track.score = det["score"]
            track.last_seen_frame = frame_index
            track.last_seen_ts = now_ts
            track.state = "tracked"
            track.lost_frames = 0
            track.history.append((now_ts, track.centroid))

            delta_t = max(1e-3, now_ts - previous_ts)
            speed = _distance(track.centroid, previous_centroid) / delta_t
            track.speed_history.append(speed)
            if speed < self.stationary_speed_px_s:
                track.stationary_seconds += delta_t
            else:
                track.stationary_seconds = 0.0

            det["track_id"] = track.track_id

    def _start_track(self, det, frame_index, now_ts):
        track = Track(
            track_id=self._next_id,
            label=det["label"],
            bbox=det["bbox"],
            score=det["score"],
            created_ts=now_ts,
            last_seen_ts=now_ts,
            last_seen_frame=frame_index,
        )
        track.history.append((now_ts, track.centroid))
        track.speed_history.append(0.0)
        self._tracks[self._next_id] = track
        det["track_id"] = self._next_id
        self._next_id += 1

    def _remove_stale_tracks(self, frame_index):
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame > self.track_buffer_frames
        ]
        for track_id in stale:
            del self._tracks[track_id]
