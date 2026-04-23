import argparse
import asyncio
from pathlib import Path

from aiohttp import ClientSession, web


DEFAULT_CAMERAS = {
    "North": "http://127.0.0.1:8011",
    "South": "http://127.0.0.1:8012",
    "East": "http://127.0.0.1:8013",
    "West": "http://127.0.0.1:8014",
}


class IncidentAggregator:
    def __init__(self, cameras):
        self.cameras = cameras
        self.events = []
        self.health = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "stream_uptime_s": 0,
            "reconnect_count": 0,
            "active_tracks": {key: 0 for key in cameras},
        }
        self._seen = set()

    async def refresh(self):
        timeout = 2.5
        async with ClientSession() as session:
            tasks = [self._fetch_camera(session, name, base_url, timeout) for name, base_url in self.cameras.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        merged_events = []
        frames_processed = 0
        frames_dropped = 0
        uptime = 0
        reconnect_count = 0
        active_tracks = {}

        for result in results:
            if not isinstance(result, dict):
                continue
            frames_processed += result["health"].get("frames_processed", 0)
            frames_dropped += result["health"].get("frames_dropped", 0)
            uptime = max(uptime, result["health"].get("stream_uptime_s", 0))
            reconnect_count += result["health"].get("reconnect_count", 0)
            active_tracks[result["name"]] = result["health"].get("active_tracks_total", 0)
            merged_events.extend(result["events"])

        merged_events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        deduped = []
        new_seen = set()
        for event in merged_events:
            key = (
                event.get("timestamp"),
                event.get("event_type"),
                event.get("track_id"),
                event.get("camera_id"),
            )
            if key in new_seen:
                continue
            new_seen.add(key)
            deduped.append(event)

        self._seen = new_seen
        self.events = deduped[:100]
        self.health = {
            "frames_processed": frames_processed,
            "frames_dropped": frames_dropped,
            "stream_uptime_s": uptime,
            "reconnect_count": reconnect_count,
            "active_tracks": active_tracks,
        }

    async def _fetch_camera(self, session, name, base_url, timeout):
        async with session.get(base_url + "/health", timeout=timeout) as health_resp:
            health_resp.raise_for_status()
            health = await health_resp.json()
        async with session.get(base_url + "/events", timeout=timeout) as events_resp:
            events_resp.raise_for_status()
            events = await events_resp.json()
        return {"name": name, "health": health, "events": events}


def build_app(aggregator):
    routes = web.RouteTableDef()

    @routes.get("/health")
    async def health(_request):
        await aggregator.refresh()
        return web.json_response(aggregator.health)

    @routes.get("/events")
    async def events(_request):
        await aggregator.refresh()
        return web.json_response(aggregator.events)

    app = web.Application()
    app.add_routes(routes)
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate incident events from per-camera YOLO streams.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    return parser.parse_args()


def main():
    args = parse_args()
    aggregator = IncidentAggregator(DEFAULT_CAMERAS)
    app = build_app(aggregator)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
