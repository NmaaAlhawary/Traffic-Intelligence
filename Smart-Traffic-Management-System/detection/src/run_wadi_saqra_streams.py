import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


CAMERAS = [
    {"video": "sense05.mov", "port": 8010, "camera_id": "WTS CAM 05"},
    {"video": "sense01.mov", "port": 8011, "camera_id": "CAM-01 NORTH"},
    {"video": "sense02.mov", "port": 8012, "camera_id": "CAM-02 SOUTH"},
    {"video": "sense03.mov", "port": 8013, "camera_id": "CAM-03 EAST"},
    {"video": "sense04.mov", "port": 8014, "camera_id": "CAM-04 WEST"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Launch all Wadi Saqra YOLO CCTV streams.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "yolov8n.pt"),
        help="Path to YOLO weights. Use a larger model path here if you already have one locally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    detection_src = Path(__file__).resolve().parent
    videos_dir = root / "CityFlow" / "data" / "wts_videos"
    python_bin = Path(sys.executable)
    http_stream = detection_src / "http_stream.py"
    incident_detector = detection_src / "incident_detector.py"

    processes = []

    def shutdown(*_args):
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5
        for proc in processes:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for cam in CAMERAS:
        video_path = videos_dir / cam["video"]
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        cmd = [
            str(python_bin),
            str(http_stream),
            "--video",
            str(video_path),
            "--host",
            args.host,
            "--port",
            str(cam["port"]),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
            "--confidence",
            str(args.confidence),
            "--infer-every",
            str(args.infer_every),
            "--camera-id",
            cam["camera_id"],
            "--model",
            args.model,
        ]
        proc = subprocess.Popen(cmd, cwd=str(detection_src))
        processes.append(proc)
        print(f"Started {cam['camera_id']} on :{cam['port']} using {cam['video']}", flush=True)

    incident_proc = subprocess.Popen(
        [str(python_bin), str(incident_detector), "--host", args.host, "--port", "5002"],
        cwd=str(detection_src),
    )
    processes.append(incident_proc)
    print("Started incident aggregator on :5002", flush=True)

    while True:
        for proc in processes:
            if proc.poll() is not None:
                raise SystemExit(proc.returncode)
        time.sleep(1)


if __name__ == "__main__":
    main()
