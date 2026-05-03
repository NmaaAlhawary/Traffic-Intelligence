import socket
import struct
import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "yolov8m.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).with_name("yolov8n.pt")

model = YOLO(str(MODEL_PATH))
street_classes = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

def detect_and_annotate(image_bytes):
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(frame, conf=0.18, imgsz=960, max_det=300)[0]

    counts = {
        "person": 0,
        "bicycle": 0,
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "traffic_light": 0,
        "traffic_sign": 0,
        "total": 0,
    }

    for r in results.boxes:
        cls = int(r.cls[0])
        if cls in street_classes:
            label_name = street_classes[cls]
            count_key = label_name.replace(" ", "_")
            if label_name == "stop sign":
                count_key = "traffic_sign"
            counts[count_key] += 1
            counts["total"] += 1

            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            label = f"{label_name} {conf:.2f}"
            if conf >= 0.7:
                color = (0, 255, 0)
            elif conf >= 0.4:
                color = (0, 255, 255)     # Yellow
            else:
                color = (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, annotated_jpeg = cv2.imencode(".jpg", frame)
    return annotated_jpeg.tobytes(), json.dumps(counts).encode()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 5001))
    sock.listen(1)
    print("🚀 Python server running...")

    while True:
        conn, _ = sock.accept()
        try:
            length = struct.unpack("!I", conn.recv(4))[0]
            image_data = b""
            while len(image_data) < length:
                chunk = conn.recv(length - len(image_data))
                if not chunk:
                    break
                image_data += chunk

            annotated_image, json_data = detect_and_annotate(image_data)

            # Send response: first 4 bytes = length of JSON, next = JSON, rest = JPEG
            conn.sendall(struct.pack("!I", len(json_data)) + json_data + annotated_image)
        finally:
            conn.close()

if __name__ == "__main__":
    main()
