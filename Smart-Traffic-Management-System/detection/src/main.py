import socket
import struct
import cv2
import numpy as np
import json
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
vehicle_classes = {"person": 0, 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def detect_and_annotate(image_bytes):
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(frame)[0]

    counts = {"person": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    for r in results.boxes:
        cls = int(r.cls[0])
        if cls in vehicle_classes:
            counts[vehicle_classes[cls]] += 1

            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            label = f"{vehicle_classes[cls]} {conf:.2f}"
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