# app.py
# Requirements: pip install flask opencv-python ultralytics numpy
import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from model_predict2 import pred_emergency_vehicle   # assuming this exists
from predicter import predict_next_hour
app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

cap = None
streaming = False
emergency_alert = False
signal_time = 5

vehicle_counts = {
    "total": 0,
    "two_wheeler": 0,
    "car": 0,
    "bus": 0,
    "truck": 0,
    "ambulence": 0,
    "fire_truck": 0
}

# ── NEW: History for recent total vehicle counts ──
recent_totals = []          # will keep last 10 values
MAX_HISTORY = 10
avg_total_vehicles = 0.0    # displayed value

def process_frame(frame):
    global emergency_alert, signal_time, vehicle_counts, recent_totals, avg_total_vehicles

    results = model(frame)
    detections = results[0].boxes

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "ambulence": 0,
        "fire_truck": 0
    }

    emergency_alert = False
    annotated_frame = frame.copy()

    for idx, box in enumerate(detections):
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            label = VEHICLE_CLASSES[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            # Check emergency only for car/bus/truck
            if label in ["car", "bus", "truck"]:
                try:
                    filename = f"temp_{label}_{idx}.png"
                    cv2.imwrite(filename, crop)
                    pred_label, conf = pred_emergency_vehicle(filename)
                    if os.path.exists(filename):
                        os.remove(filename)

                    if pred_label in ["ambulence", "fire_truck"] and conf > 0.7:
                        counts[pred_label] += 1
                        emergency_alert = True
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, pred_label,
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        continue
                except Exception as e:
                    print("Emergency model error:", e)

            # Normal vehicle counting
            counts[label] += 1
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    total = sum(counts.values())

    # Update global counts
    vehicle_counts = {
        "total": total,
        "two_wheeler": counts["motorcycle"],
        "car": counts["car"],
        "bus": counts["bus"],
        "truck": counts["truck"],
        "ambulence": counts["ambulence"],
        "fire_truck": counts["fire_truck"]
    }

    # ── NEW: Keep last 10 total counts and compute average ──
    recent_totals.append(total)
    if len(recent_totals) > MAX_HISTORY:
        recent_totals.pop(0)  # remove oldest

    if recent_totals:
        #avg_total_vehicles = sum(recent_totals) / len(recent_totals)
        avg_total_vehicles=predict_next_hour(recent_totals)
    else:
        avg_total_vehicles = 0.0

    # Signal timing logic
    if emergency_alert:
        signal_time = 1
    elif total > 20:
        signal_time = 2
    else:
        signal_time = 5

    # Draw info on frame
    cv2.putText(annotated_frame,
                f"Total Vehicles: {total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(annotated_frame,
                f"Avg (10 frames): {avg_total_vehicles:.1f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if emergency_alert:
        cv2.putText(annotated_frame,
                    "EMERGENCY VEHICLE DETECTED!",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return annotated_frame


def gen():
    global cap, streaming
    while streaming:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        ret, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/start_camera")
def start_camera():
    global cap, streaming, recent_totals, avg_total_vehicles
    if not streaming:
        cap = cv2.VideoCapture("http://192.0.0.4:8080/video")
        if not cap.isOpened():
            return "Camera failed", 500
        streaming = True
        recent_totals = []           # reset history when starting
        avg_total_vehicles = 0.0
    return "Camera started"


@app.route("/stop_camera")
def stop_camera():
    global cap, streaming
    streaming = False
    if cap:
        cap.release()
        cap = None
    return "Camera stopped"


@app.route("/video_feed")
def video_feed():
    if not streaming:
        return "Camera not started", 400
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify({
        "counts": vehicle_counts,
        "signal_time": signal_time,
        "alert": emergency_alert,
        "avg_total_vehicles": round(avg_total_vehicles, 1)   # NEW
    })


if __name__ == "__main__":
    app.run(debug=True)