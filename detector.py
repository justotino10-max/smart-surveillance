# Updated code.txt — surveillance Flask app with pre-event image buffer (timestamped files)
import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime, timedelta
import os
import threading
from flask import Flask, render_template_string, request, redirect, url_for, Response, stream_with_context, flash, send_from_directory
import time  # For sleeping during reconnect attempts
import base64
import numpy as np

import random
from collections import deque
from pathlib import Path

# ----------------------------------------------------------------
# If you're on Windows and Tesseract isn't on PATH, UNCOMMENT and
# set this to your installed location, e.g.:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ----------------------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "yolov8n.pt"
WINDOW_TITLE = "Smart Asset Surveillance"

TRACKABLE_CLASSES = ["chair", "monitor", "laptop", "cup", "bowl", "cell phone", "person"]
USE_CLASS_FILTER = True

IOU_SAME_OBJECT = 0.80
IOU_PRESENCE = 0.50
stream_reader = None
camera_index = 0

# Real-time missing detection: object considered missing after this many seconds
MISSING_GRACE_SECONDS = 5

# Occlusion threshold and optional occlusion escalation (if desired)
OCCLUSION_FG_RATIO = 0.15
OCCLUSION_ESCALATE_SECONDS = 5  # escalate occluded -> missing after this many seconds if still not visible

WARMUP_FRAMES_BG = 60
DB_PATH = "events.db"

# How many seconds to keep in the rolling buffer for pre-event capture
FRAME_BUFFER_SECONDS = 5

# Where to save event pre-event images
EVENT_IMAGES_DIR = Path("event_images")
EVENT_IMAGES_DIR.mkdir(exist_ok=True)

# baseline mode: "manual" (default) or "auto"
BASELINE_MODE = "manual"

# Global state for video processing
camera = None
model = None
baseline = []
baseline_ready = False
bg = None

# Global variable to hold the current stream URL or "webcam"
RTSP_URL = None

# Share current frame with Baseline Assistant
frame_lock = threading.Lock()
current_frame = None

# Rolling frame buffer: stores tuples (timestamp (datetime), frame (ndarray))
frame_buffer = deque()

# Create a global dictionary to hold shared data
shared_data = {
    "latest_frame": None,
    "db_lock": threading.Lock(),
}

class MonitorThread(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera = None
        self.model = YOLO(MODEL_PATH)
        self.bg = None
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        self.connect_to_stream()
        while not self.stop_event.is_set():
            

            if self.camera is None or not self.camera.isOpened():
                print("Stream disconnected, attempting to reconnect...")
                self.connect_to_stream()
                time.sleep(5)  # Wait before trying again
                continue

            ok, frame = self.camera.read()
            if not ok:
                continue

            # Process the frame here
            if self.bg is None:
                self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            
            # This is where all your existing frame processing and analysis will go
            # including:
            # - Motion detection with self.bg
            # - YOLO object detection with self.model
            # - Database interaction
            
            # Example:
            # Your original run_monitoring() logic goes here.
            # You'll need to pass the frame and other necessary info.
            fgmask = self.bg.apply(frame)
            results = self.model.predict(frame, stream=True, classes=TRACKABLE_CLASSES)

            # Store the latest processed frame in the global shared_data dictionary
            # You need to ensure this is thread-safe
            with shared_data["db_lock"]:
                shared_data["latest_frame"] = frame.copy()
            
            time.sleep(1/30) # Prevent 100% CPU usage
            
        if self.camera:
            self.camera.release()

    def connect_to_stream(self):
        try:
            if self.rtsp_url == "webcam":
                self.camera = cv2.VideoCapture(camera_index)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.camera = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Wait for the camera to open
            time.sleep(2)
            if not self.camera.isOpened():
                print(f"Failed to open stream at {self.rtsp_url}")
                self.camera = None
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            self.camera = None

    def stop(self):
        self.stop_event.set()
        self.join()

class RTSPStreamReader(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        self.started = True
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                # Optionally handle reconnect logic here
                print("Stream lost, attempting to reconnect...")
                self.cap.release()
                time.sleep(2) # Wait before trying again
                self.cap = cv2.VideoCapture(self.src)
                continue
            with self.read_lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.ret else None
            ret = self.ret
        return ret, frame

    def start(self):
        if not self.started:
            super().start()
        return self

    def stop(self):
        self.stop_event.set()
        self.cap.release()
        self.join()

app = Flask(__name__)
app.secret_key = "devkey"  # for flash messages; change for production

# -----------------------------
# Helper functions
# -----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def now_ts():
    return datetime.now().timestamp()

def ts_to_str(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")

def iou_xywh(a, b):
    x1A, y1A, w1A, h1A = a
    x2A, y2A = x1A + w1A, y1A + h1A
    x1B, y1B, w1B, h1B = b
    x2B, y2B = x1B + w1B, y1B + h1B
    x1i = max(x1A, x1B)
    y1i = max(y1A, y1B)
    x2i = min(x2A, x2B)
    y2i = min(y2A, y2B)
    iw = max(0, x2i - x1i)
    ih = max(0, y2i - y1i)
    inter = iw * ih
    ua = w1A * h1A + w1B * h1B - inter
    return inter / ua if ua > 0 else 0.0

def draw_box(frame, xywh, label, color=(0,255,0)):
    x, y, w, h = xywh
    p1 = (int(x - w/2), int(y - h/2))
    p2 = (int(x + w/2), int(y + h/2))
    cv2.rectangle(frame, p1, p2, color, 2)
    cv2.putText(frame, label, (p1[0], max(15, p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def extract_detections(frame, model):
    results = model(frame, verbose=False)
    dets = []
    if len(results) and results[0].boxes is not None:
        boxes = results[0].boxes.xywh
        cids = results[0].boxes.cls
        confs = results[0].boxes.conf
        for box, cid, conf in zip(boxes, cids, confs):
            xc, yc, w, h = [float(v) for v in box]
            label = results[0].names[int(cid)]
            dets.append((xc, yc, w, h, label, float(conf)))
    return dets

def count_by_label(objs):
    counts = {}
    for _, _, _, _, lbl, _ in objs:
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts

def draw_counts(frame, counts):
    y = 30
    x = max(10, frame.shape[1] - 300)
    for lbl in sorted(counts.keys()):
        cv2.putText(frame, f"{lbl}: {counts[lbl]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        y += 26

def clamp_box_to_frame(xywh, shape):
    H, W = shape[:2]
    x, y, w, h = xywh
    x1, y1 = int(max(0, x - w/2)), int(max(0, y - h/2))
    x2, y2 = int(min(W-1, x + w/2)), int(min(H-1, y + h/2))
    return x1, y1, x2, y2



def next_baseline_id():
    return (max([o["id"] for o in baseline]) + 1) if baseline else 1

# -----------------------------
# Database
# -----------------------------
def init_db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS baseline_objects (
            object_id INTEGER PRIMARY KEY,
            label TEXT,
            x REAL, y REAL, w REAL, h REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            event TEXT,
            object_id INTEGER,
            label TEXT,
            details TEXT,
            image_data TEXT,
            extra_images TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS current_status (
            object_id INTEGER PRIMARY KEY,
            label TEXT,
            missing INTEGER,
            missing_since TEXT
        )
    """)
    con.commit()
    return con

def upsert_status(con, obj_id, label, missing, missing_since):
    cur = con.cursor()
    cur.execute("""
        INSERT INTO current_status (object_id, label, missing, missing_since)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
            label=excluded.label,
            missing=excluded.missing,
            missing_since=excluded.missing_since
    """, (obj_id, label, int(missing), missing_since))
    con.commit()

def insert_event(con, event, object_id, label, details="", image_data=None, extra_images=None):
    cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts, event, object_id, label, details, image_data, extra_images) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (now_str(), event, object_id, label, details, image_data, extra_images)
    )
    con.commit()
    return cur.lastrowid

def update_event_extra_images(con, event_id, extra_images_csv):
    cur = con.cursor()
    cur.execute("UPDATE events SET extra_images = ? WHERE id = ?", (extra_images_csv, event_id))
    con.commit()

def clear_missing_event_image(con, object_id):
    cur = con.cursor()
    cur.execute("""
        UPDATE events
        SET image_data = NULL
        WHERE id = (
            SELECT id FROM events
            WHERE object_id = ? AND event = 'MISSING'
            ORDER BY ts DESC LIMIT 1
        )
    """, (object_id,))
    con.commit()

def clear_all_event_images(con):
    cur = con.cursor()
    # Clear DB fields
    cur.execute("UPDATE events SET image_data = NULL, extra_images = NULL")
    con.commit()
    # Remove files on disk
    if EVENT_IMAGES_DIR.exists():
        for f in EVENT_IMAGES_DIR.iterdir():
            try:
                f.unlink()
            except Exception:
                pass

def save_baseline_to_db(con, baseline_list):
    cur = con.cursor()
    cur.execute("DELETE FROM baseline_objects")
    for obj in baseline_list:
        x,y,w,h = obj["box"]
        cur.execute("INSERT INTO baseline_objects (object_id, label, x, y, w, h) VALUES (?, ?, ?, ?, ?, ?)",
                    (obj["id"], obj["display_label"], x, y, w, h))
        upsert_status(con, obj["id"], obj["display_label"], 0, None)
    con.commit()

def clear_baseline_db(con):
    cur = con.cursor()
    cur.execute("DELETE FROM baseline_objects")
    cur.execute("DELETE FROM current_status")
    con.commit()

# -----------------------------
# Helper: Save buffered frames as timestamped JPEGs for a given event
# -----------------------------
def save_buffered_images_for_event(event_id, event_ts, buffer_seconds=FRAME_BUFFER_SECONDS):
    """
    Save frames from frame_buffer where frame_ts is within (event_ts - buffer_seconds) .. event_ts.
    Returns list of saved file paths (strings).
    """
    saved_paths = []
    # Ensure directory exists
    EVENT_IMAGES_DIR.mkdir(exist_ok=True)
    lower_ts = event_ts - buffer_seconds
    # create filenames and save
    # iterate through buffer (which is ordered oldest->newest if we append normally)
    for frame_ts, frame in list(frame_buffer):
        if frame_ts < lower_ts:
            continue
        if frame_ts > event_ts:
            continue
        filename = f"event_{event_id}_{ts_to_str(frame_ts)}.jpg"
        path = EVENT_IMAGES_DIR / filename
        try:
            cv2.imencode('.jpg', frame)[1].tofile(str(path))
            saved_paths.append(str(path))
        except Exception:
            # fallback to imencode & write
            ok, buf = cv2.imencode('.jpg', frame)
            if ok:
                with open(path, 'wb') as f:
                    f.write(buf.tobytes())
                saved_paths.append(str(path))
    return saved_paths

# -----------------------------
# Flask Templates (Tailwind) - unchanged visually, identical to your previous
# -----------------------------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asset Surveillance Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .modal { display:none; position:fixed; z-index:1000; inset:0; background:rgba(0,0,0,.9); justify-content:center; align-items:center; }
        .modal-content { margin:auto; display:block; width:80%; max-width:900px; }
    </style>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen p-6 md:p-12">
    <div class="max-w-7xl mx-auto">
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="mb-4 space-y-2">
              {% for msg in messages %}
                <div class="rounded-lg bg-emerald-900/50 border border-emerald-700 text-emerald-200 px-4 py-2">{{ msg }}</div>
              {% endfor %}
            </div>
          {% endif %}
        {% endwith %}

        <header class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-8">
            <div>
              <h1 class="text-3xl font-bold tracking-tight text-indigo-400">Asset Surveillance</h1>
              <p class="text-slate-400 mt-1 text-sm">
                Baseline Mode: <span class="font-semibold">{{ baseline_mode|upper }}</span> •
                Baseline Items: <span class="font-semibold">{{ baseline_count }}</span> •
                Monitoring: <span class="font-semibold">{{ "ON" if baseline_ready else "OFF" }}</span>
              </p>
            </div>
            <div class="flex flex-wrap items-center gap-3">
    <a href="{{ url_for('show_stream') }}" class="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-lg hover:bg-indigo-700">
        View Live Stream
    </a>
    <a href="{{ url_for('baseline_assist') }}" class="inline-flex items-center justify-center rounded-lg bg-orange-500 px-4 py-2 text-sm font-semibold text-white shadow-lg hover:bg-orange-600">
        Baseline Assistant
    </a>
</div>
                <form method="POST" action="{{ url_for('set_baseline_mode') }}" class="inline">
                    <input type="hidden" name="mode" value="{{ 'auto' if baseline_mode=='manual' else 'manual' }}">
                    <button class="rounded-lg bg-slate-700 px-4 py-2 text-sm font-semibold hover:bg-slate-600">
                        Switch to {{ 'AUTO' if baseline_mode=='manual' else 'MANUAL' }}
                    </button>
                </form>
                <form method="POST" action="{{ url_for('baseline_finalize') }}" class="inline">
                    <button class="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700">
                        Start Monitoring
                    </button>
                </form>
                <form method="POST" action="{{ url_for('baseline_clear') }}" class="inline" onsubmit="return confirm('Clear baseline and reset status?')">
                    <button class="rounded-lg bg-gray-600 px-4 py-2 text-sm font-semibold text-white hover:bg-gray-700">
                        Clear Baseline
                    </button>
                </form>
                <form method="POST" action="{{ url_for('clear_db') }}" class="inline" onsubmit="return confirm('Clear all events and status?')">
                    <button class="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700">
                        Clear Events
                    </button>
                </form>
                <!-- Delete all stored images -->
                <form method="POST" action="{{ url_for('clear_all_images') }}" class="inline" onsubmit="return confirm('Delete ALL stored snapshots from events? Events remain, only images are cleared.')">
                    <button class="rounded-lg bg-fuchsia-700 px-4 py-2 text-sm font-semibold text-white hover:bg-fuchsia-800">
                        Delete All Images
                    </button>
                  <button class="rounded-lg bg-fuchsia-700 px-4 py-2 text-sm font-semibold text-white hover:bg-fuchsia-800">  <a href="{{ url_for('dashboard') }}" class="...">Go to Dashboard</a>  </button>
                </form>
            </div>
        </header>

        <main class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Configuration Panel -->
            <div class="bg-slate-900 rounded-xl shadow-lg p-6 lg:col-span-1">
                <h2 class="text-xl font-semibold mb-4 text-slate-100">Configure Stream</h2>
                <form method="POST" action="{{ url_for('set_stream') }}" class="mb-4">
                    <label for="rtsp_url" class="block text-sm font-medium text-slate-400 mb-1">RTSP URL</label>
                    <input type="text" id="rtsp_url" name="rtsp_url" value="{{ rtsp_url if rtsp_url != 'webcam' else 'rtsp://user:password@ip:port/stream' }}" required class="block w-full rounded-md border-gray-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder-slate-400 focus:border-indigo-500 focus:ring-indigo-500 shadow-sm sm:text-sm">
                    <button type="submit" class="w-full rounded-md bg-indigo-600 px-4 py-2 mt-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-indigo-500">
                        Start RTSP Stream
                    </button>
                </form>
                <form method="POST" action="{{ url_for('set_webcam') }}">
                    <button type="submit" class="w-full rounded-md bg-gray-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-gray-500">
                        Use Webcam
                    </button>
                </form>
            </div>

            <!-- Current Status & Recent Events -->
            <div class="lg:col-span-2 space-y-6">
                <!-- Current Status Table -->
                <div class="bg-slate-900 rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4 text-slate-100">Current Status</h2>
                    <div class="overflow-x-auto">
                        <table class="w-full table-auto">
                            <thead>
                                <tr class="bg-slate-800 text-slate-400 uppercase text-sm leading-normal">
                                    <th class="py-3 px-6 text-left rounded-tl-lg">ID</th>
                                    <th class="py-3 px-6 text-left">Label</th>
                                    <th class="py-3 px-6 text-left">Status</th>
                                    <th class="py-3 px-6 text-left rounded-tr-lg">Missing Since</th>
                                </tr>
                            </thead>
                            <tbody class="text-slate-300 text-sm font-light">
                                {% for row in status %}
                                <tr class="border-b border-slate-700 hover:bg-slate-800">
                                    <td class="py-3 px-6">{{ row['object_id'] }}</td>
                                    <td class="py-3 px-6">{{ row['label'] }}</td>
                                    {% if row['missing'] %}
                                        <td class="py-3 px-6"><span class="bg-red-700 text-red-100 text-xs font-semibold px-2 py-1 rounded-full uppercase">Missing</span></td>
                                    {% else %}
                                        <td class="py-3 px-6"><span class="bg-green-700 text-green-100 text-xs font-semibold px-2 py-1 rounded-full uppercase">OK</span></td>
                                    {% endif %}
                                    <td class="py-3 px-6">{{ row['missing_since'] or '-' }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Recent Events Table -->
                <div class="bg-slate-900 rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4 text-slate-100">Recent Events</h2>
                    <div class="overflow-x-auto">
                        <table class="w-full table-auto">
                            <thead>
                                <tr class="bg-slate-800 text-slate-400 uppercase text-sm leading-normal">
                                    <th class="py-3 px-6 text-left rounded-tl-lg">Time</th>
                                    <th class="py-3 px-6 text-left">Event</th>
                                    <th class="py-3 px-6 text-left">Object ID</th>
                                    <th class="py-3 px-6 text-left">Label</th>
                                    <th class="py-3 px-6 text-left">Details</th>
                                    <th class="py-3 px-6 text-center rounded-tr-lg">Image</th>
                                </tr>
                            </thead>
                            <tbody class="text-slate-300 text-sm font-light">
                                {% for e in events %}
                                <tr class="border-b border-slate-700 hover:bg-slate-800">
                                    <td class="py-3 px-6">{{ e['ts'] }}</td>
                                    {% if e['event'] == 'MISSING' %}
                                        <td class="py-3 px-6 text-red-400 font-medium">{{ e['event'] }}</td>
                                    {% elif e['event'] == 'OUT_OF_PLACE' %}
                                        <td class="py-3 px-6 text-amber-400 font-medium">{{ e['event'] }}</td>
                                    {% else %}
                                        <td class="py-3 px-6 text-green-400 font-medium">{{ e['event'] }}</td>
                                    {% endif %}
                                    <td class="py-3 px-6">{{ e['object_id'] }}</td>
                                    <td class="py-3 px-6">{{ e['label'] }}</td>
                                    <td class="py-3 px-6">{{ e['details'] }}</td>
                                    <td class="py-3 px-6 text-center">
                                        {% if e['image_data'] %}
                                            <button onclick="showImageModal('{{ e['image_data'] }}')" class="text-slate-400 hover:text-indigo-400 transition-colors">
                                                View
                                            </button>
                                        {% else %}
                                            <span class="text-slate-600">-</span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <div id="imageModal" class="modal" onclick="this.style.display='none'">
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showImageModal(imageData) {
            const modal = document.getElementById("imageModal");
            const modalImage = document.getElementById("modalImage");
            modalImage.src = "data:image/jpeg;base64," + imageData;
            modal.style.display = "flex";
        }
    </script>
</body>
</html>
"""

STREAM_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Stream</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style> body { font-family: 'Inter', sans-serif; } </style>
</head>
<body class="bg-slate-950 flex flex-col items-center justify-center h-screen p-4">
    <div class="relative w-full h-full max-w-7xl max-h-[90vh] rounded-xl overflow-hidden shadow-2xl">
        <img src="{{ url_for('video_feed') }}" class="w-full h-full object-contain" alt="Live Stream">
        <a href="{{ url_for('index') }}" class="absolute top-4 left-4 inline-flex items-center justify-center rounded-full bg-slate-800/50 backdrop-blur-sm p-3 text-white hover:bg-slate-700/70">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
                <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
            </svg>
        </a>
    </div>
</body>
</html>
"""

BASELINE_ASSISTANT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Baseline Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style> body { font-family: 'Inter', sans-serif; } </style>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen p-6 md:p-12">
  <div class="max-w-7xl mx-auto">
    <header class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-2xl font-bold text-amber-400">Baseline Assistant</h1>
        <p class="text-slate-400 text-sm mt-1">Pick exactly which detections to secure.</p>
      </div>
      <a href="{{ url_for('index') }}" class="rounded-lg bg-slate-700 px-4 py-2 text-sm hover:bg-slate-600">Back</a>
    </header>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="mb-4 space-y-2">
          {% for msg in messages %}
            <div class="rounded-lg bg-emerald-900/50 border border-emerald-700 text-emerald-200 px-4 py-2">{{ msg }}</div>
          {% endfor %}
        </div>
      {% endif %}
    {% endwith %}

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="bg-slate-900 rounded-xl shadow-lg p-6">
        <h2 class="text-lg font-semibold mb-3">1) Snapshot</h2>
        <form method="POST" action="{{ url_for('baseline_snapshot') }}">
          <button class="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700">Take Snapshot</button>
        </form>

        {% if snapshot_b64 %}
          <div class="mt-4">
            <img class="rounded-lg shadow" src="data:image/jpeg;base64,{{ snapshot_b64 }}" alt="snapshot">
          </div>
        {% else %}
          <p class="text-slate-400 mt-4 text-sm">No snapshot yet. Click <strong>Take Snapshot</strong> to capture the current frame.</p>
        {% endif %}
      </div>

      <div class="bg-slate-900 rounded-xl shadow-lg p-6">
        <h2 class="text-lg font-semibold mb-3">2) Select Detections to Secure</h2>

        {% if detections %}
          <form method="POST" action="{{ url_for('baseline_add') }}">
            <input type="hidden" name="count" value="{{ detections|length }}">
            <div class="overflow-x-auto">
              <table class="w-full table-auto">
                <thead>
                  <tr class="bg-slate-800 text-slate-400 uppercase text-xs">
                    <th class="py-2 px-3 text-left">Pick</th>
                    <th class="py-2 px-3 text-left">#</th>
                    <th class="py-2 px-3 text-left">Label</th>
                    <th class="py-2 px-3 text-left">Conf</th>
                   
                    <th class="py-2 px-3 text-left">Custom Tag</th>
                  </tr>
                </thead>
                <tbody class="text-sm">
                  {% for d in detections %}
                  <tr class="border-b border-slate-800">
                    <td class="py-2 px-3">
                      <input type="checkbox" name="select" value="{{ loop.index0 }}" {% if d['label'] in trackables %}checked{% endif %}>
                    </td>
                    <td class="py-2 px-3">{{ loop.index0 }}</td>
                    <td class="py-2 px-3">{{ d['label'] }}</td>
                    <td class="py-2 px-3">{{ '%.2f' % d['conf'] }}</td>
                    
                    <td class="py-2 px-3">
                      <input name="custom_{{ loop.index0 }}" placeholder="e.g. A1" class="w-24 rounded bg-slate-800 px-2 py-1 text-slate-100 text-sm">
                    </td>
                  </tr>
                  <input type="hidden" name="x_{{ loop.index0 }}" value="{{ d['x'] }}">
                  <input type="hidden" name="y_{{ loop.index0 }}" value="{{ d['y'] }}">
                  <input type="hidden" name="w_{{ loop.index0 }}" value="{{ d['w'] }}">
                  <input type="hidden" name="h_{{ loop.index0 }}" value="{{ d['h'] }}">
                  <input type="hidden" name="label_{{ loop.index0 }}" value="{{ d['label'] }}">
                
                  {% endfor %}
                </tbody>
              </table>
            </div>
            <div class="mt-4 flex gap-3">
              <button class="rounded-lg bg-amber-600 px-4 py-2 text-sm font-semibold text-white hover:bg-amber-700">Add to Baseline</button>
            </div>
          </form>
        {% else %}
          <p class="text-slate-400 text-sm">No detections yet. Take a snapshot when the camera sees the objects.</p>
        {% endif %}

        <div class="mt-6">
          <form method="POST" action="{{ url_for('baseline_finalize') }}">
            <button class="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700">Start Monitoring</button>
          </form>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
# -----------------------------
# HTML Templates
# -----------------------------
# (Your other templates go here)

DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Camera Dashboard</title>
    <style>
        body {
            font-family: sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            background-color: #f0f0f0;
            margin: 0;
        }
        .container {
            text-align: center;
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 1.5rem;
        }
        .camera-links a {
            display: inline-block;
            padding: 10px 20px;
            margin: 10px;
            text-decoration: none;
            color: white;
            background-color: #007BFF;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .camera-links a:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Select a Camera Feed</h1>
        <div class="camera-links">
            <a href="http://127.0.0.1:5000">Camera 1</a>
            <a href="http://127.0.0.1:5001">Camera 2</a>
        </div>
    </div>
</body>
</html>
"""

def dict_rows(cur):
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

# -----------------------------
# Routes
# -----------------------------

# -----------------------------
# New Flask Route to Switch Cameras
# -----------------------------
# -----------------------------
# New route for the dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML_TEMPLATE)

@app.route("/", methods=["GET"])
def index():
    if not os.path.exists(DB_PATH):
        init_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT object_id, label, missing, COALESCE(missing_since,'') as missing_since FROM current_status ORDER BY object_id")
    status = dict_rows(cur)
    cur.execute("SELECT id, ts, event, object_id, label, COALESCE(details,'') as details, image_data, extra_images FROM events ORDER BY id DESC LIMIT 100")
    events = dict_rows(cur)
    con.close()
    return render_template_string(
        TEMPLATE,
        status=status,
        events=events,
        rtsp_url=RTSP_URL,
        baseline_mode=BASELINE_MODE,
        baseline_count=len(baseline),
        baseline_ready=baseline_ready
    )

@app.route("/stream", methods=["GET"])
def show_stream():
    return render_template_string(STREAM_TEMPLATE)

@app.route("/clear", methods=["POST"])
def clear_db():
    if os.path.exists(DB_PATH):
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM events")
        cur.execute("DELETE FROM current_status")
        con.commit()
        con.close()
    # Also clear files
    if EVENT_IMAGES_DIR.exists():
        for f in EVENT_IMAGES_DIR.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
    flash("Events and status cleared (and event images deleted).")
    return redirect(url_for('index'))

# Clear only images (DB image_data/extra_images + files)
@app.route("/clear_images", methods=["POST"])
def clear_all_images():
    con = init_db()
    clear_all_event_images(con)
    con.close()
    flash("All stored event snapshots and pre-event images have been deleted. (Events kept.)")
    return redirect(url_for('index'))

@app.route("/set_stream", methods=["POST"])
def set_stream():
    global RTSP_URL, stream_reader, baseline, baseline_ready, bg
    new_url = request.form.get("rtsp_url")
    if not new_url:
        flash("RTSP URL cannot be empty.")
        return redirect(url_for('index'))

    RTSP_URL = new_url

    # Stop any existing stream before starting a new one
    if stream_reader and stream_reader.is_alive():
        stream_reader.stop()
        stream_reader.join() # Wait for the thread to finish

    # Start the new stream
    stream_reader = RTSPStreamReader(RTSP_URL).start()

    baseline = []
    baseline_ready = False
    bg = None
    with frame_lock:
        frame_buffer.clear()
    flash("RTSP stream configured. Baseline reset.")
    return redirect(url_for('index'))

@app.route("/set_webcam", methods=["POST"])
def set_webcam():
    global RTSP_URL, stream_reader, baseline, baseline_ready, bg
    RTSP_URL = "webcam"

    if stream_reader and stream_reader.is_alive():
        stream_reader.stop()
        stream_reader.join()

    stream_reader = RTSPStreamReader(camera_index).start()

    baseline = []
    baseline_ready = False
    bg = None
    with frame_lock:
        frame_buffer.clear()
    flash("Webcam selected. Baseline reset.")
    return redirect(url_for('index'))

@app.route("/set_baseline_mode", methods=["POST"])
def set_baseline_mode():
    global BASELINE_MODE
    mode = request.form.get("mode", "manual").strip().lower()
    if mode not in ("manual", "auto"):
        mode = "manual"
    BASELINE_MODE = mode
    flash(f"Baseline mode set to {mode.upper()}.")
    return redirect(url_for('index'))

@app.route("/baseline_clear", methods=["POST"])
def baseline_clear():
    global baseline, baseline_ready
    baseline = []
    baseline_ready = False
    con = init_db()
    clear_baseline_db(con)
    con.close()
    flash("Baseline cleared.")
    return redirect(url_for('index'))

@app.route("/baseline_finalize", methods=["POST"])
def baseline_finalize():
    global baseline_ready
    if not baseline:
        flash("Baseline is empty. Add at least one object.")
    else:
        baseline_ready = True
        con = init_db()
        save_baseline_to_db(con, baseline)
        con.close()
        flash("Monitoring started.")
    return redirect(url_for('index'))

# ---- Baseline Assistant Flow ----
@app.route("/baseline", methods=["GET"])
def baseline_assist():
    return render_template_string(
        BASELINE_ASSISTANT_TEMPLATE,
        snapshot_b64=None,
        detections=None,
        trackables=TRACKABLE_CLASSES
    )

import random # Add this line to the top of your file

@app.route("/baseline/snapshot", methods=["POST"])
def baseline_snapshot():
    global model
    with frame_lock:
        frame = None if current_frame is None else current_frame.copy()
    if frame is None:
        flash("No frame available. Start the stream or webcam first.")
        return redirect(url_for('baseline_assist'))

    if model is None:
        model = YOLO(MODEL_PATH)

    dets = extract_detections(frame, model)
    if USE_CLASS_FILTER:
        dets = [d for d in dets if d[4] in TRACKABLE_CLASSES]

    annotated = frame.copy()
    detections = []
    for idx, (xc, yc, w, h, lbl, conf) in enumerate(dets):
        # Generate a random BGR color for each detection
        random_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        box = (xc, yc, w, h)
       
        # Use the new random_color variable
        draw_box(annotated, box, f"{idx}: {lbl}", random_color)
        detections.append({
            "x": xc, "y": yc, "w": w, "h": h, "label": lbl, "conf": conf,
        })

    # ... (rest of the function)

    ok, buf = cv2.imencode(".jpg", annotated)
    snapshot_b64 = base64.b64encode(buf).decode("utf-8") if ok else None

    return render_template_string(
        BASELINE_ASSISTANT_TEMPLATE,
        snapshot_b64=snapshot_b64,
        detections=detections,
        trackables=TRACKABLE_CLASSES
    )

@app.route("/baseline/add", methods=["POST"])
def baseline_add():
    count = int(request.form.get("count", "0"))
    selected = request.form.getlist("select")
    if not selected:
        flash("No detections selected.")
        return redirect(url_for('baseline_assist'))

    added = 0
    for s in selected:
        i = int(s)
        if i < 0 or i >= count:
            continue
        x = float(request.form.get(f"x_{i}"))
        y = float(request.form.get(f"y_{i}"))
        w = float(request.form.get(f"w_{i}"))
        h = float(request.form.get(f"h_{i}"))
        lbl = request.form.get(f"label_{i}")
        
        custom = request.form.get(f"custom_{i}", "").strip()

        display_label = custom if custom else lbl 

        duplicate = any(iou_xywh((x,y,w,h), b["box"]) > IOU_SAME_OBJECT for b in baseline)
        if duplicate:
            continue

        obj = {
            "id": next_baseline_id(),
            "label": lbl,
            "display_label": display_label,
           
            "box": (x, y, w, h),
            "absent_since": None,
            "missing": False,
            "missing_since": None,
            "occluded": False,
            "occluded_since": None
        }
        baseline.append(obj)
        added += 1

    if added > 0:
        con = init_db()
        save_baseline_to_db(con, baseline)
        con.close()
        flash(f"Added {added} item(s) to baseline.")
    else:
        flash("Nothing added (possible duplicates).")

    return redirect(url_for('baseline_assist'))

# -----------------------------
# Video generator & monitoring
# -----------------------------
def generate_frames():
    global model, baseline, baseline_ready, bg, current_frame, frame_buffer, stream_reader

    if model is None:
        model = YOLO(MODEL_PATH)

    if bg is None:
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    con = init_db()
    stable_frames = 0
    bg_warm = 0

    if not stream_reader or not stream_reader.started:
        # Handle case where stream is not yet started
        error_frame = 255 * np.ones((540, 960, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error: Stream not started", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    while True:
        ok, frame = stream_reader.read()
        if not ok:
            # If a frame can't be read, it's likely a stream issue.
            # The stream_reader thread will handle the reconnection.
            # We just need to yield an error frame.
            error_frame = 255 * np.ones((540, 960, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error: Camera Disconnected", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1) # Sleep to prevent a tight loop on a disconnected stream
            continue

        frame_ts = now_ts()
        with frame_lock:
            frame_buffer.append((frame_ts, frame.copy()))
            current_frame = frame.copy()
            cutoff = frame_ts - (FRAME_BUFFER_SECONDS + 1.0)
            while frame_buffer and frame_buffer[0][0] < cutoff:
                frame_buffer.popleft()

        # Your existing processing logic continues from here
        fgmask = bg.apply(frame)
        bg_warm += 1
        detections = extract_detections(frame, model)
        # ... (rest of your run_monitoring and display logic)]

        # ---- MANUAL MODE: do NOT auto-build baseline ----
        if BASELINE_MODE == "manual":
            if not baseline_ready:
                cv2.putText(frame, "Manual baseline mode: use Baseline Assistant to add objects.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                for d in detections:
                    draw_box(frame, d[0:4], f"{d[4]} (detected)", (255,255,0))
            else:
                run_monitoring(frame, frame_ts, detections, fgmask, con)
        else:
            # ---- AUTO MODE: automatic baseline building ----
            if not baseline_ready:
                new_added = False
                for xc, yc, w, h, lbl, conf in detections:
                    box = (xc, yc, w, h)
                    duplicate = any(iou_xywh(box, b["box"]) > IOU_SAME_OBJECT for b in baseline)
                    if not duplicate:
                      
                        display_label = lbl
                        baseline.append({
                            "id": next_baseline_id(),
                            "label": lbl,
                            "display_label": display_label,
                            
                            "box": box,
                            "absent_since": None,
                            "missing": False,
                            "missing_since": None,
                            "occluded": False,
                            "occluded_since": None
                        })
                        draw_box(frame, box, f"Secured: (ID: {baseline[-1]['id']}) {display_label}", (0,255,0))
                        new_added = True

                stable_frames = 0 if new_added else (stable_frames + 1)
                if stable_frames > 30 and bg_warm > WARMUP_FRAMES_BG and baseline:
                    baseline_ready = True
                    save_baseline_to_db(con, baseline)

                cv2.putText(frame, "Securing baseline objects (AUTO)...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            else:
                run_monitoring(frame, frame_ts, detections, fgmask, con)

        # overlay live counts
        draw_counts(frame, count_by_label(detections))

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def run_monitoring(frame, frame_ts, detections, fgmask, con):
    """
    frame: current ndarray
    frame_ts: timestamp (float seconds)
    detections: list of (xc,yc,w,h,label,conf)
    fgmask: foreground mask used for occlusion heuristic
    con: sqlite connection
    """
    present_baseline_ids = set()
    new_detections_by_label = {}
    for det in detections:
        xc, yc, w, h, lbl, conf = det
        found_match = False
        for obj in baseline:
            if iou_xywh(obj["box"], (xc, yc, w, h)) > IOU_PRESENCE:
                present_baseline_ids.add(obj["id"])
                found_match = True
                break
        if not found_match:
            if lbl not in new_detections_by_label:
                new_detections_by_label[lbl] = []
            new_detections_by_label[lbl].append(det)

    for obj in baseline:
        is_present = obj["id"] in present_baseline_ids
        is_occluded = False

        if not is_present:
            x1, y1, x2, y2 = clamp_box_to_frame(obj["box"], frame.shape)
            roi_mask = fgmask[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if roi_mask.size > 0:
                fg_ratio = (roi_mask > 0).sum() / float(roi_mask.size)
                is_occluded = fg_ratio >= OCCLUSION_FG_RATIO

        is_out_of_place = not is_present and not is_occluded and obj["label"] in new_detections_by_label
        base_label = f"(ID: {obj['id']}) {obj['display_label']}"

        if is_present:
            # Reset any absent/occlusion timers
            obj["absent_since"] = None
            if obj.get("occluded"):
                obj["occluded"] = False
                obj["occluded_since"] = None
            if obj["missing"]:
                details = ""
                if obj["missing_since"]:
                    try:
                        t0 = datetime.strptime(obj["missing_since"], "%Y-%m-%d %H:%M:%S")
                        sec = (datetime.now() - t0).total_seconds()
                        details = f"restored after {sec:.1f}s"
                    except:
                        details = "restored"
                # insert RESTORED event; clear previous MISSING image if desired
                insert_event(con, "RESTORED", obj["id"], obj["display_label"], details, image_data=None, extra_images=None)
                clear_missing_event_image(con, obj["id"])

            obj["missing"] = False
            obj["missing_since"] = None
            upsert_status(con, obj["id"], obj["display_label"], 0, None)
            draw_box(frame, obj["box"], base_label, (0,255,0))

        elif is_occluded:
            # If occlusion just started, mark it and save OCCLUDED event + buffered images
            if not obj.get("occluded", False):
                obj["occluded"] = True
                obj["occluded_since"] = now_str()
                # Save full-frame snapshot (Base64) for DB quick preview
                is_ok, buffer = cv2.imencode('.jpg', frame)
                img_as_text = base64.b64encode(buffer).decode('utf-8') if is_ok else None
                # insert event first to get its event_id for naming images
                event_id = insert_event(con, "OCCLUDED", obj["id"], obj["display_label"], "object is occluded", image_data=img_as_text, extra_images=None)
                # Save buffered frames as timestamped JPEGs
                saved_paths = save_buffered_images_for_event(event_id, frame_ts, buffer_seconds=FRAME_BUFFER_SECONDS)
                if saved_paths:
                    update_event_extra_images(con, event_id, ",".join(saved_paths))
            draw_box(frame, obj["box"], f"{base_label} (occluded)", (0,165,255))

            # Optionally escalate occluded -> missing if occlusion lasted longer than OCCLUSION_ESCALATE_SECONDS
            if obj.get("occluded_since"):
                try:
                    t0 = datetime.strptime(obj["occluded_since"], "%Y-%m-%d %H:%M:%S")
                    occl_secs = (datetime.now() - t0).total_seconds()
                    if occl_secs >= OCCLUSION_ESCALATE_SECONDS and not obj.get("missing", False):
                        # escalate to MISSING
                        obj["missing"] = True
                        obj["missing_since"] = now_str()
                        is_ok, buffer = cv2.imencode('.jpg', frame)
                        img_as_text = base64.b64encode(buffer).decode('utf-8') if is_ok else None
                        event_id = insert_event(con, "MISSING", obj["id"], obj["display_label"], f"after {occl_secs:.1f}s of occlusion", image_data=img_as_text, extra_images=None)
                        saved_paths = save_buffered_images_for_event(event_id, frame_ts, buffer_seconds=FRAME_BUFFER_SECONDS)
                        if saved_paths:
                            update_event_extra_images(con, event_id, ",".join(saved_paths))
                        upsert_status(con, obj["id"], obj["display_label"], 1, obj["missing_since"])
                except Exception:
                    pass

        elif is_out_of_place:
            # If out of place just detected, log it immediately and save buffered frames
            if not obj.get("missing", False):
                obj["missing"] = True
                obj["missing_since"] = now_str()
                is_ok, buffer = cv2.imencode('.jpg', frame)
                img_as_text = base64.b64encode(buffer).decode('utf-8') if is_ok else None
                event_id = insert_event(con, "OUT_OF_PLACE", obj["id"], obj["display_label"], "object of same type detected elsewhere", image_data=img_as_text, extra_images=None)
                saved_paths = save_buffered_images_for_event(event_id, frame_ts, buffer_seconds=FRAME_BUFFER_SECONDS)
                if saved_paths:
                    update_event_extra_images(con, event_id, ",".join(saved_paths))
                upsert_status(con, obj["id"], obj["display_label"], 1, obj["missing_since"])
            draw_box(frame, obj["box"], f"{base_label} Out of Place!", (0,165,255))

        else:
            # Not present and not occluded and not out_of_place — apply missing grace period
            if obj.get("absent_since") is None:
                obj["absent_since"] = now_str()
            else:
                # compute duration absent
                try:
                    t0 = datetime.strptime(obj["absent_since"], "%Y-%m-%d %H:%M:%S")
                    sec = (datetime.now() - t0).total_seconds()
                except Exception:
                    sec = 0

                if sec >= MISSING_GRACE_SECONDS and not obj.get("missing", False):
                    # Now declare MISSING
                    obj["missing"] = True
                    obj["missing_since"] = now_str()
                    is_ok, buffer = cv2.imencode('.jpg', frame)
                    img_as_text = base64.b64encode(buffer).decode('utf-8') if is_ok else None
                    event_id = insert_event(con, "MISSING", obj["id"], obj["display_label"], f"absent for {sec:.1f}s", image_data=img_as_text, extra_images=None)
                    # Save buffered frames
                    saved_paths = save_buffered_images_for_event(event_id, frame_ts, buffer_seconds=FRAME_BUFFER_SECONDS)
                    if saved_paths:
                        update_event_extra_images(con, event_id, ",".join(saved_paths))
                    upsert_status(con, obj["id"], obj["display_label"], 1, obj["missing_since"])

            draw_box(frame, obj["box"], f"{base_label} Missing!", (0,0,255))

            if obj.get("missing") and obj.get("missing_since"):
                x, y, w, h = obj["box"]
                p = (int(x - w/2), int(y - h/2) - 35)
                cv2.putText(frame, f"since {obj['missing_since'].split(' ')[1]}", p,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Draw boxes for new detections (not baseline)
    for lbl, dets in new_detections_by_label.items():
        for det in dets:
            draw_box(frame, det[0:4], f"NEW {det[4]}", (255,255,0))

# -----------------------------
# Video endpoints
# -----------------------------
@app.route("/video_feed")
def video_feed():
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/release_stream")
def release_stream():
    global camera
    if camera:
        camera.release()
        camera = None
    return "Stream released", 200

# Serve event images (optional)
@app.route("/event_images/<path:filename>")
def serve_event_image(filename):
    return send_from_directory(str(EVENT_IMAGES_DIR.resolve()), filename)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    init_db()
    # Ensure event images dir exists
    EVENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    # Register routes (already decorated, but keep startup actions)
    app.run(host="0.0.0.0", port=5000, debug=False)
