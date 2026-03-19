import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np
import uuid
import requests
import time
from datetime import datetime
from insightface.app import FaceAnalysis
import socket
import threading
import webbrowser
from flask import Flask, Response, render_template_string

# ============================================
# FLASK & NETWORK SETUP
# ============================================
app = Flask(__name__)
lock = threading.Lock()
output_frame = None
global_fps = 0

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

LOCAL_IP = get_local_ip()

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            # Encode the processed frame to JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01) # Small sleep to prevent CPU hogging

@app.route('/')
def dashboard():
    return render_template_string(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Smart Lobby Monitoring Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
    body {{
        margin: 0;
        background: linear-gradient(135deg, #280C49, #4B2B83);
        font-family: 'Poppins', sans-serif;
        color: white;
        text-align: center;
    }}
    .header {{
        padding: 18px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 28px;
        letter-spacing: 2px;
        background: #4B2B83;
        border-bottom: 3px solid #DCB477;
    }}
    .container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        padding: 30px 40px;
    }}
    .card {{
        background: #3A175C;
        padding: 25px;
        border-radius: 15px;
        width: 240px;
        text-align: left;
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
    }}
    .card h3 {{
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        color: #DCB477;
        margin-bottom: 15px;
    }}
    .video-container {{
        position: relative;
        border: 4px solid #DCB477;
        border-radius: 15px;
        overflow: hidden;
    }}
    .video-container img {{
        width: 640px;
        height: auto;
        display: block;
    }}
    .footer {{ margin-top: 15px; }}
    .copy-btn {{
        background: #A37930;
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        font-family: 'Poppins', sans-serif;
    }}
    .copy-btn:hover {{ background: #DCB477; }}
    .url-display {{
        margin-top: 10px;
        font-size: 14px;
        color: #DCB477;
    }}
    .status-live {{ color: #00FF9D; font-weight: 600; }}
    </style>
    </head>
    <body>
    <div class="header">SMART LOBBY MONITORING DASHBOARD</div>
    <div class="container">
        <div class="card">
            <h3>Session Info</h3>
            <p><b>Device:</b> Main Camera</p>
            <p><b>Status:</b> <span class="status-live">Live</span></p>
            <p><b>FPS:</b> <span id="fps">{int(global_fps)}</span></p>
            <p><b>Session Time:</b> <span id="timer">0 sec</span></p>
        </div>
        <div>
            <div class="video-container">
                <img src="/video">
            </div>
            <div class="footer">
                <button class="copy-btn" onclick="copyURL()">Copy Dashboard URL</button>
                <div class="url-display" id="urlText">http://{LOCAL_IP}:5000</div>
            </div>
        </div>
        <div class="card">
            <h3>Network Info</h3>
            <p><b>IP Address:</b> {LOCAL_IP}</p>
            <p><b>Port:</b> 5000</p>
            <p><b>Mode:</b> Production</p>
            <p><b>System:</b> Ready</p>
        </div>
    </div>
    <script>
    let startTime = Date.now();
    setInterval(() => {{
        let seconds = Math.floor((Date.now() - startTime) / 1000);
        document.getElementById("timer").innerText = seconds + " sec";
    }}, 1000);
    function copyURL() {{
        const text = "http://{LOCAL_IP}:5000";
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        textarea.setSelectionRange(0, 99999);
        document.execCommand("copy");
        document.body.removeChild(textarea);
        alert("Dashboard URL Copied!");
    }}
    </script>
    </body>
    </html>
    """)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    # Flask runs in a background thread so it doesn't block OpenCV
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ============================================
# PERFORMANCE & THRESHOLDS
# ============================================
FRAME_SKIP = 3
CAM_W, CAM_H = 640, 480
DET_SIZE = (416, 416)

BASE_KNOWN_THRESHOLD = 0.65
UNKNOWN_MATCH_THRESHOLD = 0.60
TRACK_SCORE_THRESHOLD = 0.5

MIN_DET_SIZE = 35
MIN_RECOG_SIZE = 70

MAX_TRACK_DIST = 130
MAX_MISSED = 6
MAX_UNKNOWN_CACHE = 300

MIN_STABLE_AGE = 4
MIN_HISTORY = 3
UNKNOWN_DECISION_AGE = 6

# ============================================
# BACKEND CONFIG
# ============================================
CAMERA_ID = "cam_01"
BACKEND_URL = "https://is-server-qczc.onrender.com/api/ingest"
SEND_TO_BACKEND = True

# ============================================
# MODEL & CAMERA INIT
# ============================================
print("Loading Face Analysis Model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DET_SIZE)

try:
    known_db = np.load("known_db.npy", allow_pickle=True).item()
    known_names = list(known_db.keys())
    known_matrix = np.array([known_db[n] for n in known_names])
    known_matrix = known_matrix / np.linalg.norm(known_matrix, axis=1, keepdims=True)
except FileNotFoundError:
    print("Warning: known_db.npy not found. Proceeding without known faces.")
    known_names = []
    known_matrix = np.empty((0, 512))

# *** CRITICAL CHANGE: Reading directly from the local webcam now ***
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

# ============================================
# MEMORY & HELPERS
# ============================================
tracks = {}
next_track_id = 0
unknown_counter = 0
unknown_cache = {}
frame_count = 0
prev_time = time.time()

def cos_sim(a, b):
    return np.dot(a, b)

def centroid(b):
    x1, y1, x2, y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def send_event(tid, label, confidence, is_known):
    if not SEND_TO_BACKEND: return
    payload = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "camera_id": CAMERA_ID,
        "track_id": f"track_{tid}",
        "is_known": is_known,
        "person_id": label,
        "confidence": float(confidence)
    }
    try:
        requests.post(BACKEND_URL, json=payload, timeout=2)
        print("Sent:", payload)
    except Exception as e:
        print("Backend error:", e)

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print(f"Starting Smart Lobby Dashboard at: http://{LOCAL_IP}:5000")
    
    # Open browser automatically
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{LOCAL_IP}:5000")).start()
    
    # Start Flask server in a background daemon thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    print("FaceRecog1 Stable Unknown Re-ID Version Running")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame_count += 1
        run_heavy = (frame_count % FRAME_SKIP == 0)

        if run_heavy:
            faces = face_app.get(frame)
            detections = []

            for f in faces:
                if f.det_score < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, f.bbox)
                w = x2 - x1
                h = y2 - y1

                if w < MIN_DET_SIZE or h < MIN_DET_SIZE:
                    continue

                emb = f.embedding
                emb = emb / np.linalg.norm(emb)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "emb": emb,
                    "size": w
                })

            used_tracks = set()

            # ---------------- TRACKING ----------------
            for det in detections:
                c = centroid(det["bbox"])
                best_id = None
                best_score = -1

                for tid, t in tracks.items():
                    tc = t["centroid"]
                    dist = np.linalg.norm(np.array(c) - np.array(tc))

                    if dist > MAX_TRACK_DIST:
                        continue

                    pos_score = 1 - dist / MAX_TRACK_DIST
                    emb_score = cos_sim(det["emb"], t["stable_emb"])
                    score = emb_score * 0.6 + pos_score * 0.4

                    if score > best_score:
                        best_score = score
                        best_id = tid

                if best_score > TRACK_SCORE_THRESHOLD:
                    t = tracks[best_id]
                    t["bbox"] = det["bbox"]
                    t["centroid"] = c
                    t["miss"] = 0
                    t["age"] += 1
                    t["face_size"] = det["size"]

                    t["emb_history"].append(det["emb"])
                    if len(t["emb_history"]) > 5:
                        t["emb_history"].pop(0)

                    stable = np.mean(t["emb_history"], axis=0)
                    stable /= np.linalg.norm(stable)
                    t["stable_emb"] = stable

                    used_tracks.add(best_id)
                else:
                    tracks[next_track_id] = {
                        "bbox": det["bbox"],
                        "centroid": c,
                        "miss": 0,
                        "age": 1,
                        "face_size": det["size"],
                        "emb_history": [det["emb"]],
                        "stable_emb": det["emb"],
                        "label": None,
                        "confidence": None,
                        "event_sent": False
                    }
                    used_tracks.add(next_track_id)
                    next_track_id += 1

            # ---------------- REMOVE LOST ----------------
            for tid in list(tracks.keys()):
                if tid not in used_tracks:
                    tracks[tid]["miss"] += 1
                    if tracks[tid]["miss"] > MAX_MISSED:
                        tracks.pop(tid)

            # ---------------- RECOGNITION ----------------
            for tid, t in tracks.items():
                if t["event_sent"] or t["age"] < MIN_STABLE_AGE or len(t["emb_history"]) < MIN_HISTORY or t["face_size"] < MIN_RECOG_SIZE:
                    continue

                emb = t["stable_emb"]

                # Dynamic threshold
                known_threshold = 0.70 if t["face_size"] < 90 else BASE_KNOWN_THRESHOLD

                if len(known_names) > 0:
                    sims = known_matrix @ emb
                    best_idx = np.argmax(sims)
                    best_sim = sims[best_idx]
                else:
                    best_sim = 0

                # -------- KNOWN --------
                if best_sim >= known_threshold:
                    name = known_names[best_idx]
                    t["label"] = name
                    t["confidence"] = best_sim
                    t["event_sent"] = True
                    send_event(tid, name, best_sim, True)

                # -------- UNKNOWN RE-ID --------
                elif t["age"] >= UNKNOWN_DECISION_AGE:
                    best_uid = None
                    best_u_sim = -1

                    for uid, uemb in unknown_cache.items():
                        sim = cos_sim(emb, uemb)
                        if sim > best_u_sim:
                            best_u_sim = sim
                            best_uid = uid

                    if best_u_sim >= UNKNOWN_MATCH_THRESHOLD:
                        label = f"unknown_{best_uid}"
                        t["label"] = label
                        t["confidence"] = best_u_sim
                        t["event_sent"] = True

                        updated = 0.7 * unknown_cache[best_uid] + 0.3 * emb
                        unknown_cache[best_uid] = updated / np.linalg.norm(updated)
                        send_event(tid, label, best_u_sim, False)
                    else:
                        unknown_counter += 1
                        uid = unknown_counter
                        label = f"unknown_{uid}"

                        t["label"] = label
                        t["confidence"] = 1.0
                        t["event_sent"] = True

                        unknown_cache[uid] = emb
                        if len(unknown_cache) > MAX_UNKNOWN_CACHE:
                            first_key = next(iter(unknown_cache))
                            unknown_cache.pop(first_key)

                        send_event(tid, label, 1.0, False)

        # ====================================
        # LIVE COUNTERS & FPS
        # ====================================
        current_known = sum(1 for t in tracks.values() if t["miss"] == 0 and t["label"] and not t["label"].startswith("unknown"))

        current_time = time.time()
        global_fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # ====================================
        # DRAW PANEL & BOUNDING BOXES
        # ====================================
        cv2.rectangle(frame, (5,5), (360,110), (0,0,0), -1)
        cv2.rectangle(frame, (5,5), (360,110), (255,255,255), 2)

        cv2.putText(frame, f"Known in Lobby: {current_known}", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Unique Unknown: {unknown_counter}", (15,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"FPS: {int(global_fps)}", (15,95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        for t in tracks.values():
            if t["miss"] > 0: continue
            x1, y1, x2, y2 = t["bbox"]
            label = t["label"]

            color = (0,255,0) if label and not label.startswith("unknown") else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            if label:
                cv2.putText(frame, label, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pass the fully processed frame to the Flask web dashboard
        with lock:
            output_frame = frame.copy()

        # Display locally
        #cv2.imshow("Smart Lobby Monitor", frame)

        #if cv2.waitKey(1) == 27:
            #break

cap.release()
cv2.destroyAllWindows()
