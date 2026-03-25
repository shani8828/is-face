import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
import socket
import threading
import webbrowser
from flask import Flask, Response, render_template
from uploader import upload_event_with_image
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
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def dashboard():
    return render_template('index.html', local_ip=LOCAL_IP)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host="0.0.0.0", port=8828, debug=False, use_reloader=False)

# ============================================
# PERFORMANCE & THRESHOLDS
# ============================================
FRAME_SKIP = 3
CAM_W, CAM_H = 640, 480
DET_SIZE = (320, 320)

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
# NODE_BACKEND_CONFIG
# ============================================
CAMERA_ID = "cam_01"
UPLOAD_URL = "http://localhost:5000/api/upload-image"
# SEND_TO_BACKEND = True

# ============================================
# MODEL & CAMERA INIT
# ============================================
print("Loading Lightweight Face Analysis Model (buffalo_sc)...")
face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DET_SIZE)

try:
    known_db = np.load("known_db.npy", allow_pickle=True).item()
    known_names = list(known_db.keys())
    known_matrix = np.array([known_db[n] for n in known_names])
    if len(known_matrix) > 0:
        known_matrix = known_matrix / np.linalg.norm(known_matrix, axis=1, keepdims=True)
except FileNotFoundError:
    print("Warning: known_db.npy not found. Proceeding without known faces.")
    known_names = []
    known_matrix = np.empty((0, 512))

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

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print(f"Starting Smart Lobby Camera Streaming at: http://{LOCAL_IP}:8828")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{LOCAL_IP}:8828")).start()
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame_count += 1
        run_heavy = (frame_count % FRAME_SKIP == 0)
        if run_heavy:
            faces = face_app.get(frame)
            detections = []
            for f in faces:
                if f.det_score < 0.5: continue
                x1, y1, x2, y2 = map(int, f.bbox)
                w, h = x2 - x1, y2 - y1
                if w < MIN_DET_SIZE or h < MIN_DET_SIZE: continue
                emb = f.embedding / np.linalg.norm(f.embedding)
                detections.append({"bbox": [x1, y1, x2, y2], "emb": emb, "size": w})
            used_tracks = set()
            # ---------------- 1. TRACKING ----------------
            for det in detections:
                c = centroid(det["bbox"])
                best_id, best_score = None, -1
                for tid, t in tracks.items():
                    tc = t["centroid"]
                    dist = np.linalg.norm(np.array(c) - np.array(tc))
                    if dist > MAX_TRACK_DIST: continue
                    pos_score = 1 - dist / MAX_TRACK_DIST
                    emb_score = cos_sim(det["emb"], t["stable_emb"])
                    score = emb_score * 0.6 + pos_score * 0.4
                    if score > best_score:
                        best_score, best_id = score, tid
                if best_score > TRACK_SCORE_THRESHOLD:
                    t = tracks[best_id]
                    t.update({"bbox": det["bbox"], "centroid": c, "miss": 0, "age": t["age"]+1, "face_size": det["size"]})
                    t["emb_history"].append(det["emb"])
                    if len(t["emb_history"]) > 5: t["emb_history"].pop(0)
                    stable = np.mean(t["emb_history"], axis=0)
                    t["stable_emb"] = stable / np.linalg.norm(stable)
                    used_tracks.add(best_id)
                else:
                    tracks[next_track_id] = {
                        "bbox": det["bbox"], "centroid": c, "miss": 0, "age": 1, "face_size": det["size"],
                        "emb_history": [det["emb"]], "stable_emb": det["emb"], "label": None,
                        "confidence": 0.0, "event_sent": False, "last_sent_confidence": 0.0
                    }
                    used_tracks.add(next_track_id)
                    next_track_id += 1
            # ---------------- 2. REMOVE LOST ----------------
            for tid in list(tracks.keys()):
                if tid not in used_tracks:
                    tracks[tid]["miss"] += 1
                    if tracks[tid]["miss"] > MAX_MISSED: tracks.pop(tid)
            # ---------------- 3. RECOGNITION ----------------
            for tid, t in tracks.items():
                if t["event_sent"] or t["age"] < MIN_STABLE_AGE or t["face_size"] < MIN_RECOG_SIZE:
                    continue
                emb = t["stable_emb"]
                known_threshold = 0.70 if t["face_size"] < 90 else BASE_KNOWN_THRESHOLD
                # Check Known Database
                best_sim = 0
                if len(known_names) > 0:
                    sims = known_matrix @ emb
                    best_idx = np.argmax(sims)
                    best_sim = sims[best_idx]
                if best_sim >= known_threshold:
                    t["label"] = known_names[best_idx]
                    t["confidence"] = best_sim
                    t["event_sent"] = True
                # Check Unknown Re-ID Cache
                elif t["age"] >= UNKNOWN_DECISION_AGE:
                    best_uid, best_u_sim = None, -1
                    for uid, uemb in unknown_cache.items():
                        sim = cos_sim(emb, uemb)
                        if sim > best_u_sim:
                            best_u_sim, best_uid = sim, uid
                    if best_u_sim >= UNKNOWN_MATCH_THRESHOLD:
                        t["label"] = f"unknown_{best_uid}"
                        t["confidence"] = best_u_sim
                        t["event_sent"] = True
                        # Update cache identity
                        updated = 0.7 * unknown_cache[best_uid] + 0.3 * emb
                        unknown_cache[best_uid] = updated / np.linalg.norm(updated)
                    else:
                        unknown_counter += 1
                        t["label"] = f"unknown_{unknown_counter}"
                        t["confidence"] = 1.0
                        t["event_sent"] = True
                        unknown_cache[unknown_counter] = emb
                        if len(unknown_cache) > MAX_UNKNOWN_CACHE:
                            unknown_cache.pop(next(iter(unknown_cache)))
        # ---------------- 4. BEST-SHOT UPLOAD ----------------
        for tid, t in tracks.items():
            if t["label"] and t["miss"] == 0:
                current_conf = t["confidence"]
                last_sent = t.get("last_sent_confidence", 0)
                if current_conf > (last_sent + 0.05):
                    x1, y1, x2, y2 = t["bbox"]
                    h, w, _ = frame.shape
                    face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if face_crop.size > 0:
                        is_known_status = not t["label"].startswith("unknown")
                        upload_event_with_image(
                            backend_url=UPLOAD_URL,
                            tid=tid, 
                            face_img=face_crop,
                            confidence=current_conf, 
                            label=t["label"],
                            camera_id=CAMERA_ID,
                            is_known=is_known_status 
                        )
                        t["last_sent_confidence"] = current_conf
        # ---------------- 5. UI & FPS ----------------
        current_known = sum(1 for t in tracks.values() if t["miss"] == 0 and t["label"] and not t["label"].startswith("unknown"))
        current_time = time.time()
        global_fps = 1 / (current_time - prev_time)
        prev_time = current_time

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
        with lock:
            output_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()