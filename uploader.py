import cv2
import requests
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=3)

def upload_event_with_image(backend_url, tid, face_img, confidence, label, camera_id, is_known):
    """
    Sends a single request containing both the Event Data and the Image File.
    """
    success, buffer = cv2.imencode('.jpg', face_img)
    if not success:
        return
    # 1. Prepare the Event Data (Everything previously in send_event)
    data = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "camera_id": camera_id,
        "track_id": f"track_{tid}",
        "person_id": label,
        "is_known": str(is_known).lower(), # Convert bool to string for form-data
        "confidence": round(float(confidence), 3)
    }
    # 2. Prepare the Image File
    files = {'face_img': (f'{label}.jpg', buffer.tobytes(), 'image/jpeg')}
    # 3. Background execution
    executor.submit(_execute_combined_request, backend_url, files, data)

def _execute_combined_request(url, files, data):
    try:
        # One request to rule them all
        response = requests.post(url,  data=data, files=files, timeout=5)
        if response.status_code == 201:
            print(f"[Combined] Event & Image synced for {data['person_id']}")
        else:
            print(f"[Server Error] {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Network] Combined upload failed: {e}")