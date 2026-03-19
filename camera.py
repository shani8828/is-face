import cv2
from flask import Flask, Response

app = Flask(__name__)

# Open laptop webcam
camera = cv2.VideoCapture(0)

# Optional: Set resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # 0.0.0.0 makes it accessible on LAN
    app.run(host="0.0.0.0", port=5000, threaded=True)