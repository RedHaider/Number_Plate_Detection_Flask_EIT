from flask import Flask, Response, render_template, jsonify, request, url_for
import cv2
import os
import threading
import time
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the Haar Cascade for plate detection
harcascade = "models/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Frame holder for threading
frame_holder = {
    "frame": None
}

# Lock for accessing the frame
frame_lock = threading.Lock()

def capture_frames():
    while True:
        success, frame = cap.read()
        if success:
            with frame_lock:
                frame_holder["frame"] = frame
        time.sleep(0.1)  # To reduce CPU usage

def generate_frames():
    while True:
        with frame_lock:
            frame = frame_holder["frame"]
        if frame is not None:
            img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(img_grey, 1.1, 4)
            for (x, y, w, h) in plates:
                area = w * h
                if area > 600:  # min_area
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 240, 150), 3)
                    cv2.putText(frame, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_plate(image):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_grey, 1.1, 4)
    plate_img = None
    for (x, y, w, h) in plates:
        area = w * h
        if area > 600:  # min_area
            plate_img = image[y:y+h, x:x+w]
            break
    if plate_img is None:
        print("No plate detected")
    return image, plate_img


@app.route('/api/capture', methods=['POST'])
def api_capture():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    img = Image.open(file.stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    full_img, plate_img = detect_plate(img)

    if not os.path.exists('plates'):
        os.makedirs('plates')

    full_image_path = os.path.join('plates', 'full_image.jpg')
    plate_image_path = os.path.join('plates', 'plate_image.jpg')

    cv2.imwrite(full_image_path, full_img)
    if plate_img is not None:
        cv2.imwrite(plate_image_path, plate_img)
    else:
        plate_image_path = None

    full_image_url = url_for('static', filename='plates/full_image.jpg', _external=True)
    plate_image_url = url_for('static', filename='plates/plate_image.jpg', _external=True) if plate_image_path else None

    return jsonify({
        "full_image_url": full_image_url,
        "plate_image_url": plate_image_url
    })

if __name__ == "__main__":
    # Start the thread to capture frames
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)