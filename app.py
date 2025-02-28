from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import random
from threading import Lock

app = Flask(__name__)

# Load your pre-trained model and other initializations
face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Downloads\new\P226 Stress\Harcascade\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\Admin\Downloads\new\P226 Stress\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
camera = cv2.VideoCapture(0)

# Use threading to avoid freezing of Flask server
camera_lock = Lock()
camera = cv2.VideoCapture(0)
frame = None
stop_thread = False
latest_emotion = ''
latest_url = ''

def get_stress_level(label):
    if label in ['Happy', 'Surprise']:
        return "Stress Level-1: "+str(random.randint(0,10))+'%',""
    elif label in ['Neutral', 'Angry']:
        return "Stress Level-2: "+str(random.randint(30,50))+'%', "https://youtu.be/YoSuVws4OTQ"
    elif label in ['Disgust', 'Fear']:
        return "Stress Level-3: "+str(random.randint(60,80))+'%', "https://youtu.be/lHVYgnlukTw"
    return "Stress Level: Unknown", ""

def capture_frames():
    global frame, stop_thread, latest_emotion, latest_url
    while not stop_thread:
        _, frame = camera.read()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    stress_level, url = get_stress_level(label)

                    # Here we combine label and stress level in the message
                    message = f"{label} - {stress_level}"
                    latest_emotion = message  # Update with the combined message
                    latest_url = url  # Update URL but don't redirect yet

                    cv2.putText(frame, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        time.sleep(0.1)
        


def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            if camera.isOpened():
                camera.release()
            camera = None

def start_camera_capture():
    global camera, stop_thread, frame
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not open camera")
        stop_thread = False
        frame = None
        threading.Thread(target=capture_frames, daemon=True).start()

@app.route('/start_camera', methods=['POST'])
def start_camera():
    try:
        start_camera_capture()
        return jsonify(status="started")
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global stop_thread
    stop_thread = True
    time.sleep(0.1)  # Sleep briefly to let the capture thread stop
    release_camera()
    return jsonify(status="stopped", url=latest_url, last_emotion=latest_emotion)

@app.route('/video_feed')
def video_feed():
    global stop_thread, frame
    def gen_frames():
        while not stop_thread:
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
