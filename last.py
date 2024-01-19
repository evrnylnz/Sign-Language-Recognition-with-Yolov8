from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import torch
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables to track consecutive predictions and the last predicted letter
consecutive_predictions = 0
last_predicted_letter = None
threshold = 0.8

@app.route('/')
def index():
    return render_template('index2.html', predicted_letter=last_predicted_letter)

def detect_objects():
    global consecutive_predictions
    global last_predicted_letter

    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height

    model = YOLO("C:/Users/evrny/Desktop/yolo_try/runs/detect/train6/weights/best.pt")

    while True:
        ret, frame = cap.read()
        results = model(frame)[0]

        current_predicted_letter = None

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                current_predicted_letter = results.names[int(class_id)].upper()

        # Check if the current letter is the same as the last predicted letter
        if current_predicted_letter == last_predicted_letter:
            consecutive_predictions += 1
        else:
            consecutive_predictions = 0

        # Check if the same letter has been predicted for 2 seconds
        if consecutive_predictions >= 20:  # Assuming 10 frames per second
            print(f"Predicted Letter: {last_predicted_letter}")
            socketio.emit('update_letter', {'letter': last_predicted_letter})  # Send update to the client
            consecutive_predictions = 0

        last_predicted_letter = current_predicted_letter

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    threading.Thread(target=detect_objects).start()  # Run detect_objects in a separate thread

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
