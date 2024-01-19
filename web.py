from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import torch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def detect_objects():
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height

    model = YOLO("C:/Users/evrny/Desktop/yolo_try/runs/detect/train6/weights/best.pt")
    threshold = 0.8

    torch.cuda.set_device(0)

    while True:
        ret, frame = cap.read()
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
