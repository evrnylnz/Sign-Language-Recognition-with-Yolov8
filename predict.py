import os
from ultralytics import YOLO
import cv2
import torch

VIDEOS_DIR = os.path.join('C:/Users/evrny/Desktop', 'videos') 

video_path = os.path.join(VIDEOS_DIR, 'letters.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

print(video_path)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Error reading the video file. Check if the file exists and is valid.")
    exit()

H, W, _ = frame.shape
# ...
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
# ...

model_path = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')
print(model_path)
# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.8

torch.cuda.set_device(0)

while ret:
    results = model(frame)[0]
    print(results)
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
