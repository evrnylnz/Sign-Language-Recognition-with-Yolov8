import os
from ultralytics import YOLO
import cv2
import torch

VIDEOS_DIR = os.path.join('C:/Users/evrny/Desktop/yolo_dataset/train', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'letters.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

print(video_path)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Error reading the video file. Check if the file exists and is valid.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('C:/Users/evrny/Desktop/yolo_try', 'runs', 'detect', 'train', 'weights', 'last.pt')
print(model_path)
# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0

fixed_width = 1920
fixed_height = 1056

while ret:
    
    # Resize frame to a shape divisible by 32
    frame = cv2.resize(frame, (fixed_width, fixed_height))

    # Convert the frame to the format expected by YOLO (BGR to RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize the image
    rgb_frame = rgb_frame.astype("float32") / 255.0
    
    # Convert to PyTorch tensor
    rgb_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1).unsqueeze(0)

    # Perform inference
    results = model(rgb_frame)[0]
    
    
    
    
    # Check if there are any detections
    
    for result in results.boxes.data.tolist():
        print(results)
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
