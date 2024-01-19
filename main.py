from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.yaml")

    # Train the model
    model.train(data="C:/Users/evrny/Desktop/yolo_try/config.yaml", epochs=5)

if __name__ == '__main__':
    main()


