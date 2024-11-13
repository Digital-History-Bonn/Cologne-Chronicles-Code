from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="data/YOLO_dataset/CGD.yaml", epochs=1, imgsz=2048)
    
   
if __name__ == '__main__':
    main()
