from ultralytics import YOLO


def main(task: str):
    # Load a model
    model = YOLO("yolov8n.pt" if task == 'detect' else "yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="data/YOLO_dataset/CGD.yaml" if task == 'detect' else "data/YOLO_Textlines/CGD.yaml",
                          epochs=500,
                          imgsz=2048 if task == 'detect' else 1024,
                          batch=16,
                          device=[0, 1])

   
if __name__ == '__main__':
    main(task='segment')
