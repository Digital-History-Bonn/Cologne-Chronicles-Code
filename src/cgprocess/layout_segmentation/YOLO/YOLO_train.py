from ultralytics import YOLO


def main(task: str):
    # Load a model
    model_file = {"detect": "yolov8n.pt",
                  "segment": "yolov8n-seg.pt",
                  "pose": "yolov8n-pose.pt"}[task]
    model = YOLO(model_file)

    # Train the model
    yaml = {"detect": "data/YOLO_dataset/CGD.yaml",
                  "segment": "data/YOLO_Textlines/CGD.yaml",
                  "pose": "data/YOLO_Baselines/CGD.yaml"}[task]
    results = model.train(data=yaml,
                          epochs=500,
                          imgsz=2048 if task == 'detect' else 1024,
                          batch=16,
                          device=[0, 1])

   
if __name__ == '__main__':
    main(task='segment')
