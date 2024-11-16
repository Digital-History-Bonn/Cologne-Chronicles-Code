import argparse

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


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="predict")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="detect",
        help="task to train must be \"detect\" or \"segment\" or \"pose\""
    )

    return parser.parse_args()

   
if __name__ == '__main__':
    args = get_args()
    main(task=args.task)
