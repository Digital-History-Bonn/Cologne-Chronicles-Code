import glob
from os.path import basename
from typing import List, Optional

import torch
from matplotlib import pyplot as plt
from torchvision.ops import box_iou
from ultralytics import YOLO


CLASS_ASSIGNMENTS = {
    0: "unknown",
    1: "caption",
    2: "table",
    3: "article",
    4: "heading",
    5: "header",
    6: "separator_vertical",
    7: "separator_horizontal",
    8: "image",
    9 : "inverted_text"
}

colors = ['tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:pink',
          'tab:brown', 'tab:cyan', 'tab:gray']


def plot_target(image, tar_cls, tar_bboxs, title: Optional[str] = 'Page'):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()

    # Draw each bounding box
    for class_id, bbox in zip(tar_cls, tar_bboxs):
        x_min, y_min, x_max, y_max = bbox

        # Create a rectangle patch
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=1, edgecolor=colors[int(class_id)], facecolor='none')
        ax.add_patch(rect)
        # Add class ID text above the bounding box
        plt.text(x_min, y_min - 5, f"{CLASS_ASSIGNMENTS[int(class_id)]}",
                 color='black', fontsize=6, backgroundcolor=None)

    plt.axis('off')
    plt.title(title)
    plt.show()


def iou(pred_bboxs: torch.Tensor,
        target_bboxs: torch.Tensor,
        pred_cls: torch.Tensor,
        target_cls: torch.Tensor):

    matrix = box_iou(pred_bboxs, target_bboxs)
    matrix[~(pred_cls.unsqueeze(1) == target_cls.unsqueeze(0))] = 0.0
    return matrix

def read_txt(target, shape):
    classes = []
    coordinates = []

    with open(target, 'r') as file:
        for line in file:
            values = line.strip().split()
            classes.append(int(values[0]))
            x_center, y_center = float(values[1]) * shape[1], float(values[2]) * shape[0]
            width, height = float(values[3]) * shape[1], float(values[4]) * shape[0]
            coordinates.append([x_center - width / 2,
                                y_center - height / 2,
                                x_center + width / 2,
                                y_center + height / 2])

    return torch.tensor(classes, dtype=torch.long), torch.tensor(coordinates, dtype=torch.float32)


def metrics(pred_bboxs, tar_bboxs, pred_cls, tar_cls, threshold: float= 0.5):
    tar_count = torch.bincount(tar_cls, minlength=10)
    pred_count = torch.bincount(pred_cls, minlength=10)

    matrix = iou(pred_bboxs, tar_bboxs, pred_cls, tar_cls)

    tps = torch.zeros(10)
    tp = 0

    while True:
        max_index = torch.argmax(matrix)
        max_row = max_index // matrix.size(1)
        max_col = max_index % matrix.size(1)
        max_value = matrix[max_row, max_col]

        if max_value < threshold:
            break

        matrix[max_row, :] = 0
        matrix[:, max_col] = 0

        tps[tar_cls[max_col]] += 1
        tp += 1

    fps = pred_count - tps
    fns = tar_count - tps

    fp = matrix.shape[0] - tp
    fn = matrix.shape[1] - tp

    precisions = tps / (tps + fps + 1e-10)
    recalls = tps / (tps + fns + 1e-10)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    return (tar_count,
            precision,
            recall,
            f1_score,
            precisions,
            recalls,
            f1_scores)


def predict(model: YOLO, images: List[str], targets: Optional[List[str]] = None):
    # Predict with the model
    results = model(images)

    counts = torch.zeros(10)
    avg_recalls = torch.zeros(10)
    avg_precisions = torch.zeros(10)
    avg_f1s = torch.zeros(10)

    avg_recall = 0
    avg_precision = 0
    avg_f1 = 0


    for result, target in zip(results, targets):
        shape = result.orig_shape
        tar_cls, tar_bboxs = read_txt(target, shape)
        # plot_target(result.orig_img, tar_cls, tar_bboxs, title="target")

        pred_bboxs = result.boxes.xyxy  # Boxes object for bounding box outputs
        pred_cls = result.boxes.cls.int()
        # plot_target(result.orig_img, pred_cls, pred_bboxs, title="prediction")

        print(f"{result.path}")
        for threshold in [0.9]:
            tar_count, precision, recall, f1_score, precisions, recalls, f1_scores = metrics(pred_bboxs, tar_bboxs, pred_cls, tar_cls, threshold=threshold)

            counts += tar_count.bool()
            avg_recalls += recalls
            avg_precisions += precisions
            avg_f1s += f1_scores

            avg_recall += recall
            avg_precision += precision
            avg_f1 += f1_score

            # print(f"{threshold=}")
            # print(f"\t{precision=}")
            # print(f"\t{recall=}")
            # print(f"\t{f1_score=}")
            # print(f"{tar_count=}")
            # print(f"\t{precisions=}")
            # print(f"\t{recalls=}")
            # print(f"\t{f1_scores=}")
            # print()
        # print("\n\n")

    print(f"Overall:")
    print(f"\trecall:{avg_recall / counts.sum():.4f}")
    print(f"\tprecision:{avg_precision / counts.sum():.4f}")
    print(f"\tf1 score:{avg_f1 / counts.sum():.4f}\n"),

    for idx, (r, p, f) in enumerate(zip(avg_recalls / counts, avg_precisions / counts, avg_f1s / counts)):
        print(f"{CLASS_ASSIGNMENTS[idx]}:")
        print(f"\trecall:{r:.4f}")
        print(f"\tprecision:{p:.4f}")
        print(f"\tf1 score:{f:.4f}\n"),



def main(image_path: str, target_path: Optional[str] = None):
    # load model
    model = YOLO("models/yolov8_1.pt")
    images = list(glob.glob(f"{image_path}/*.jpg"))
    
    target = [f"{target_path}/{basename(file)[:-4]}.txt" for file in images] if target_path else None
    
    predict(model, images, target)

if __name__ == '__main__':
    image_path = "data/YOLO_dataset/test/images"
    target_path = "data/YOLO_dataset/test/labels"
    main(image_path, target_path)
