import argparse
import glob
import json
import re
import warnings
from os.path import basename

import numpy as np
import torch
from bs4 import BeautifulSoup, PageElement
from shapely.geometry import Polygon
from skimage.draw import polygon as sk_polygon
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm

from src.cgprocess.layout_segmentation.utils import adjust_path

# Labels to evaluate
EVAL_LABELS = ["table", "paragraph", "caption", "heading", "header"]

MAPPING = {'article': 'paragraph',
           'inverted_text': 'paragraph',
           'paragraph': 'paragraph',
           'caption': 'caption',
           'heading': 'heading',
           'header': 'header',
           'newspaper_header': 'header',
           'headline': 'heading',
           'table': 'table'}


def get_tag(textregion: PageElement) -> str:
    """
    Returns the tag of the given textregion.

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'

    tag = match.group()[6:-2]
    if tag == 'article_':
        tag = "paragraph"
    if tag == 'article':
        tag = "paragraph"
    return tag


def multi_class_f1(
        pred: torch.Tensor, target: torch.Tensor, metric: MulticlassConfusionMatrix
) -> torch.Tensor:
    """Calculate csi score using true positives, true negatives and false negatives from confusion matrix.
    Csi score is used as substitute for accuracy, calculated separately for each class.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param target: target tensor
    :return:
    """
    pred = pred.flatten()
    target = target.flatten()

    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csi = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return csi


def read_json(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)

    bboxes = data.get('bboxes', [])

    region_coords = []
    region_labels = []

    for bbox_data in bboxes:
        label = bbox_data.get("class", "No Label")
        label = MAPPING.get(label, 'NoLabel')

        if label not in EVAL_LABELS:
            continue

        region_labels.append(label)

        bbox = bbox_data.get('bbox', {})

        # Convert bbox to a 2D tensor representing the polygon
        polygon = Polygon([
            [bbox['x0'], bbox['y0']],  # Top-left corner
            [bbox['x1'], bbox['y0']],  # Top-right corner
            [bbox['x1'], bbox['y1']],  # Bottom-right corner
            [bbox['x0'], bbox['y1']]  # Bottom-left corner
        ])

        region_coords.append(polygon)

    return region_coords, region_labels


def read_xml(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    width, height = int(page["imageWidth"]), int(page["imageHeight"])
    region_coords = []
    region_labels = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        coords = region.find_all('Coords')[0]
        label = get_tag(region)
        polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                point in coords['points'].split()])

        if len(polygon) <= 3:
            print(f"Found invalid TextRegion in {file_path}.")
            continue

        if label not in EVAL_LABELS:
            continue

        region_labels.append(label)
        region_coords.append(polygon)

    table_regions = page.find_all('TableRegion')
    for region in table_regions:
        coords = region.find_all('Coords')[0]
        polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                point in coords['points'].split()])
        if len(polygon) <= 3:
            print(f"Found invalid table in {file_path}.")
            continue

        region_labels.append('table')
        region_coords.append(polygon)

    return [Polygon(poly) for poly in region_coords], region_labels, height, width


def sort_polygons_and_labels(polygons, labels):
    """
    Sorts polygons and labels based on a given order of labels.

    :param polygons: List of polygons.
    :param labels: List of labels corresponding to the polygons.
    :return: A tuple containing two lists: sorted polygons and sorted labels.
    """
    # if lists are empty return
    if len(polygons) == 0:
        return [], []

    order = ['paragraph', 'table', 'heading', 'header']

    # Create a dictionary to map each label to its index in the order list
    order_index = {label: i for i, label in enumerate(order)}

    # Create a list of tuples containing (label, polygon) and sort it based on label's order index
    sorted_pairs = sorted(zip(labels, polygons),
                          key=lambda pair: order_index.get(pair[0], float('inf')))

    # Unzip the sorted pairs into two lists: sorted_labels and sorted_polygons
    sorted_labels, sorted_polygons = zip(*sorted_pairs)

    # Convert sorted_labels and sorted_polygons back to lists and return
    return list(sorted_polygons), list(sorted_labels)


def draw_image(polygons, labels, shape):
    # Create an empty numpy array
    arr = torch.zeros(shape, dtype=torch.uint8)

    polygons, labels = sort_polygons_and_labels(polygons, labels)

    # Iterate through each polygon
    for polygon, label in zip(polygons, labels):
        # Get the exterior coordinates
        exterior_coords = torch.tensor(polygon.exterior.coords, dtype=torch.int32)

        # Use skimage.draw.polygon to fill the polygon in the array
        rr, cc = sk_polygon(exterior_coords[:, 1], exterior_coords[:, 0], shape)
        arr[rr, cc] = EVAL_LABELS.index(label) + 1

    return arr.flatten()


def evaluate(target: str, prediction: str):
    pred_polygons, pred_labels = read_json(prediction)
    tar_polygons, tar_labels, width, height = read_xml(target)

    pred_tensor = draw_image(pred_polygons, pred_labels, shape=(width, height)).flatten()
    tar_tensor = draw_image(tar_polygons, tar_labels, shape=(width, height)).flatten()

    confusion_metric = MulticlassConfusionMatrix(num_classes=len(EVAL_LABELS) + 1).to(
        pred_tensor.device)
    batch_class_f1 = multi_class_f1(pred_tensor, tar_tensor, confusion_metric)

    return batch_class_f1.numpy(), len(tar_tensor)


def main():
    args = get_args()

    pred_dir = adjust_path(args.prediction_dir)
    target_dir = adjust_path(args.ground_truth_dir)

    targets = list(glob.glob(f"{target_dir}/*.xml"))
    predictions = [f"{pred_dir}/{basename(x)[:-4]}.json" for x in targets]

    class_f1_list = []
    class_f1_weights = []
    for target, prediction in tqdm(zip(targets, predictions), total=len(targets),
                                   desc="Evaluating"):
        f1_values, size = evaluate(target, prediction)
        class_f1_list.append(np.nan_to_num(f1_values, nan=0))
        class_f1_weights.append((1 - np.isnan(f1_values).astype(np.int8)) * size)


    batch_class_f1 = np.average(np.array(class_f1_list), axis=0, weights=np.array(class_f1_weights))

    print(args.prediction_dir)
    print(args.ground_truth_dir)
    print("f1 scores:")
    for idx, label in enumerate(["background"] + EVAL_LABELS):
        print(f"{label}: {batch_class_f1[idx]}")


# pylint: disable=duplicate-code
def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument(
        "--prediction_dir",
        "-p",
        type=str,
        default=None,
        help="path for folder with prediction xml files."
    )

    parser.add_argument(
        "--ground_truth_dir",
        "-g",
        type=str,
        default=None,
        help="path for folder with ground truth xml files."
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
