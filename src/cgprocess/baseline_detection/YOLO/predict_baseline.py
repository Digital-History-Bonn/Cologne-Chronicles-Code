"""Prediction script for Pero baseline detection."""
import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon, LineString
from skimage.io import imread

from bs4 import BeautifulSoup
from torchvision.ops import box_area
from torchvision.ops._utils import _upcast

from tqdm import tqdm
from ultralytics import YOLO

from src.cgprocess.baseline_detection.utils import adjust_path, add_baselines
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


def box_inter(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="predict")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=None,
        help="path for folder with images. Images need to be jpg."
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--layout_dir",
        "-l",
        type=str,
        default=None,
        help="path for folder with layout xml files."
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="path to the folder where to save the preprocessed files",
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="path to the model file",
    )

    return parser.parse_args()


def extract_layout(xml_path: str) -> List[List[int]]:
    """
    Extracts the textregions and regions to mask form xml file.

    Args:
        xml_path: path to xml file

    Returns:
        mask regions and textregions
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    rois = []

    text_regions = page.find_all(['TextRegion'])
    for region in text_regions:
        points = np.array(xml_polygon_to_polygon_list(region.Coords["points"]))
        rois.append([int(x) for x in Polygon(points).bounds])

    return rois


def crop_layout(image_path, layout_xml_path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rois = extract_layout(layout_xml_path)
    image = imread(image_path)

    return ([image[y_min: y_max, x_min:x_max] for x_min, y_min, x_max, y_max in rois],
            [np.array([x_min, y_min]) for x_min, y_min, _, _ in rois])


def suppression(bboxs, lines, threshold):
    bboxs = bboxs.xyxy
    lines = lines.xy
    
    if len(bboxs) == 0:
        return bboxs, lines

    intersections = box_inter(bboxs, bboxs)
    intersections = intersections.fill_diagonal_(0).amax(dim=0)
    area = box_area(bboxs)

    mask = (intersections / area) < threshold
    return bboxs[mask], lines[mask]


def predict(model: YOLO, image_path: str, layout_xml_path: str, output_file: str,
            device: str) -> None:
    crops, shifts = crop_layout(image_path, layout_xml_path)
    results = model.predict(crops, device=device, verbose=False)

    # extract textlines form result
    page_textlines = []
    page_baselines = []
    for result, shift in zip(results, shifts):
        result = result.cpu()
        textlines = []
        baselines = []
        bboxs, lines = suppression(result.boxes, result.keypoints, threshold=.9)
        for bbox, line in zip(bboxs.numpy(), lines.numpy()):
            bbox = bbox + np.tile(shift, 2)
            textlines.append(Polygon([[bbox[0], bbox[1]],
                                      [bbox[2], bbox[1]],
                                      [bbox[2], bbox[3]],
                                      [bbox[0], bbox[3]]]))
            baselines.append(LineString(line + shift))

        page_textlines.append(textlines)
        page_baselines.append(baselines)

    # add textlines to outputfile
    add_baselines(layout_xml_path, output_file, page_textlines, page_baselines)


def main() -> None:
    """Predicts textlines and baselines for all files in given folder."""
    args = get_args()

    input_dir = adjust_path(args.input_dir)
    layout_dir = adjust_path(args.layout_dir)
    output_dir = adjust_path(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list(glob.glob(f"{input_dir}/*.jpg"))
    layout_xml_paths = [f"{layout_dir}/{os.path.basename(i)[:-4]}.xml" for i in image_paths]
    output_paths = [f"{output_dir}/{os.path.basename(i)[:-4]}.xml" for i in image_paths]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print(f"Using {num_gpus} gpu device(s).")
    else:
        print("Using cpu.")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(f"models/{args.model}.pt", verbose=True)

    bar = tqdm(zip(image_paths, layout_xml_paths, output_paths),
               total=len(image_paths),
               desc="Predicting")
    for image_path, layout_path, output_path in bar:
        predict(model, image_path, layout_path, output_path, device=device)


if __name__ == '__main__':
    main()
