"""Prediction script for Pero baseline detection."""
import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon
from skimage.io import imread

from bs4 import BeautifulSoup

from tqdm import tqdm
from ultralytics import YOLO

from src.cgprocess.baseline_detection.utils import adjust_path, polygon_to_string, \
    order_lines
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


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


def add_baselines(layout_xml: str,
                  output_file: str,
                  textlines: List[List[Polygon]]) -> None:
    """
    Adds testline and baseline prediction form model to the layout xml file.

    Args:
        layout_xml: path to layout xml file
        output_file: path to output xml file
        textlines: list of list of shapely LineString predicted by model
    """
    with open(layout_xml, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')

    # Find and remove all TextLines if exists
    page_elements = page.find_all('TextLine')
    for page_element in page_elements:
        page_element.decompose()

    textregions = page.find_all('TextRegion')

    # adds all predicted textlines to annotation
    for textregion, region_textlines in zip(textregions, textlines):
        for i, textline in enumerate(region_textlines):
            new_textline = soup.new_tag('TextLine')
            new_textline['custom'] = f"readingOrder {{index:{i};}}"
            new_textline['id'] = textregion['id'] + f'_tl_{i}'

            # add textline
            coords_element = soup.new_tag("Coords")
            points_list = np.array(textline.exterior.coords).ravel().tolist()
            coords_element["points"] = polygon_to_string(points_list)
            new_textline.append(coords_element)

            textregion.append(new_textline)

    for region in textregions:
        order_lines(region)

    # Write the modified XML back to file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(soup.prettify())


def predict(model: YOLO, image_path: str, layout_xml_path: str, output_file: str,
            devices: List[int]) -> None:
    crops, shifts = crop_layout(image_path, layout_xml_path)
    results = model.predict(crops, device=devices)

    # extract textlines form result
    textlines = []
    for result, shift in zip(results, shifts):
        textlines.append(
            [Polygon((mask.xy[0] + shift)) for mask in result.masks if mask.xy[0].shape[0] > 3])

    # add textlines to outputfile
    add_baselines(layout_xml_path, output_file, textlines)


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

    devices = list(range(num_gpus)) if torch.cuda.is_available() else 'cpu'
    model = YOLO("models/yolo_seg_best.pt", verbose=True)

    bar = tqdm(zip(image_paths, layout_xml_paths, output_paths),
               total=len(image_paths),
               desc="Predicting")
    for image_path, layout_path, output_path in bar:
        predict(model, image_path, layout_path, output_path, devices=devices)


if __name__ == '__main__':
    main()
