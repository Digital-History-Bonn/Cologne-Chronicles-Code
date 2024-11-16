import glob
import json
import os
from os.path import basename
from typing import List, Tuple

import numpy as np
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib import patches
from shapely.geometry import Polygon
from skimage.io import imread, imsave
from tqdm import tqdm


def plot_segments(target_path, image_path):
    """
    Plots polygons (segments) on an image from a file with segment coordinates.

    Args:
        target_path (str): Path to the file containing segment data in the format:
                           <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
        image_path (str): Path to the image on which to draw the polygons.

    Returns:
        None: Displays the image with polygons.
    """
    # Load the image using skimage
    image = imread(image_path)
    img_height, img_width = image.shape[:2]

    with open(target_path, "r", encoding="utf-8") as file:
        segment_data = file.readlines()

    # Parse the segment data
    polygons = []
    for line in segment_data:
        parts = line.strip().split()
        class_id = int(parts[0])  # Extract the class index (if needed)
        coords = list(map(float, parts[1:]))
        # Normalize coordinates to pixel values
        points = [(coords[i] * img_width, coords[i + 1] * img_height) for i in
                  range(0, len(coords), 2)]
        polygons.append((class_id, points))

    # Plot the image and overlay polygons
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()

    # Draw each polygon
    for class_id, points in polygons:
        # Create a polygon patch
        polygon = patches.Polygon(points, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(polygon)

    plt.axis('off')
    plt.show()


def read_xml(path: str):
    """
    Reads out polygon information and classes from xml file.

    Args:
        path (str): path to xml file

    Returns:
        polygons (list): list of polygons
        classes (list): list of classes
        width (int): width of page
        height (int): height of page
    """
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()

    soup = BeautifulSoup(data, "xml")

    polygons = []
    page_lines = []

    # Loop through each relevant region tag
    for region in soup.find_all('TextRegion'):
        # Extract the points for the polygon
        coords = region.find("Coords")
        custom = region.get("custom", "")
        if coords and 'structure' in custom:
            # Convert the points string to a list of (x, y) tuples
            points = [(int(x), int(y)) for x, y in
                      (pair.split(",") for pair in coords['points'].split())]
            # Create a shapely Polygon from the points
            if len(points) > 2:
                polygons.append(Polygon(points))

                # get lines
                article_lines = []
                for line in region.find_all("TextLine"):
                    coords = line.find("Coords")
                    points = [(int(x), int(y)) for x, y in
                              (pair.split(",") for pair in coords['points'].split())]

                    if len(points) > 2:
                        article_lines.append(Polygon(points))

                page_lines.append(article_lines)

    return polygons, page_lines


def save_crop(image: np.ndarray, segment: Polygon, path: str) -> np.ndarray:
    minx, miny, maxx, maxy = segment.bounds
    crop = image[int(miny):int(maxy), int(minx):int(maxx)]
    imsave(path, crop)

    return np.array([minx, miny, maxx, maxy])


def save_target(article_lines: List[Polygon], bbox: Tuple[int, int, int, int], path: str):
    shift = np.array([bbox[0], bbox[1]])
    factor = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
    with open(path, "w", encoding="utf-8") as file:
        for line in article_lines:
            coords = (line.exterior.coords[:-1] - shift) / factor
            coord_str = " ".join(f"{x} {y}" for x, y in coords)
            file.write(f"1 {coord_str}\n")


def create(target, image, output_path):
    os.makedirs(f"{output_path}/images", exist_ok=True)
    os.makedirs(f"{output_path}/labels", exist_ok=True)

    segments, page_lines = read_xml(target)
    image = imread(image)

    for i, (segment, article_lines) in enumerate(zip(segments, page_lines)):
        if not os.path.exists(f"{output_path}/images/{basename(target)[:-4]}_{i}.jpg"):
            # create crop
            bbox = save_crop(image, segment,
                             f"{output_path}/images/{basename(target)[:-4]}_{i}.jpg")

            # create target .txt file
            save_target(article_lines, bbox, f"{output_path}/labels/{basename(target)[:-4]}_{i}.txt")


def main(image_path: str, xml_path: str, output_path: str, split_file: str):
    # get xml-files and images
    targets = glob.glob(f"{xml_path}/*.xml")
    images = [f"{image_path}/{basename(x)[:-4]}.jpg" for x in targets]

    with open(split_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    data['train'] = data.pop('Training')
    data['val'] = data.pop('Validation')
    data['test'] = data.pop('Test')

    split_dict = {value: key for key, values in data.items() for value in values}

    # create image_crops and target txt files
    for target, image in tqdm(zip(targets, images), desc="Preprocessing data", total=len(images)):
        split = split_dict[basename(target)[:-4]]
        create(target, image, f"{output_path}/{split}")


if __name__ == '__main__':
    main(image_path="data/Chronicling-Germany-Dataset-main-data/data/images",
         xml_path="data/Chronicling-Germany-Dataset-main-data/data/annotations",
         output_path="data/YOLO_Textlines",
         split_file="data/Chronicling-Germany-Dataset-main-data/data/split.json")
