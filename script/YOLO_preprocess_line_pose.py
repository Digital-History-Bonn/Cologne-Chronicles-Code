import glob
import json
import os
from os.path import basename

import numpy as np
import yaml
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, LineString
from skimage.io import imread
from tqdm import tqdm

from script.YOLO_preprocess_line_segmentation import save_crop, save_target


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

    # Plot the image and overlay polygons
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()

    with open(target_path, "r", encoding="utf-8") as file:
        segment_data = file.readlines()

    # Parse the segment data
    for line in segment_data:
        parts = line.strip().split()
        coords = list(map(float, parts[1:]))
        points = np.array([(coords[i] * img_width, coords[i + 1] * img_height) for i in
                  range(0, len(coords), 2)])

        ax.plot(points[:, 0], points[:, 1])

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
            region_points = [(int(x), int(y)) for x, y in
                      (pair.split(",") for pair in coords['points'].split())]
            # Create a shapely Polygon from the points
            if len(region_points) > 2:
                # get lines
                article_lines = []
                for line in region.find_all("Baseline"):
                    coords = line.find("points")
                    points = [(int(x), int(y)) for x, y in
                              (pair.split(",") for pair in coords['points'].split())]

                    if len(points) > 2:
                        article_lines.append(LineString(points))

                if len(article_lines) > 0:
                    page_lines.append(article_lines)
                    polygons.append(Polygon(region_points))

    return polygons, page_lines


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
            save_target(article_lines, bbox,
                        f"{output_path}/labels/{basename(target)[:-4]}_{i}.txt")


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

    # create .yaml-file
    # Create the base dictionary structure
    dataset_config = {
        'path': output_path,
        'train': "train/images",
        'val': "val/images",
        'names': {0: 'Textline'}
    }

    # Write the data to a YAML file
    with open(f"{output_path}/CGD.yaml", 'w') as file:
        yaml.dump(dataset_config, file, default_flow_style=False)


if __name__ == '__main__':
    main(image_path="data/Chronicling-Germany-Dataset-main-data/data/images",
         xml_path="data/Chronicling-Germany-Dataset-main-data/data/annotations",
         output_path="data/YOLO_Baselines",
         split_file="data/Chronicling-Germany-Dataset-main-data/data/split.json")

    # plot_segments("data/YOLO_Textlines/train/labels/Koelnische_Zeitung_1866-06_1866-09_0073_7.txt",
    #               "data/YOLO_Textlines/train/images/Koelnische_Zeitung_1866-06_1866-09_0073_7.jpg")