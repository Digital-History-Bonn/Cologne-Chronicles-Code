import glob
import json
import os
from os.path import basename
from typing import List, Tuple

from PIL import Image
from bs4 import BeautifulSoup
from shapely.geometry import Polygon
from skimage.io import imread, imsave
from tqdm import tqdm

ALPHABET = ['<ctc>',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ſ', 'ß', 'à', 'á', 'è', 'é', 'ò', 'ó', 'ù', 'ú',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            ',', '.', '?', '!', '-', '—', '_', ':', ';', '/', '\\', '(', ')', '[', ']', '{', '}',
            '%', '$', '£', '§', '\"', '„', '“', '»', '«', '\'', '’', '&', '+', '~', '*', '=', '†']

MAPPING = {x:x for x in ALPHABET}
MAPPING[' '] = '<space>'

def read_xml(path: str) -> Tuple[List[List[str]], List[List[Tuple[float, float, float, float]]]]:
    """
    Reads out polygon information and classes from xml file.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()

    soup = BeautifulSoup(data, "xml")

    page_text = []
    page_segments = []

    # Loop through each relevant region tag
    for region in soup.find_all('TextRegion'):
        # Extract the points for the polygon
        region_coords = region.find("Coords")
        custom = region.get("custom", "")
        if region_coords and 'structure' in custom:
            # get lines
            article_text = []
            line_segmenents = []
            for line in region.find_all("TextLine"):
                segment_coords = line.find("Coords")["points"]
                segment_coords = [(int(x), int(y)) for x, y in (pair.split(",") for pair in
                                                                segment_coords.split())]
                text = line.find("TextEquiv").find('Unicode').text

                if len(segment_coords) > 2:
                    article_text.append(text)
                    line_segmenents.append(Polygon(segment_coords).bounds)

            if len(article_text) > 0:
                page_text.append(article_text)
                page_segments.append(line_segmenents)

    return page_text, page_segments


def tokenize(text):
    return " ".join([MAPPING.get(x, '<unk>') for x in text])


def save_target(path: str, split: str, text: str, file_name: str):
    with open(f"{path}/{split}_text.txt", 'a', encoding='utf-8') as file:
        file.write(f"{split}/{file_name} {text}\n")

    with open(f"{path}/{split}.txt", 'a', encoding='utf-8') as file:
        file.write(f"{split}/{file_name} {tokenize(text)}\n")

    with open(f"{path}/{split}_ids.txt", 'a', encoding='utf-8') as file:
        file.write(f"{split}/{file_name}\n")


def save_crop(image, bbox, path):
    minx, miny, maxx, maxy = bbox
    crop = image[int(miny):int(maxy), int(minx):int(maxx)]
    image_pil = Image.fromarray(crop)

    aspect_ratio = image_pil.width / image_pil.height
    resized_crop = image_pil.resize((int(128 * aspect_ratio), 128), Image.Resampling.LANCZOS)

    resized_crop.save(path)


def create(annotation: str, image: str, output_path: str, split: str):
    os.makedirs(f"{output_path}/images", exist_ok=True)

    texts, segments = read_xml(annotation)
    image = imread(image)

    for i, (article_texts, line_segments) in enumerate(zip(texts, segments)):
        for j, (text, segment) in enumerate(zip(article_texts, line_segments)):
            file_name = f"{basename(annotation)[:-4]}_seg{i}_line{j}"

            # save crop
            os.makedirs(f"{output_path}/images/{split}/", exist_ok=True)
            save_crop(image, segment, f"{output_path}/images/{split}/{file_name}.jpg")

            # add line to files
            save_target(output_path, split, text, file_name)


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
        create(target, image, output_path, split)


    signs = ""
    for i, sign in enumerate(ALPHABET):
        signs += f"{sign} {i}\n"

    signs += f"<space> {i+1}\n"
    signs += f"<unk> {i+2}\n"

    with open(f"{output_path}/syms.txt", "w", encoding="utf-8") as file:
        file.write(signs)


if __name__ == '__main__':
    main(image_path="data/Chronicling-Germany-Dataset-main-data/data/images",
         xml_path="data/Chronicling-Germany-Dataset-main-data/data/annotations",
         output_path="data/PyLaia/dataset",
         split_file="data/Chronicling-Germany-Dataset-main-data/data/split.json")

