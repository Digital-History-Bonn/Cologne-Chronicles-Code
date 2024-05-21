"""Utility functions for baseline detection."""
import re
from typing import Tuple, Union, List, Dict

import numpy as np
import torch
from bs4 import PageElement, BeautifulSoup
from scipy import ndimage


def get_bbox(points: Union[np.ndarray, torch.Tensor],  # type: ignore
             ) -> Tuple[int, int, int, int]:
    """
    Creates a bounding box around all given points.

    Args:
        points: np.ndarray of shape (N x 2) containing a list of points

    Returns:
        coordinates of bounding box in the format (x_min, y_min, x_max, y_max)
    """
    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()
    return x_min, y_min, x_max, y_max  # type: ignore


def is_valid(box: torch.Tensor) -> bool:
    """
    Checks if given bounding box has a valid size.

    Args:
        box: bounding box (xmin, ymin, xmax, ymax)

    Returns:
        True if bounding box is valid
    """
    if box[2] - box[0] <= 0:
        return False
    if box[3] - box[1] <= 0:
        return False
    return True


def convert_coord(element: PageElement) -> np.ndarray:
    """
    Converts PageElement with Coords in to a numpy array.

    Args:
        element: PageElement with Coords for example a Textregion

    Returns:
        np.ndarray of shape (N x 2) containing a list of coordinates
    """
    coords = element.find('Coords')
    return np.array([tuple(map(int, point.split(','))) for
                     point in coords['points'].split()])[:, np.array([1, 0])]


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
    return match.group()[6:-2]


def get_reading_order_idx(textregion: PageElement) -> int:
    """
    Extracts reading order from textregion PageElement.

    Args:
        textregion: PageElement of Textregion

    Returns:
         Reading Order Index as int
    """
    desc = textregion['custom']
    match = re.search(r"readingOrder\s*\{index:(\d+);\}", desc)
    if match is None:
        return -1
    return int(match.group(1))


def extract(xml_path: str
            ) -> Tuple[List[Dict[str, Union[torch.Tensor, List[torch.Tensor], int]]],
                       List[torch.Tensor]]:
    """
    Extracts the annotation from the xml file.

    Args:
        xml_path: path to the xml file.

    Returns:
        A list of dictionary representing all Textregions in the given document
        A list of polygons as torch tensors for masking areas
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    paragraphs = []
    mask_regions = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        tag = get_tag(region)
        coords = region.find('Coords')
        region_polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])[:, torch.tensor([1, 0])]

        if tag in ['table', 'header']:
            if is_valid(torch.tensor(get_bbox(region_polygon))):
                mask_regions.append(region_polygon)

        if tag in ['heading', 'article_', 'caption', 'paragraph']:
            region_bbox = torch.tensor(get_bbox(region_polygon))

            if is_valid(region_bbox):
                region_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor], int]] = {
                    'region_bbox': region_bbox,
                    'bboxes': [],
                    'textline_polygone': [],
                    'baselines': [],
                    'readingOrder': get_reading_order_idx(region)}

                text_region = region.find_all('TextLine')
                for text_line in text_region:
                    polygon = text_line.find('Coords')
                    baseline = text_line.find('Baseline')
                    if baseline:
                        # get and shift baseline
                        line = torch.tensor([tuple(map(int, point.split(','))) for
                                             point in baseline['points'].split()])
                        line = line[:, torch.tensor([1, 0])]

                        line -= region_bbox[:2].unsqueeze(0)

                        region_dict['baselines'].append(line)  # type: ignore

                        # get mask
                        polygon_pt = torch.tensor([tuple(map(int, point.split(','))) for
                                                   point in polygon['points'].split()])
                        polygon_pt = polygon_pt[:, torch.tensor([1, 0])]

                        # move mask to be in subimage
                        polygon_pt -= region_bbox[:2].unsqueeze(0)

                        # calc bbox for line
                        box = torch.tensor(get_bbox(polygon_pt))[torch.tensor([1, 0, 3, 2])]
                        box = box.clip(min=0)

                        # add bbox to data
                        if is_valid(box):
                            region_dict['bboxes'].append(box)  # type: ignore

                            # add mask to data
                            region_dict['textline_polygone'].append(polygon_pt)  # type: ignore

                # only adding regions with at least one textline
                if (region_dict['textline_polygone']) > 0:
                    paragraphs.append(region_dict)

    return paragraphs, mask_regions


def nonmaxima_suppression(input_array: np.ndarray, element_size: Tuple[int, int] = (7, 1)) -> np.ndarray:
    """
    From https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py.

    Vertical non-maxima suppression.

    Args:
        input_array: input array
        element_size: structure element for greyscale dilations

    Returns:
        non maxima suppression of baseline input image
    """
    if len(input_array.shape) == 3:
        dilated = np.zeros_like(input_array)
        for i in range(input_array.shape[0]):
            dilated[i, :, :] = ndimage.grey_dilation(
                input_array[i, :, :], size=element_size)
    else:
        dilated = ndimage.grey_dilation(input_array, size=element_size)

    return input_array * (input_array == dilated)