"""Utility functions for baseline detection."""

import re
from typing import Tuple, Union

import numpy as np
import torch
from bs4 import PageElement


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


def get_tag(textregion: PageElement):
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
