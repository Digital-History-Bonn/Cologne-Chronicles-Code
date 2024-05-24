"""Module for creating Transkribus PAGE XML data."""
import argparse
import os
from typing import Dict, List

from bs4 import BeautifulSoup

from src.news_seg.class_config import LABEL_NAMES


def export_xml(args: argparse.Namespace, file: str, reading_order_dict: Dict[int, int],
               segmentations: Dict[int, List[List[float]]]) -> None:
    """
    Open pre created transkribus xml files and save polygon xml data.
    :param args: args
    :param file: xml path
    :param reading_order_dict: reading order value for each index
    :param segmentations: polygon dictionary sorted by labels
    """
    with open(
            f"{args.data_path}page/{os.path.splitext(file)[0]}.xml",
            "r",
            encoding="utf-8",
    ) as xml_file:
        xml_data = create_xml(xml_file.read(), segmentations, reading_order_dict, args.scale)
    with open(
            f"{args.data_path}page/{os.path.splitext(file)[0]}.xml",
            "w",
            encoding="utf-8",
    ) as xml_file:
        xml_file.write(xml_data.prettify())


def create_xml(
        xml_file: str, segmentations: Dict[int, List[List[float]]], reading_order: Dict[int, int], scale: float
) -> BeautifulSoup:
    """
    Creates a soup object containing Page Tag and Regions
    :param xml_file: xml file, to which the page data will be written
    :param segmentations: dictionary assigning labels to polygon lists
    :param file_name: image file name
    :param size: image size
    """
    xml_data = BeautifulSoup(xml_file, "xml")
    page = xml_data.find("Page")
    page.clear()
    order = xml_data.new_tag("ReadingOrder")
    order_group = xml_data.new_tag(
        "OrderedGroup", attrs={"caption": "Regions reading order"}
    )

    add_regions_to_xml(order_group, page, reading_order, segmentations, xml_data, scale)
    order.append(order_group)
    page.insert(0, order)
    return xml_data


def add_regions_to_xml(order_group: BeautifulSoup, page: BeautifulSoup, reading_order: Dict[int, int],
                       segmentations: Dict[int, List[List[float]]], xml_data: BeautifulSoup, scale: float) -> None:
    """
    Add ReadingOrder XML and Text Region List to Page
    :param order_group: BeautifulSOup Object for ReadingOrder
    :param page: Page BeautifulSOup Object
    :param reading_order: dict
    :param segmentations: dictionary assigning labels to polygon lists
    :param xml_data: final BeautifulSOup object
    """
    index = 0
    for label, segmentation in segmentations.items():
        for polygon in segmentation:
            order_group.append(
                xml_data.new_tag(
                    "RegionRefIndexed",
                    attrs={"index": str(reading_order[index]), "regionRef": str(index)},
                )
            )
            # TODO: add other region types
            region = xml_data.new_tag(
                "TextRegion",
                attrs={
                    "id": str(index),
                    "custom": f"readingOrder {{index:{reading_order[index]};}} structure "
                              f"{{type:{get_label_name(label)};}}",
                },
            )
            region.append(
                xml_data.new_tag("Coords", attrs={"points": polygon_to_string(polygon, scale)})
            )
            page.append(region)
            index += 1


def get_label_name(label: int) -> str:
    """
    Get label name from LABEL_NAMES list
    :param label: int label value
    :return: label name
    """
    return LABEL_NAMES[label - 1]


def polygon_to_string(input_list: List[float], scale: float) -> str:
    """
    Converts a list to string, while converting each element in the list to an integer. X and y coordinates are
    separated by a comma, each pair is separated from other coordinate pairs by a space. This format is required
    for transkribus
    :param input_list: list withcoordinates
    :return: string
    """
    generator_expression = (
        f"{int(input_list[index] * scale**-1)},{int(input_list[index + 1] * scale**-1)}"
        for index in range(0, len(input_list), 2)
    )
    string = " ".join(generator_expression)

    return string