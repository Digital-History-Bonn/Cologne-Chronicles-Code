"""Module to compare two transcript files. The transcriptions will be compare region by region and line by line."""
import argparse
import difflib
import json
import os
import statistics
from typing import List, Any, Tuple, Optional

import Levenshtein
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm  # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.OCR.utils import adjust_path
from src.OCR.utils import line_has_text


def main(parsed_args: argparse.Namespace) -> None:
    """Compare xml transcript files line by line."""

    confidence_threshold = parsed_args.confidence_threshold

    ground_truth_path = adjust_path(parsed_args.ground_truth_path)
    ocr_path = adjust_path(parsed_args.ocr_path)
    output_path = adjust_path(parsed_args.output_path)

    gt_paths = [
        f[:-4] for f in os.listdir(ground_truth_path) if f.endswith(".xml")
    ]
    ocr_paths = [
        f[:-4] for f in os.listdir(ocr_path) if f.endswith(".xml")
    ]

    if parsed_args.custom_split_file:
        with open(parsed_args.custom_split_file, "r", encoding="utf-8") as file:
            split = json.load(file)
            gt_paths = split["Test"]
    else:
        assert len(gt_paths) == len(ocr_paths), (
            f"Found {len(gt_paths)} ground truth files, but ther are {len(ocr_paths)} "
            f"ocr files. Make sure, for every ground truth file exists an ocr file")

    if output_path and not os.path.exists(output_path):
        print(f"creating {output_path}.")
        os.makedirs(output_path)

    multi_page_distance_list: list = []
    multi_page_bad: list = []
    multi_page_correct: list = []
    multi_page_bad_list: list = []
    bleu_sum = 0.0

    for path in tqdm(gt_paths):
        multi_page_bad_list, multi_page_distance_list, bleu_sum = compare_page(confidence_threshold,
                                                                     multi_page_bad,
                                                                     multi_page_bad_list,
                                                                     multi_page_correct,
                                                                     multi_page_distance_list,
                                                                     parsed_args, path, bleu_sum)

    print(f"overall levensthein distance per character: {calculate_ratio(multi_page_distance_list)}")
    print(f"overall correct lines: {calculate_ratio(multi_page_correct)}")
    print(f"overall bad lines: {calculate_ratio(multi_page_bad)}")
    print(f"overall bleu score: {bleu_sum}")

    with open(f"{output_path}multi_page_bad_list.json", "w", encoding="utf8") as file:
        json.dump(multi_page_bad_list, file)


def compare_page(confidence_threshold: float, multi_page_bad: list, multi_page_bad_list: list,
                 multi_page_correct: list,
                 multi_page_distance_list: list, parsed_args: argparse.Namespace,
                 path: str, bleu_sum: float) -> Tuple[list, list, float]:
    """
    Load lines from ground truth and ocr files and compare them directly. This function assumes, that the
    lines are already assigned from one file to the other, and text can be directly compare.
    """
    ground_truth, _, _, _ = read_lines(f"{parsed_args.ground_truth_path}{path}", "TextRegion",
                                       "TextLine")
    ocr, _, confidence_list, _ = read_lines(f"{parsed_args.ocr_path}{path}", "TextRegion",
                                            "TextLine",
                                            confidence=bool(confidence_threshold))
    result = difflib.HtmlDiff().make_file(
        merge_lists_conf(ocr, confidence_list, confidence_threshold),
        merge_lists_conf(ground_truth, confidence_list, confidence_threshold))
    with open(f"{parsed_args.output_path}{path}.html", "w", encoding="utf8") as file:
        file.write(result)
    lev_dis, lev_med, ratio_list, distance_list, text_list, bleu_score = evaluate_lines(ground_truth, ocr,
                                                                            confidence_list,
                                                                            confidence_threshold)

    char_ratio = calculate_ratio(distance_list)
    print(f"{path} correct lines: {len(np.array(ratio_list)[np.array(ratio_list) == 1.0]) / len(ratio_list)}")
    print(f"{path} bad lines: {len(np.array(ratio_list)[np.array(ratio_list) < 0.9]) / len(ratio_list)}")
    print(f"{path} normalized levensthein distance per line: {lev_dis}")
    print(f"{path} normalized levensthein distance per character: {char_ratio}")
    print(f"{path} levensthein median: {lev_med}")
    print(f"{path} levensthein worst line: {min(ratio_list)}\n")
    print(f"{path} bleu score: {bleu_score}\n")

    multi_page_distance_list += distance_list
    multi_page_correct.append((len(np.array(ratio_list)[np.array(ratio_list) == 1.0]), len(ratio_list)))
    multi_page_bad.append((len(np.array(ratio_list)[np.array(ratio_list) < 0.9]), len(ratio_list)))
    multi_page_bad_list += np.array(text_list)[np.array(ratio_list) < 0.9].tolist()
    bleu_sum += bleu_score

    return multi_page_bad_list, multi_page_distance_list, bleu_sum


def merge_lists_conf(list_list: List[List[Any]], conf_list: List[List[float]],
                     confidence_threshold: float) -> List[
    Any]:
    """merges a list, which contains lists. This allows for confidence filtering."""
    result = []
    for element_list, conf_region in zip(list_list, conf_list):
        for element, conf in zip(element_list, conf_region):
            if sufficient_confidence(conf, confidence_threshold):
                result.append(element)
    return result


def sufficient_confidence(conf, confidence_threshold):
    """returns True if, the confidence is sufficient or """
    if not confidence_threshold:
        return True
    return conf >= confidence_threshold


def calculate_ratio(data_list: List[Tuple[int, int]]) -> float:
    """Calculate ratio from list containing values and lengths"""
    data_ndarray = np.array(data_list)
    sums = np.sum(data_ndarray, axis=0)
    ratio: float = sums[0] / sums[1]
    return ratio


def evaluate_lines(gt: List[List[str]], ocr: List[List[str]],
                   confidence_list: List[List[float]],
                   confidence_threshold: Optional[float] = None) -> Tuple[
    float, float, List[float], List[Tuple[int, int]], List[Tuple[str, str]], float]:
    """
    Computes the levensthein ratio between all lines and returns the mean, median and list of all ratios.
    """
    chencherry = SmoothingFunction()
    ratio_sum = 0.0
    count = 0
    ratio_list = []
    text_list = []
    distance_list = []
    bleu_sum = 0
    for gt_region, ocr_region, conf_region in zip(gt, ocr, confidence_list):
        count += len(gt_region)
        for gt_line, ocr_line, conf in zip(gt_region, ocr_region, conf_region):
            if sufficient_confidence(conf, confidence_threshold):
                bleu_sum += sentence_bleu([ocr_line.split(" ")], gt_line.split(" "), smoothing_function=chencherry.method1)
                ratio = Levenshtein.ratio(gt_line, ocr_line)
                distance = Levenshtein.distance(gt_line, ocr_line)
                distance_list.append((distance, len(gt_line) + len(ocr_line)))
                text_list.append((gt_line, ocr_line))
                ratio_sum += ratio
                ratio_list.append(ratio)

    return 1 - ratio_sum / count, 1 - statistics.median(
        ratio_list), ratio_list, distance_list, text_list, bleu_sum/count


def read_lines(
        path: str,
        tag: str,
        child_tag: str,
        confidence: bool = False
) -> Tuple[List[List[str]], List[List[List[Tuple]]], List[List[float]], List[BeautifulSoup]]:
    """
    Reads xml file and extracts all the lines from specified regions.
    :param confidence_threshold: lines below this threshold are ignored
    :param path: path to xml file without xml ending
    :param tag: tag to be found
    :param child_tag: children tag to be found
    :return: Region lists, containing a List of text, line coordinates and confidence for each region seperatly, as
    well as the complete beautiful soup object for all lines
    """
    with open(f"{path}.xml", "r", encoding="utf-8") as file:
        data = file.read()

    bs_data = BeautifulSoup(data, "xml")

    regions = bs_data.find_all(tag)

    region_text_list = []
    region_coords_list = []
    region_conf_list = []
    lines_data = []

    for region in regions:
        lines = region.find_all(child_tag)
        text_list = []
        coords_list = []
        conf_list = []
        for line in lines:
            if line_has_text(line):
                text_list.append(line.TextEquiv.Unicode.contents[0])
                coords_list.append(
                    [tuple(pair.split(",")) for pair in line.Coords["points"].split()])
                conf_list.append(line_confidence(confidence, line))
                lines_data.append(line)
        region_text_list.append(text_list)
        region_coords_list.append(coords_list)
        region_conf_list.append(conf_list)
    return region_text_list, region_coords_list, region_conf_list, lines_data


def line_confidence(confidence, line) -> float:
    """Returns True if threshold ist not supplied or no confidence for the line is known. If both are present it only
    returns true, if the confidence is grater or equal to the threshold."""
    if not confidence:
        return 0.0
    if "conf" in line.TextEquiv.attrs:
        return float(line.TextEquiv.attrs["conf"])
    return 0.0


# pylint: disable=duplicate-code
def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--ground-truth-path",
        "-gt",
        type=str,
        default="data/ground_truth/",
        help="path for folder with groundtruth xml files. Names have to match exactly to the ocr files",
    )
    parser.add_argument(
        "--ocr-path",
        "-oc",
        type=str,
        default="data/ocr/",
        help="path for folder with ocr prediction xml files. Names have to match exactly to the ground truth files"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default="data/output/",
        help="path for output folder. This will contain html comparison files for every page.",
    )
    parser.add_argument(
        "--confidence-threshold",
        "-ct",
        type=float,
        dest="confidence_threshold",
        default=None,
        help="Confidence Threshold. Only lines above the threshold are compared. If this is None, or no confidence is "
             "supplied in the ocr xml files, no threshold will be applied.",
    )
    parser.add_argument(
        "--custom-split-file",
        type=str,
        default=None,
        help="Provide path for custom split json file. This should contain a list with file stems "
             "of train, validation and test images. This will only evaluate the test dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    main(parameter_args)
