"""Module for SSM OCR prediction."""
import argparse
import glob
import os
from multiprocessing import set_start_method, Queue
from pathlib import Path
from typing import List, Optional

import torch

from src.cgprocess.shared.multiprocessing_handler import MPPredictor, get_cpu_count


class OCRPreprocess :
    """Preprocessing includes cropping and masking individual lines for OCR."""
    def __init__(self,
            kwargs: dict,
            image_height: int,
            cfg: dict,
            path_queue: Queue,
            num_processes: Optional[int] = None
            ):
        """
        Args:
            image_path: path to folder with images
            crop_height: Fixed height. All crops will be scaled to this height
            cfg: configuration file tied to the model
        """
        super().__init__(**kwargs)
        self.image_height = image_height
        self.num_processes = num_processes if num_processes else get_cpu_count() // 8
        self.cfg = cfg

    def extract_data(self, image_path: Path, annotations_path: Path, output_path: Path) -> None:
        """Load xml ."""

        file_stems = [
            f[:-4] for f in os.listdir(annotations_path) if f.endswith(".xml")
        ]

        target_stems = [
            f[:-4] for f in os.listdir(output_path) if f.endswith(".npz")
        ]
        path_queue: Queue = Queue()
        total = len(file_stems)

        print(f"num processes: {self.num_processes}")
        processes = [Process(target=extract_page,
                             args=(path_queue, (self.image_path, self.annotations_path, self.target_path),
                                   self.image_extension,
                                   self.cfg)) for _ in range(self.num_processes)]

        for file_stem in tqdm(file_stems, desc="Put paths in queue"):
            if file_stem in target_stems:
                continue
            path_queue.put((file_stem, False))

        run_processes({"method": get_progress, "args": self.target_path}, processes, path_queue, total,
                      "Page converting")

def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Predict SSM OCR")
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="default-ocr-prediction",
        help="Name of the model and the log files."
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        help="path for folder with folders 'images' and 'annotations'",
    )
    parser.add_argument(
        "--layout_dir",
        "-l",
        type=str,
        help="path for folder with layout xml files."
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="Number of processes that are used for preprocessing.",
    )
    parser.add_argument(
        "--thread-count",
        "-t",
        type=int,
        default=1,
        help="Select number of threads that are launched per process.",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="model.pt",
        help="path to model .pt file",
    )

    return parser.parse_args()


def main() -> None:
    """Predicts OCR for all images with xml annotations in given folder."""
    args = get_args()
    data_path = Path(args.data_path)
    layout_path = Path(args.layout_dir)

    # get file names
    image_paths = list(glob.glob(f'{data_path}/*.jpg'))
    layout_paths = [f'{layout_path}/{os.path.basename(path)[:-4]}.xml' for path in image_paths]

    assert len(image_paths) == len(layout_paths), "Images and annotations path numbers do not match."

    num_gpus = torch.cuda.device_count()

    # put paths in queue
    path_queue = create_path_queue(layout_paths, image_paths)

    model_list = create_device_list(num_gpus)

    predictor = MPPredictor("OCR prediction", predict, init_model, path_queue, model_list, str(data_path), True, True)
    predictor.launch_processes(num_gpus, args.thread_count)


def create_path_queue(annotations: List[str], images: List[str]) -> Queue:
    """
    Create path queue for OCR prediction containing image and layout path.
    Elements are required to have the image path at
    index 0 and the bool variable for terminating processes at index -1.
    :param annotations: list of annotation paths
    :param images: list of image paths
    """
    path_queue: Queue = Queue()
    for image_path, annotation_path in zip(images, annotations):
        path_queue.put((image_path,
                        annotation_path,
                        False))
    return path_queue


def create_device_list(num_gpus: int) -> list:
    """
    Create OCR model list containing one separate model for each process.
    """
    device_list = [[f"cuda:{i}"] for i in
                  range(num_gpus)] if (
            torch.cuda.is_available() and num_gpus > 0) else \
        [["cpu"]]
    return device_list


def init_model(device: object):
    """Init function for compatibility with the MPPredictor handling baseline and layout predictions as well."""
    torch.load

if __name__ == '__main__':
    set_start_method('spawn')
    main()