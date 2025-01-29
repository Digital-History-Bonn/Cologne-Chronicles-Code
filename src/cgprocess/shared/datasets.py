"""Module for shared Dataset classes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union, List, Optional, Any

import numpy as np
import torch
# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch import randperm
from torch.utils.data import Dataset

from src.cgprocess.layout_segmentation.datasets.train_dataset import IMAGE_PATH
from src.cgprocess.shared.utils import initialize_random_split, get_file_stems, prepare_file_loading


class PageDataset(Dataset):
    """
    Dataset to handle page based data split.
    """

    def __init__(
            self,
            image_path: str = IMAGE_PATH,
            dataset: str = "transkribus",
            file_stems: Union[List[str], None] = None
    ) -> None:

        if file_stems:
            self.file_stems = file_stems
        else:
            extension, _ = prepare_file_loading(dataset)
            self.file_stems = get_file_stems(extension, Path(image_path))

    def __len__(self) -> int:
        """
        Returns:
            int: number of items in dateset
        """
        return len(self.file_stems)

    def __getitem__(self, item: int) -> str:
        """
        returns one file stem
        """

        return self.file_stems[item]

    def random_split(
            self, ratio: Tuple[float, float, float]
    ) -> Tuple[PageDataset, PageDataset, PageDataset]:
        """
        splits the dataset in parts of size given in ratio

        Args:
            ratio(list): Ratio for train, val and test dataset

        Returns:
            tuple: Train, val and test PageDatasets
        """
        indices, splits = initialize_random_split(len(self), ratio)

        train_dataset = PageDataset(
            image_path="",
            dataset="",
            file_stems=np.array(self.file_stems)[indices[: splits[0]]].tolist()
        )
        valid_dataset = PageDataset(
            image_path="",
            dataset="",
            file_stems=np.array(self.file_stems)[indices[splits[0]: splits[1]]].tolist(),
        )
        test_dataset = PageDataset(
            image_path="",
            dataset="",
            file_stems=np.array(self.file_stems)[indices[splits[1]:]].tolist(),
        )

        return train_dataset, valid_dataset, test_dataset


class TrainDataset(Dataset, ABC):
    """
    Abstract Dataset Class for training. This Class can be supplied with a list of file stems.
    Otherwise, files to load will be determined based on the data path.
    Dataset Splitting on page level needs to be done beforehand.
    """

    def __init__(
            self,
            data_path: Path,
            limit: Optional[int] = None,
            data_source: str = "transkribus",
            file_stems: Optional[List[str]] = None,
            sort: bool = False,
            name: str = "default"
    ) -> None:
        """
        Args:
            data_path(Path): uses this list instead of loading data from disc
            limit(int): limits the quantity of loaded pages
            image_type(str): Name of image source, this influences the loading process.
            file_stems(list): File stems for images and targets
            sort(bool): sort file_names for testing purposes
            name(str): name of the dataset. E.g. train, val, test
        """
        self.image_path = data_path / "images"
        self.target_path = data_path / "targets"
        self.data_source = data_source
        self.limit = limit
        self.name = name

        if file_stems:
            self.file_stems = file_stems
        else:
            extension, get_file_name = prepare_file_loading(data_source)
            self.file_stems = get_file_stems(extension, self.image_path)
        if sort:
            self.file_stems.sort()

        self.image_extension = extension

        if limit is not None:
            assert limit <= len(
                self.file_stems), (f"Provided limit with size {limit} is greater than dataset size"
                                   f"with size {len(self.file_stems)}.")
            self.file_stems = self.file_stems[:limit]


    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            int: number of items in dateset
        """
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        """
        Returns:
            tuple: Input and target tensors
        """
        pass
