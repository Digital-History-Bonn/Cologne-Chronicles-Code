"""
module for Dataset class
"""
from __future__ import annotations

from typing import Tuple, List
import os
from PIL import Image  # type: ignore
import numpy as np
import torch  # type: ignore
from torch import randperm # type: ignore
from torch.utils.data import Dataset  # type: ignore
from torchvision import transforms  # type: ignore


PATH = 'crops/'


class NewsDataset(Dataset):
    """
        A dataset class for the newspaper datasets
    """

    def __init__(self, path: str = PATH, files: List[str] = None, limit: int = None):
        """
        Dataset object
        load images and targets from folder

        :param path: path to folders with images and targets
        """
        # path to the over all folder
        self.path = path

        # list paths of images and targets
        if files is None:
            self.file_names = [f[:-3] for f in os.listdir(f"{path}") if f.endswith(".pt")]
        else:
            self.file_names = files

        if limit is not None:
            self.file_names = self.file_names[:limit]

        self.augmentations = True

    def __len__(self):
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.file_names)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """
        # load data
        data = torch.load(f"{self.path}{self.file_names[item]}.pt")

        # do augmentations
        if self.augmentations:
            augmentations = self.get_augmentations()
            data = augmentations(data)

        return data[:-1].float(), data[-1].long()

    def random_split(self, ratio: Tuple[float, float, float]) \
            -> Tuple[NewsDataset, NewsDataset, NewsDataset]:
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        :return (list): list of NewsDatasets
        """
        assert sum(ratio) == 1, "ratio does not sum up to 1."
        assert len(ratio) == 3, "ratio does not have length 3"

        splits = int(ratio[0] * len(self)), int(ratio[0] * len(self)) + int(ratio[1] * len(self))

        indices = randperm(len(self), generator=torch.Generator().manual_seed(42)).tolist()
        nd_paths = np.array(self.file_names)

        train_dataset = NewsDataset(path=self.path, files=list(nd_paths[indices[:splits[0]]]))
        test_dataset = NewsDataset(path=self.path, files=list(nd_paths[indices[splits[0]:splits[1]]]))
        valid_dataset = NewsDataset(path=self.path, files=list(nd_paths[indices[splits[1]:]]))

        return train_dataset, test_dataset, valid_dataset

    @staticmethod
    def get_augmentations() -> transforms.Compose:
        """Defines transformations"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180)
        ])


if __name__ == '__main__':
    dataset = NewsDataset()
    print(f"{len(dataset)=}")

    train, valid, test = dataset.random_split(ratio=(.9, .05, .05))
    print(f"{len(train)=}")
    print(f"{len(valid)=}")
    print(f"{len(test)=}")

    img, tar = train[0]
    print(f"{img.shape=}, {type(img)=}")
    print(f"{tar.shape=}, {type(tar)=}")
