""" 
dataset and dataloader generation
"""

import os

import pandas as pd
import pyxis.torch as pxt
import torch
from PIL import Image
from torchvision.transforms import v2

from .imagenet import ImageNetDataset
from .xrd import (
    Normalize,
    PairedDataset,
    PeakHeightShiftTransform,
    RandomNoiseTransform,
)

norm_dict = {
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    "cifar100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "old-nfs": {"mean": [0.3843, 0.3843, 0.3843], "std": [0.1692, 0.1692, 0.1692]},
    "sem": {"mean": [0.4366, 0.4366, 0.4366], "std": [0.1684, 0.1684, 0.1684]},
    "paired": {"mean": [0.4366, 0.4366, 0.4366], "std": [0.1684, 0.1684, 0.1684]},
    "xrd": {"mean": [0], "std": [0]},
    "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "imagenet-o": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "openimage-o": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "svhn": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
    "teneo": {"mean": [0.3843, 0.3843, 0.3843], "std": [0.1692, 0.1692, 0.1692]},
    "mount": {"mean": [0.3843, 0.3843, 0.3843], "std": [0.1692, 0.1692, 0.1692]},
    "impurities": {"mean": [0.3843, 0.3843, 0.3843], "std": [0.1692, 0.1692, 0.1692]},
}

ROUTES = [
    "U3O8ADU",
    "U3O8AUC",
    "U3O8MDU",
    "U3O8SDU",
    "UO2ADU",
    "UO2AUCd",
    "UO2AUCi",
    "UO2SDU",
    "UO3ADU",
    "UO3AUC",
    "UO3MDU",
    "UO3SDU",
]


class TransformTorchDataset(pxt.TorchDataset):
    """
    reimplements pyxis's TorchDataset for LMDBs
    but allows for custom image transformation
    with `transform` arg
    """

    def __init__(self, dirpath, transform=None):
        super().__init__(dirpath)
        self.transform = transform

    def __getitem__(self, key):
        data = self.db[key]
        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        if self.transform:
            data["data"] = self.transform(data["data"].to(torch.float))

        return tuple(data.values())

    def __repr__(self):
        if self.transform is None:
            return str(self.db)
        else:
            format_string = "Transforms: "
            for t in self.transform.transforms:
                format_string += "\n"
                format_string += f"        {t}"
            format_string += "\n"
            return str(self.db) + "\n" + format_string


class ImageFolderDataset(torch.utils.data.Dataset):
    """
    Nuclear Forensics dataset.

    Args:
        split (string): Dataset split to load, one of "train", "val", or ood splits defined in dataset_config.yml
        root_dir (string): Root directory of built dataset
        transform (callable, optional): Optional transform to be applied to a sample
    """

    def __init__(self, split: str, root_dir: str, transform=None, fold_num: int = 1):
        self.dirpath = root_dir
        self.transform = transform

        # load dataset metadata file
        try:
            self.df = pd.read_csv(os.path.join(self.dirpath, "metadata.csv"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset {self.dirpath} does not exist, make sure it has been built.") from exc

        # filter metadata by fold number
        if split == "train":
            self.df = self.df[self.df["fold"] != fold_num]
        else:
            self.df = self.df[self.df["fold"] == fold_num]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        data = self.df.iloc[key].to_dict()
        data["image"] = Image.open(os.path.join(self.dirpath, data["filename"])).convert("RGB")

        if self.transform:
            data["image"] = self.transform(data["image"])

        return data["image"], data["label"]

    def __repr__(self):
        return "ImageFolderDataset"


class Convert:
    """convert image to RGB mode"""

    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(configs) -> dict:

    transforms = {}

    transforms["val"] = v2.Compose(
        [
            v2.Resize(configs.resize_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(configs.crop_size),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(
                mean=norm_dict[configs.dataset.lower()]["mean"],
                std=norm_dict[configs.dataset.lower()]["std"],
            ),
        ]
    )

    # create transformation list
    temp_transf = [
        v2.Resize(configs.resize_size, interpolation=v2.InterpolationMode.BILINEAR),
    ]

    # add transforms from configs
    for tf in configs.transforms:
        match tf:
            case "randomcrop":
                temp_transf.append(v2.RandomCrop(configs.crop_size))
            case "randomresizedcrop":
                temp_transf.append(v2.RandomResizedCrop(configs.crop_size, antialias=True))
            case "centercrop":
                temp_transf.append(v2.CenterCrop(configs.crop_size))
            case "randomhorizontalflip":
                temp_transf.append(v2.RandomHorizontalFlip())
            case "randomverticalflip":
                temp_transf.append(v2.RandomVerticalFlip())
            case "randomrotation":
                temp_transf.append(v2.RandomRotation(45))
            case "colorjitter":
                temp_transf.append(v2.ColorJitter(brightness=0.5))
            case "randomposterize":
                temp_transf.append(v2.RandomPosterize(4))
            case "gaussianblur":
                temp_transf.append(v2.GaussianBlur(3))
            case "magnification":
                pass

    # add final transforms
    temp_transf.extend(
        [
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(
                mean=norm_dict[configs.dataset.lower()]["mean"],
                std=norm_dict[configs.dataset.lower()]["std"],
            ),
        ]
    )
    transforms["train"] = v2.Compose(temp_transf)

    if configs.use_train_transf_for_val:
        transforms["val"] = transforms["train"]

    return transforms


def get_datasets(configs) -> dict:
    """
    load datasets

    """

    id_datasets = {
        "train": None,
        "val": None,
        "num_classes": None,
    }

    transforms = get_transforms(configs)

    match configs.dataset.lower():
        case "old-nfs":
            ### just nfs routes, no xrd. uo4 routes included.
            id_datasets["num_classes"] = 17

            id_datasets["train"] = ImageFolderDataset(
                split="train",
                fold_num=configs.fold_num,
                root_dir=configs.dataset_root,
                transform=transforms["train"],
            )
            id_datasets["val"] = ImageFolderDataset(
                split="val",
                fold_num=configs.fold_num,
                root_dir=configs.dataset_root,
                transform=transforms["val"],
            )

        case "sem":
            ### paired dataset with just SEM images
            id_datasets["num_classes"] = 13

            id_datasets["train"] = PairedDataset(
                root=configs.dataset_root,
                split="train",
                fold_num=configs.fold_num,
                sem_transform=transforms["train"],
                mode="sem",
            )
            id_datasets["val"] = PairedDataset(
                root=configs.dataset_root,
                split="val",
                fold_num=configs.fold_num,
                sem_transform=transforms["val"],
                mode="sem",
            )

        case "xrd":
            ### paired dataset with just XRD images
            id_datasets["num_classes"] = 3

            xrd_transform = v2.Compose(
                [
                    torch.from_numpy,
                    PeakHeightShiftTransform(shift_scale=0.15),
                    RandomNoiseTransform(noise_level=0.002),
                    Normalize(),
                ]
            )

            id_datasets["train"] = PairedDataset(
                root=configs.dataset_root,
                split="train",
                fold_num=configs.fold_num,
                xrd_transform=xrd_transform,
                mode="xrd",
            )
            id_datasets["val"] = PairedDataset(
                root=configs.dataset_root,
                split="val",
                fold_num=configs.fold_num,
                xrd_transform=torch.from_numpy,
                mode="xrd",
            )

        case "paired":
            ### paired dataset with both SEM and XRD images
            id_datasets["num_classes"] = 13

            xrd_transform = v2.Compose(
                [
                    torch.from_numpy,
                    PeakHeightShiftTransform(shift_scale=0.15),
                    RandomNoiseTransform(noise_level=0.002),
                    Normalize(),
                ]
            )

            id_datasets["train"] = PairedDataset(
                root=configs.dataset_root,
                split="train",
                sem_transform=transforms["train"],
                xrd_transform=xrd_transform,
                mode="paired",
            )
            id_datasets["val"] = PairedDataset(
                root=configs.dataset_root,
                split="val",
                sem_transform=transforms["val"],
                xrd_transform=torch.from_numpy,
                mode="paired",
            )

    return id_datasets
