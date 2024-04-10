""" 
dataset and dataloader generation
"""

import os

import pandas as pd
import pyxis.torch as pxt
import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2

from .imagenet import ImageNetDataset

norm_dict = {
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    "cifar100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "nfs": {"mean": [0.3843, 0.3843, 0.3843], "std": [0.1692, 0.1692, 0.1692]},
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

    def __init__(self, split: str, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dirpath = os.path.join(self.root_dir, split)

        # load dataset metadata file
        try:
            self.df = pd.read_csv(os.path.join(self.dirpath, "metadata.csv"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset {self.dirpath} does not exist, make sure it has been built.") from exc

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


class RandomNoiseTransform:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, data):
        noise = torch.randn_like(data) * self.noise_level
        return data + noise


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
        case "nfs":
            id_datasets["num_classes"] = 14

            id_datasets["train"] = ImageFolderDataset(
                split="train",
                root_dir=configs.dataset_root,
                transform=transforms["train"],
            )
            id_datasets["val"] = ImageFolderDataset(
                split="val",
                root_dir=configs.dataset_root,
                transform=transforms["val"],
            )

        case "xrd":
            id_datasets["num_classes"] = 3

            id_datasets["train"] = TransformTorchDataset(
                dirpath=os.path.join(configs.dataset_root, "finalmat", "train"),
                transform=v2.Compose([RandomNoiseTransform(noise_level=0.002)]),
            )
            id_datasets["val"] = TransformTorchDataset(
                dirpath=os.path.join(configs.dataset_root, "finalmat", "val"),
                # transform=v2.Compose([RandomNoiseTransform(noise_level=0.1)]),
            )

        case "cifar10":
            id_datasets["num_classes"] = 10

            id_datasets["train"] = datasets.CIFAR10(
                root=configs.dataset_root,
                train=True,
                download=True,
                transform=transforms["train"],
            )
            id_datasets["val"] = datasets.CIFAR10(
                root=configs.dataset_root,
                train=False,
                download=True,
                transform=transforms["val"],
            )

        case "imagenet":
            id_datasets["num_classes"] = 1000

            id_datasets["train"] = ImageNetDataset(
                root=configs.dataset_root,
                split="train",
                transform=transforms["train"],
                fake_multiview=configs.use_fake_multiview,
            )

            id_datasets["val"] = ImageNetDataset(
                root=configs.dataset_root,
                split="val",
                transform=transforms["val"],
                fake_multiview=configs.use_fake_multiview,
            )

    return id_datasets
