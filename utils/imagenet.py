import os
from typing import Any, Callable, List, Tuple

import numpy as np
from PIL import Image
from torchvision import datasets


class ImageNetDataset(datasets.ImageNet):
    """
    overload of ImageNet Dataset to allow for fake multiview
    used for ViSM and Ave-ViM methods
    """

    def __init__(self, root: str, split: str = "train", fake_multiview: bool = False, **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)
        self.fake_multiview = fake_multiview
        self.num_images = 4

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if self.fake_multiview:
                sample = [self.transform(sample) for _ in range(self.num_images)]
            else:
                sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class SVHNDataset(datasets.SVHN):
    """
    overload of SVHN Dataset to allow for fake multiview
    used for ViSM and Ave-ViM methods
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
        fake_multiview: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.fake_multiview = fake_multiview
        self.num_images = 4

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            if self.fake_multiview:
                img = [self.transform(img) for _ in range(self.num_images)]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class INaturalistDataset(datasets.INaturalist):
    """
    overload of iNaturalist Dataset to allow for fake multiview
    used for ViSM and Ave-ViM methods
    """

    def __init__(
        self,
        root: str,
        version: str = "2021_train",
        target_type: List[str] | str = "full",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
        fake_multiview: bool = False,
    ) -> None:
        super().__init__(root, version, target_type, transform, target_transform, download)
        self.fake_multiview = fake_multiview
        self.num_images = 4

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        target: Any = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            if self.fake_multiview:
                img = [self.transform(img) for _ in range(self.num_images)]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
