import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import os
import monai.transforms as T


class LAHeart(Dataset):
    """LA Dataset"""

    def __init__(
        self,
        base_dir=None,
        split="tr",
        data_type="LA",
        fold=0,
        crop_size=(96, 96, 96),
    ):
        self._base_dir = base_dir
        self.split = split

        if split == "tr":
            directory = os.path.join(
                self._base_dir, data_type, "Flods", f"train{fold}.list"
            )
            print(f"train{fold}.list")
        elif split == "vl" or split == "te":
            directory = os.path.join(
                self._base_dir, data_type, "Flods", f"test{fold}.list"
            )
            print(f"test{fold}.list")
        else:
            raise ValueError("Wrong split type!")

        with open(directory, "r") as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace("\n", "") for item in self.image_list]
        self.data_type = data_type  # LA or pancreas
        self.random_crop = T.Compose(
            [
                T.SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=crop_size,
                    constant_values=0,
                    mode="constant",
                ),
                T.RandSpatialCropd(keys=["image", "label"], roi_size=crop_size),
            ]
        )

        print("total {} unlabel_samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # print("Index: {}".format(idx))
        image_name = self.image_list[idx]
        # Determine the file path based on the data type
        if self.data_type == "Pancreas":
            file_path = os.path.join(self._base_dir, image_name)
        elif self.data_type == "LA":
            file_path = os.path.join(self._base_dir, image_name, "mri_norm2.h5")
        else:
            raise ValueError("Wrong data type!")

        # Open the file using h5py
        h5f = h5py.File(file_path, "r")  # +"/mri_norm2.h5", 'r')
        image = h5f["image"][:].astype(np.float32)
        image = np.expand_dims(image, axis=0)
        label = h5f["label"][:].astype(np.uint8)
        label = np.expand_dims(label, axis=0)
        sample = {"image": image, "label": label}
        if self.split == "tr":
            sample = self.random_crop(sample)
        sample["image"] = (
            torch.as_tensor(sample["image"]).permute(0, 3, 1, 2).type(torch.float32)
        )
        sample["label"] = (
            torch.as_tensor(sample["label"]).permute(0, 3, 1, 2).type(torch.long)
        )
        return sample
