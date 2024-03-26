#!/usr/bin/env python3
# encoding: utf-8
# Code modified from https://github.com/Wangyixinxin/ACN
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random
import monai.transforms as T


class Brats2021(Dataset):
    def __init__(
        self,
        data_root: str,
        crop_size: tuple or list = [128, 128, 128],
        modalities: list = ["t1", "t1ce", "t2", "flair"],
        normalization: bool = True,
        augmentation: bool = True,
        mode: str = "tr",
        vl_split: float = 0.1,
        te_split: float = 0.2,
        p: float = 0.5,
    ):  # , train=True,):
        super(Brats2021, self).__init__()
        self.data_root = data_root
        self.modalities = modalities
        # self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.vl_split = vl_split
        self.te_split = te_split
        self.mode = mode
        self.test_split = te_split
        self.val_split = vl_split
        self.p = p

        patients_dir = glob.glob(os.path.join(data_root, "BraTS2021_*"))
        patients_dir.sort()
        self.split_dataset(patients_dir)
        self.define_transforms()

    def __len__(self):
        return len(self.patients_list)

    def __getitem__(self, index):
        data_dict = {
            "volume": 0,
            "seg-volume": 0,
            "patient_name": os.path.basename(self.patients_list[index]),
            "affinity": 0,
        }
        patient_path = self.patients_list[index]
        modalities = list(self.modalities) + ["seg"]
        volumes, seg_volume, data_dict["affinity"] = self._load_data(
            patient_path, modalities
        )

        # we have to change the ET lable which is 4 to 3
        seg_volume[seg_volume == 4] = 3

        volumes = np.stack(volumes, axis=0).astype(np.float32)  # [C, H, W, D]
        seg_volume = np.expand_dims(seg_volume, axis=0).astype(np.uint8)  # [1, H, W, D]

        if self.mode == "tr":
            volumes, seg_volume = self._aug_sample(volumes, seg_volume)

        data_dict["volume"] = (
            torch.as_tensor(volumes).permute(0, 3, 1, 2).float()
        )  # [C, D, H, W]
        data_dict["seg-volume"] = (
            torch.as_tensor(seg_volume).permute(0, 3, 1, 2).type(torch.uint8)
        )  # [1, D, H, W]

        return data_dict

    def define_transforms(self):
        keys = ["volume", "seg-volume"]
        self.pad_or_crop_transform = T.ResizeWithPadOrCropd(
            keys=keys,
            spatial_size=self.crop_size,
            mode="constant",
            constant_values=0,
            value=0,
        )
        self.aug_sample_transform = T.Compose(
            [
                T.RandRotated(
                    keys=keys,
                    prob=0.15,
                    range_x=np.deg2rad(30),
                    range_y=np.deg2rad(30),
                    range_z=np.deg2rad(30),
                    mode=("trilinear", "nearest"),
                    padding_mode="zeros",
                ),
                T.RandZoomd(
                    keys=keys,
                    prob=0.15,
                    min_zoom=0.7,
                    max_zoom=1.4,
                    mode=("trilinear", "nearest"),
                    padding_mode="constant",
                    constant_values=0,
                ),
                T.RandSpatialCropd(
                    keys=keys,
                    random_center=True,
                    roi_size=self.crop_size,
                ),
                T.RandFlipd(
                    keys=keys,
                    prob=self.p,
                    spatial_axis=0,
                ),  # vertical flip
                T.RandFlipd(
                    keys=keys,
                    prob=self.p,
                    spatial_axis=1,
                ),  # horizontal flip
                T.RandFlipd(
                    keys=keys,
                    prob=self.p,
                    spatial_axis=2,
                ),  # depth flip
                T.RandGaussianNoised(keys="volume", std=0.05, prob=0.15),
                T.RandGaussianSmoothd(keys="volume", prob=0.15),
                T.RandScaleIntensityd(keys="volume", factors=0.3, prob=0.15),
                T.RandShiftIntensityd(keys="volume", offsets=1, prob=0.15),
                T.RandAdjustContrastd(keys="volume", gamma=(0.7, 1.5), prob=0.1),
            ]
        )

    def _load_data(self, patient_path, modalities):
        volumes = []
        for modality in modalities:
            patient_id = os.path.basename(patient_path)
            volume_path = os.path.join(
                patient_path, patient_id + "_" + modality + ".nii.gz"
            )
            nib_file = nib.load(volume_path)
            volume = nib_file.get_fdata()
            affinity = nib_file.affine
            if not modality == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)  # [h, w, d]
        return volumes[:-1], volumes[-1], affinity

    def split_dataset(self, patients_dir):
        vl_index = int(len(patients_dir) * self.vl_split)
        te_index = int(len(patients_dir) * self.te_split)
        if self.mode == "tr":
            self.patients_list = patients_dir[te_index + vl_index :]
        elif self.mode == "te":
            self.patients_list = patients_dir[:te_index]
        elif self.mode == "vl":
            self.patients_list = patients_dir[te_index : te_index + vl_index]
        else:
            raise ValueError("mode must be tr, vl or te")

    def _aug_sample(self, volumes, mask):
        data_dict = {"volume": volumes, "seg-volume": mask}
        transformed = self.aug_sample_transform(data_dict)
        return transformed["volume"], transformed["seg-volume"]

    def normlize(self, x):
        if x.max() == x.min():
            return x
        return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    data_root = "E:\Brats 2018 data"
    crop_size = [128, 128, 128]
    dataset = Brats2021(data_root, crop_size=crop_size, mode="tr")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    for i, batch in enumerate(dataloader):
        x = batch["volume"]
        y = batch["seg-volume"]
        print(f"patient name: {batch['patient_name']}")
        print(x.shape, y.shape)
        break
