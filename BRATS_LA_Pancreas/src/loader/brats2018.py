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
from dataset.brats2018 import Brats2018


def create_brats_dataset_fast(config: dict, mode: str):
    """
    Helper function to create a BraTS Inpainting Dataset instance.

    Args:
    - config (dict): Configuration dictionary.
    - mode (str): Either "tr", "vl", or "te".

    Returns:
    - BratsInpainting dataset instance.
    """
    crop_size = np.array(config["dataset"]["input_size"])
    modes_dict = {"tr": "train", "vl": "validation", "te": "test"}

    return Brats2018(
        crop_size=crop_size,
        mode=mode,
        vl_split=config["dataset"]["validation_split"],
        te_split=config["dataset"]["test_split"],
        **config["dataset"][modes_dict[mode]]["params"],
    )


def brats2018_loader(config, verbose: bool = True) -> dict:
    # train_list, val_list = split_dataset(
    #     config["path_to_data"], float(config["test_p"])
    # )
    modes_dict = {"tr": "train", "vl": "validation", "te": "test"}
    modes = ["tr", "vl", "te"]
    datasets, loaders = {}, {}

    for mode in modes:
        dataset = create_brats_dataset_fast(config, mode)
        datasets[mode] = dataset

    if verbose:
        print("BRATS 2018 3D:")
        for mode in modes:
            print(f"├──> Length of {mode}_dataset: {len(datasets[mode])}")

    # Create data loaders
    for mode in modes:
        loaders[mode] = DataLoader(
            datasets[mode], **config["data_loader"][modes_dict[mode]]
        )

    return {
        "tr": {"dataset": datasets["tr"], "loader": loaders["tr"]},
        "vl": {"dataset": datasets["vl"], "loader": loaders["vl"]},
        "te": {"dataset": datasets["te"], "loader": loaders["te"]},
    }
