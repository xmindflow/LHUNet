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
from dataset.la_heart import LAHeart


def create_LA_dataset_fast(config: dict, mode: str):
    """
    Helper function to create a BraTS Inpainting Dataset instance.

    Args:
    - config (dict): Configuration dictionary.
    - mode (str): Either "tr", "vl", or "te".

    Returns:
    - BratsInpainting dataset instance.
    """
    crop_size = np.array(config["dataset"]["crop_size"])
    modes_dict = {"tr": "train", "vl": "validation", "te": "test"}

    return LAHeart(
        crop_size=crop_size,
        split=mode,
        data_type=config["datatype"],
        fold=config["fold"],
        base_dir=config["dataset"]["path_to_data"],
    )


def la_heart_loader(config, verbose: bool = True) -> dict:
    # train_list, val_list = split_dataset(
    #     config["path_to_data"], float(config["test_p"])
    # )
    modes_dict = {"tr": "train", "vl": "validation", "te": "test"}
    modes = ["tr", "vl", "te"]
    datasets, loaders = {}, {}

    for mode in modes:
        dataset = create_LA_dataset_fast(config, mode)
        datasets[mode] = dataset

    if verbose:
        print(f"{config['datatype']} 3D:")
        for mode in modes:
            print(f"├──> Length of {mode}_dataset: {len(datasets[mode])}")

    # Create data loaders
    def worker_init_fn(worker_id):
        random.seed(1337 + worker_id)

    for mode in modes:
        loaders[mode] = DataLoader(
            datasets[mode],
            worker_init_fn=worker_init_fn,
            **config["data_loader"][modes_dict[mode]],
        )

    return {
        "tr": {"dataset": datasets["tr"], "loader": loaders["tr"]},
        "vl": {"dataset": datasets["vl"], "loader": loaders["vl"]},
        "te": {"dataset": datasets["te"], "loader": loaders["te"]},
    }
