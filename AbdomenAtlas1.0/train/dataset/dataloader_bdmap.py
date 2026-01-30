from monai.transforms import (
    Compose,
    CropForegroundd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
    EnsureChannelFirstd,
    ToNumpyd,
    CastToTyped,
)

import nibabel as nib
import os

import numpy as np
from typing import Optional, Union
import time


from monai.data import (
    DataLoader,
    Dataset,
    list_data_collate,
    DistributedSampler,
    SmartCacheDataset,
)
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
from monai.data import MetaTensor

DEFAULT_POST_FIX = PostFix.meta()

class_map_abdomenatlas_1_0 = {
    1: "aorta",
    2: "gall_bladder",
    3: "kidney_left",
    4: "kidney_right",
    5: "liver",
    6: "pancreas",
    7: "postcava",
    8: "spleen",
    9: "stomach",
}


class LoadSelectedImaged(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.int16,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(
            reader,
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            *args,
            **kwargs,
        )
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys is not None else ensure_tuple_rep(None, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            loaded_data = self._loader(d[key], reader)
            d[key], d_meta = loaded_data if not self._loader.image_only else (loaded_data, {})
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            if meta_key in d and not self.overwriting:
                raise KeyError(f"Metadata with key {meta_key} already exists and overwriting is False.")
            d[meta_key] = d_meta
        
        label_parent_path = d["label_parent"]
        name=os.path.basename(os.path.dirname(label_parent_path))
        label_organs = class_map_abdomenatlas_1_0
        first_organ_path = os.path.join(label_parent_path, f"{label_organs[1]}.nii.gz")
        first_organ_img = nib.load(first_organ_path)
        W, H, D = first_organ_img.shape
        label = np.zeros((W, H, D), dtype=np.uint8)

        for index, organ in label_organs.items():
            organ_path = os.path.join(label_parent_path, f"{organ}.nii.gz")
            organ_data = nib.load(organ_path).get_fdata()
            label[organ_data > 0] = index

        d["label"] = MetaTensor(label,meta=d[f"{key}_meta_dict"])
        if d["label"].sum() == 0:
            print(label_parent_path,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return d
    

def get_loader(args):
    train_transforms = Compose(
        [
            LoadSelectedImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("trilinear", "nearest"),
            ),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=5,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToNumpyd(keys=["image", "label"]),
        ]
    )
    ## training dict part
    train_img = []
    train_lbl_parents = []
    train_name = []

    for item in args.dataset_list:
        for line in open(os.path.join(args.data_txt_path, item + ".txt")):
            name = line.strip().split("\t")[0]
            train_img_path = os.path.join(args.data_root_path, name, "ct.nii.gz")
            folder_name = os.path.join(args.data_root_path, name, "segmentations/")
            train_img.append(train_img_path)
            train_lbl_parents.append(folder_name)
            train_name.append(name)

    data_dicts_train = [
        {"image": image, "label_parent": label, "name": name}
        for image, label, name in zip(train_img, train_lbl_parents, train_name)
    ]
    print("train len {}".format(len(data_dicts_train)))
    
    # save the start time
    start_time = time.time()  
    # print("start time: ", start_time)

    if args.cache_dataset:
        train_dataset = SmartCacheDataset(
            data=data_dicts_train,
            transform=train_transforms,
              cache_num=args.cache_num,
              cache_rate=args.cache_rate,
              num_init_workers=args.num_workers,
              num_replace_workers=args.num_workers,
        )
    else:
        train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    train_sampler = (
        DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True)
        if args.dist
        else None
    )
    end_time = time.time()
    # print the time it took to load the data in a readable format
    print(f"Time to load data: {end_time - start_time} seconds")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, train_sampler, train_dataset
