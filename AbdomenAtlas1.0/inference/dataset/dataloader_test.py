from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
    CastToTyped,
    EnsureTyped,
)
import os
import numpy as np
from monai.data import (
    DataLoader,
    Dataset,
    list_data_collate,
)
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()


def get_loader(args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("trilinear"),
            ),  # process h5 to here
            CastToTyped(keys=["image"], dtype=(np.float32)),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        ]
    )
    ## test dict part

    folders = [
        name
        for name in os.listdir(args.data_root_path)
        if os.path.isdir(os.path.join(args.data_root_path, name))
    ]
    test_img = []
    test_name_img = []
    for name_img in folders:
        test_img.append(os.path.join(args.data_root_path, name_img, "ct.nii.gz"))
        test_name_img.append(name_img)
    data_dicts_test = [
        {"image": image, "name_img": name_img}
        for image, name_img in zip(test_img, test_name_img)
    ]
    print("test len {}".format(len(data_dicts_test)))

    # print(data_dicts_test)

    test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    return test_loader, val_transforms


