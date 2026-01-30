import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import shutil
import nibabel as nib
from monai.inferers import sliding_window_inference
from dataset.dataloader_test import get_loader 
from model.network_architecture.synapse.lhunet.models.v7 import LHUNet
import time
from monai.transforms import (
    Compose,
    Activationsd,
    Invertd,
    AsDiscreted,
    CastToTyped,
)
from monai.data import decollate_batch

torch.multiprocessing.set_sharing_strategy("file_system")

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


def validation(model, ValLoader, val_transforms, args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()

    post_transforms = Compose(
        [
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=val_transforms,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            AsDiscreted(keys="pred", argmax=True),
            CastToTyped(keys="pred", dtype=(np.uint8)),
            # SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )

    if args.customize:
        count = 0
        for index, batch in enumerate(tqdm(ValLoader)):
            # print(batch.keys())
            image, name = batch["image"].to(args.device), batch["name_img"]
            # print(type(image))
            image_file_path = os.path.join(args.data_root_path, name[0], "ct.nii.gz")
            case_save_path = os.path.join(save_dir, name[0].split("/")[0])
            # print(case_save_path)
            if not os.path.isdir(case_save_path):
                os.makedirs(case_save_path)
            organ_seg_save_path = os.path.join(
                save_dir, name[0].split("/")[0], "predictions"
            )
            if not os.path.isdir(organ_seg_save_path):
                os.makedirs(organ_seg_save_path)
            # print(image_file_path)
            # print(image.shape)
            # print(name)
            if args.copy_ct:
                destination_ct = os.path.join(case_save_path, "ct.nii.gz")
                if not os.path.isfile(destination_ct):
                    shutil.copy(image_file_path, destination_ct)
                    print("CT scans copied successfully.")
            name = (
                name.item() if isinstance(name, torch.Tensor) else name
            )  # Convert to Python str if it's a Tensor
            original_affine = nib.load(image_file_path).affine
            # print(f"original image shape: {nib.load(image_file_path).shape}")
            # print("Image: {}, shape: {}".format(name, image.shape))
            with torch.inference_mode():
                # print("Image: {}, shape: {}".format(name[0], image.shape))
                # print the time in the format of hour:minute:second
                start_time = time.time()
                print(
                    f"starting processing the image ({name}) at time: {time.strftime('%H:%M:%S', time.localtime(start_time))}"
                )
                val_outputs = sliding_window_inference(
                    inputs=image,
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=args.overlap,
                    mode="gaussian",
                    sw_device="cuda",
                    device="cuda",
                )
                end_time = time.time()
                print(
                    f"finished processing image ({name}) at time: {time.strftime('%H:%M:%S', time.localtime(end_time))}"
                )
                # print the processing time in seconds
                print(f"processing time: {end_time - start_time} seconds")
                # print(f"processing time: {end_time - start_time}")
                # print(f"val_outputs shape: {val_outputs.shape}")

            batch["pred"] = val_outputs
            batch = [post_transforms(i) for i in decollate_batch(batch)]
            # print(batch[0]["pred"].shape)
            # print(torch.unique(batch[0]["pred"]))

            pred = batch[0]["pred"].cpu().numpy()[0]
            # print(pred.shape)
            for index, organ_name in class_map_abdomenatlas_1_0.items():
                pred_class = pred == index
                file_path_pattern = os.path.join(
                    organ_seg_save_path, f"{organ_name}.nii.gz"
                )
                nib.save(
                    nib.Nifti1Image(pred_class.astype(np.uint8), original_affine),
                    file_path_pattern,
                )
                
            count += 1
            print("[{}/{}] Saved {}".format(count, len(ValLoader), name[0]))

        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument(
        "--dist",
        dest="dist",
        type=bool,
        default=False,
        help="distributed training or not",
    )
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epoch", default=0, type=int)
    ## logging
    parser.add_argument("--save_dir", default="...", help="The dataset save path")
    ## model load
    parser.add_argument(
        "--checkpoint", default="...", help="The path of trained checkpoint"
    )
    parser.add_argument("--pretrain", default="...", help="The path of pretrain model")
    ## hyperparameter
    parser.add_argument(
        "--max_epoch", default=1000, type=int, help="Number of training epoches"
    )
    parser.add_argument(
        "--store_num", default=10, type=int, help="Store model how often"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight Decay")

    ## dataset
    parser.add_argument("--data_root_path", default="...", help="data root path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="workers numebr for DataLoader"
    )
    parser.add_argument(
        "--a_min", default=-175, type=float, help="a_min in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--a_max", default=250, type=float, help="a_max in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--space_x", default=1.5, type=float, help="spacing in x direction"
    )
    parser.add_argument(
        "--space_y", default=1.5, type=float, help="spacing in y direction"
    )
    parser.add_argument(
        "--space_z", default=1.5, type=float, help="spacing in z direction"
    )
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument(
        "--num_samples", default=1, type=int, help="sample number in each ct"
    )
    parser.add_argument("--num_class", default=10, type=int, help="class number")
    parser.add_argument("--map_type", default="vertebrae", help="class map type")
    parser.add_argument("--overlap", default=0.5, type=float, help="overlap")
    parser.add_argument(
        "--copy_ct",
        action="store_true",
        default=False,
        help="copy ct file to the save_dir",
    )

    parser.add_argument("--phase", default="test", help="train or validation or test")
    parser.add_argument(
        "--original_label",
        action="store_true",
        default=False,
        help="whether dataset has original label",
    )
    parser.add_argument(
        "--cache_dataset",
        action="store_true",
        default=False,
        help="whether use cache dataset",
    )
    parser.add_argument(
        "--store_result",
        action="store_true",
        default=False,
        help="whether save prediction result",
    )
    parser.add_argument(
        "--cache_rate",
        default=0.6,
        type=float,
        help="The percentage of cached data in total",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="The entire inference process is performed on the GPU ",
    )
    parser.add_argument("--threshold_organ", default="Pancreas Tumor")
    parser.add_argument(
        "--backbone", default="unet", help="backbone [swinunetr or unet]"
    )
    parser.add_argument("--create_dataset", action="store_true", default=False)
    parser.add_argument("--suprem", action="store_true", default=False)
    parser.add_argument("--customize", action="store_true", default=False)

    args = parser.parse_args()

    
    if args.customize:
        model = LHUNet(
            spatial_shapes=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.num_class,
            do_ds=False,
            # encoder params
            cnn_kernel_sizes=[3, 3],
            cnn_features=[12, 16],
            cnn_strides=[2, 2],
            cnn_maxpools=[True, True],
            cnn_dropouts=0.0,
            cnn_blocks="nn",  # n= resunet, d= deformconv, b= basicunet,
            hyb_kernel_sizes=[3, 3, 3],
            hyb_features=[32, 64, 128],
            hyb_strides=[2, 2, 2],
            hyb_maxpools=[True, True, True],
            hyb_cnn_dropouts=0.0,
            hyb_tf_proj_sizes=[64, 32, 0],
            hyb_tf_repeats=[1, 1, 1],
            hyb_tf_num_heads=[8, 8, 16],
            hyb_tf_dropouts=0.1,
            hyb_cnn_blocks="nnn",  # n= resunet, d= deformconv, b= basicunet,
            hyb_vit_blocks="SSC",  # s= dlka_special_v2, S= dlka_sp_seq, c= dlka_channel_v2, C= dlka_ch_seq,
            # hyb_vit_sandwich= False,
            hyb_skip_mode="cat",  # "sum" or "cat",
            hyb_arch_mode="residual",  # sequential, residual, parallel, collective,
            hyb_res_mode="sum",  # "sum" or "cat",
            # decoder params
            dec_hyb_tcv_kernel_sizes=[5, 5, 5],
            dec_cnn_tcv_kernel_sizes=[5, 7],
            dec_cnn_blocks=None,
            dec_tcv_bias=False,
            dec_hyb_tcv_bias=False,
            dec_hyb_kernel_sizes=None,
            dec_hyb_features=None,
            dec_hyb_cnn_dropouts=None,
            dec_hyb_tf_proj_sizes=None,
            dec_hyb_tf_repeats=None,
            dec_hyb_tf_num_heads=None,
            dec_hyb_tf_dropouts=None,
            dec_cnn_kernel_sizes=None,
            dec_cnn_features=None,
            dec_cnn_dropouts=None,
            dec_hyb_cnn_blocks=None,
            dec_hyb_vit_blocks=None,
            # dec_hyb_vit_sandwich= None,
            dec_hyb_skip_mode=None,
            dec_hyb_arch_mode="collective",  # sequential, residual, parallel, collective, sequential-lite,
            dec_hyb_res_mode=None,
        )
    

    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')["net"])
    print("Use pretrained weights")
    model.cuda()
    torch.backends.cudnn.benchmark = True
    # model=torch.compile(model)
    test_loader, val_transforms = get_loader(args)
    # torch.cuda.set_device(args.device)

    validation(model, test_loader, val_transforms, args)


if __name__ == "__main__":
    main()
