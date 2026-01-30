import os
import glob
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
from joblib import Parallel, delayed
from tqdm import tqdm
import nibabel as nib


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

def read_nii(file_path):
    """Read NIfTI file and return its array."""
    return nib.load(file_path).get_fdata()

def calculate_dice(pred, label):
    """Calculate the Dice coefficient."""
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2.0 * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def calculate_hd(pred, gt):
    """Calculate Hausdorff distance."""
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.hd95(pred, gt)
    else:
        return 0

def process_label(label):
    """Process the label to separate different organs."""
    organs = [1, 2, 3, 4, 6, 7, 8, 11]
    return tuple(label == organ for organ in organs)

def process_pair(label_path, infer_path):
    """Process a pair of label and infer files."""
    dice_scores = []
    hd_scores = []
    for index, organ in class_map_abdomenatlas_1_0.items():
        gt_file=nib.load(os.path.join(label_path, f'{organ}.nii.gz')).get_fdata().astype(np.bool_)
        pred_file=nib.load(os.path.join(infer_path, f'{organ}.nii.gz')).get_fdata().astype(np.bool_)    
        dice_scores.append(calculate_dice(pred_file, gt_file))
        hd_scores.append(calculate_hd(pred_file, gt_file))

    return label_path, infer_path, dice_scores, hd_scores

def write_results(file, results):
    """Write the processing results to a file."""
    with open(file, 'a') as fw:
        organs = list(class_map_abdomenatlas_1_0.values())
        hd_dict = {organ: [] for organ in organs}
        dice_dict = {organ: [] for organ in organs}
        total_hd = []
        total_dice = []
                
        for label_path, infer_path, dice_scores, hd_scores in results:
            fw.write("*" * 20 + "\n")
            fw.write(f"{os.path.basename(os.path.dirname(infer_path))}\n")
            fw.write("\n".join([f"Dice_{organ}: {score:.4f}" for organ, score in zip(organs, dice_scores)]))
            for organ, score in zip(organs, dice_scores):
                dice_dict[organ].append(score)
            fw.write("\n")
            fw.write("\n".join([f"HD_{organ}: {score:.4f}" for organ, score in zip(organs, hd_scores)]))
            for organ, score in zip(organs, hd_scores):
                hd_dict[organ].append(score)
            fw.write("\n")
            fw.write(f"Mean Dice: {np.mean(dice_scores):.4f}\n")
            fw.write(f"Mean HD: {np.mean(hd_scores):.4f}\n")
            
        fw.write("*" * 20 + "\n")
        fw.write("*" * 20 + "\n")
        for organ in organs:
            fw.write(f"Mean Dice_{organ}: {np.mean(dice_dict[organ]):.4f}\n")
            fw.write(f"Mean HD_{organ}: {np.mean(hd_dict[organ]):.4f}\n")
            total_dice.append(np.mean(dice_dict[organ]))
            total_hd.append(np.mean(hd_dict[organ]))
            
        fw.write(f"Total Mean Dice: {np.mean(total_dice):.4f}\n")
        fw.write(f"Total Mean HD: {np.mean(total_hd):.4f}\n")


def test(args):
    gt_path=sorted(glob.glob(os.path.join(args.seg_path, 'BDMAP*', 'segmentations')))
    pred_path=sorted(glob.glob(os.path.join(args.pred_path, 'BDMAP*', 'predictions')))

    print("Loading success...")
    print('LEN GROUND TRUTH',len(gt_path))
    print('LEN PREDICTIONS',len(pred_path))

    output_dir = os.path.join(args.pred_path, "inference_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dice_pre.txt")
    print(f"Writing results to {output_file}")

    results = Parallel(n_jobs=-1)(delayed(process_pair)(label_path, infer_path) for label_path, infer_path in tqdm(zip(gt_path, pred_path)))
    write_results(output_file, results)

    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_path", default=None, type=str, required=True, help="segmentation path.")
    parser.add_argument("--pred_path", default=None, type=str, required=True, help="prediction path.")
    args = parser.parse_args()
    test(args)
