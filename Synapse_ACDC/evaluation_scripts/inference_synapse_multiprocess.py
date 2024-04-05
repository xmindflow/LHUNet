import os
import glob
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
from joblib import Parallel, delayed
from tqdm import tqdm

def read_nii(file_path):
    """Read NIfTI file and return its array."""
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

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
    label, infer = read_nii(label_path), read_nii(infer_path)
    label_organs = process_label(label)
    infer_organs = process_label(infer)

    dice_scores = [calculate_dice(infer_org, label_org) for infer_org, label_org in zip(infer_organs, label_organs)]
    hd_scores = [calculate_hd(infer_org, label_org) for infer_org, label_org in zip(infer_organs, label_organs)]

    return label_path, infer_path, dice_scores, hd_scores

def write_results(file, results):
    """Write the processing results to a file."""
    with open(file, 'a') as fw:
        organs = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"]
        hd_dict = {organ: [] for organ in organs}
        dice_dict = {organ: [] for organ in organs}
        total_hd = []
        total_dice = []
                
        for label_path, infer_path, dice_scores, hd_scores in results:
            fw.write("*" * 20 + "\n")
            fw.write(f"{os.path.basename(infer_path)}\n")
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


def test(fold):
    path = "/home/say26747/Desktop/git/ACDC-nnunet/OUTPUT/test_synapse/lhunet/3d_fullres/Task002_Synapse/lhunet_trainer_synapse__lhunet_Plansv2.1/fold_0"
    label_list = sorted(glob.glob(os.path.join(path, "gt_niftis", "subset_for_inference", "*.nii.gz")))
    infer_list = sorted(glob.glob(os.path.join(path, "validation_raw_postprocessed", "*.nii.gz")))
    print("Loading success...")
    print('LEN LABELS',len(label_list))
    print('LEN PRED',len(infer_list))

    output_dir = os.path.join(path, "inference_results", fold)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dice_pre.txt")
    print(f"Writing results to {output_file}")

    results = Parallel(n_jobs=-1)(delayed(process_pair)(label_path, infer_path) for label_path, infer_path in tqdm(zip(label_list, infer_list)))
    write_results(output_file, results)

    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIfTI files for medical imaging analysis.")
    parser.add_argument("fold", help="Fold name for processing.")
    args = parser.parse_args()

    test(args.fold)
