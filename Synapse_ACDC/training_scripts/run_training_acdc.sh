#!/bin/sh

DATASET_PATH=/home/say26747/Desktop/datasets/DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=../OUTPUT/SCARTCH_with_shortcut_fixed_kernel_size
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_acdc 1 0 
