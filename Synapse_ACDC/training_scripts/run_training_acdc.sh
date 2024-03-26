#!/bin/sh

DATASET_PATH=/media/say26747/EC2426FA2426C782/LHUNET/DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=/home/say26747/Desktop/TEST
export lhunet_preprocessed="$DATASET_PATH"/lhunet_raw/lhunet_raw_data/Task01_ACDC
export lhunet_raw_data_base="$DATASET_PATH"/lhunet_raw

python ../lhunet/run/run_training.py 3d_fullres lhunet_trainer_acdc 1 0 
