#!/bin/sh

DATASET_PATH=/home/say26747/Desktop/datasets/DATASET_Synapse_lhunet
CHECKPOINT_PATH=../OUTPUT/test_synapse

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export lhunet_preprocessed="$DATASET_PATH"/lhunet_raw/lhunet_raw_data/Task02_Synapse
export lhunet_raw_data_base="$DATASET_PATH"/lhunet_raw

python ../lhunet/run/run_training.py 3d_fullres lhunet_trainer_synapse 2 0 -val --valbest
