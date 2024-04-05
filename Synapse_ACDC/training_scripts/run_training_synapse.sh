#!/bin/sh

# DATASET_PATH=/cabinet/dataset/Synapse/NN-Unet/DATASET_Synapse_lhunet
DATASET_PATH=/media/say26747/EC2426FA2426C782/LHUNET/DATASET_Synapse

export PYTHONPATH=.././
# export RESULTS_FOLDER=/cabinet/yousef/Synapse-nnunet/output_synapse_lhunet_BEST_kernels_3_dec_5
export RESULTS_FOLDER=/home/say26747/Desktop/TEST
export lhunet_preprocessed="$DATASET_PATH"/lhunet_raw/lhunet_raw_data/Task02_Synapse
export lhunet_raw_data_base="$DATASET_PATH"/lhunet_raw

python ../lhunet/run/run_training.py 3d_fullres lhunet_trainer_synapse 2 0
