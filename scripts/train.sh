#!/bin/bash


source activate lhunet

export nnUNet_raw="/path/to/raw_data"
export nnUNet_preprocessed="/path/to/pre_processed"
export nnUNet_results="/path/to/where_to_save_results"
export nnUNet_compile=False ### LHU-Net is not compatible with PyTorch Compile Yet

DatasetNumber=700 # choices: 700 (Synapse), 703 (Brats), 708 (LA), 709 (Lung)
trainer=lhunetSynapseTrainer # choices: lhunetSynapseTrainer, lhunetBratsTrainer, lhunetLATrainer, lhunetLungTrainer

# if you want to use Automatic Mixed Precision (AMP) training add the flag --amp

nnUNetv2_train $DatasetNumber 3d_fullres 0 -tr $trainer # --amp
