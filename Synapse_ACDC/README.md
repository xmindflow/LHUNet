# Model weights and Visulaizations

You can download the learned weights in the following and their results.
   Dataset   | Visulization         | Weights
  -----------|----------------|----------------
   Synapse  | [[Download]()]| [[Download]()] 
   ACDC       | [[Download]()]| [[Download]()] 

# Synapse dataset
1. Download the preprocessed data (It follows the nnU-Net preprocessing): [Download]()

## Training
```bash
cd training_scripts
./run_training_synapse.sh
```
**Before running the code above please adjust the script's paths.** 

## Inference 
```bash
cd evaluation_scripts
./run_evaluation_synapse.sh
```
**Before running the code above please adjust the script's paths.**
