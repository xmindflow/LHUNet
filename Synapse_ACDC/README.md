# Model weights and Visulaizations

You can download the learned weights in the following and their results.
   Dataset   | Visulization         | Weights
  -----------|----------------|----------------
   Synapse  | [[Download]()]| [[Download]()] 
   ACDC       | [[Download]()]| [[Download]()] 

# Synapse dataset
Download the preprocessed data (It follows the nnU-Net preprocessing): [Download](https://mega.nz/file/fopXXQyQ#X0KOHXTakaSz3ORJH4NS3fG8iB13JJe6Eq0W3eA02tE)

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

after running the inference for more deltailed metrics you could do:
```bash
cd evaluation_scripts
python inference_synapse_multiprocess.py 0
```

# ACDC dataset
Download the preprocessed data (It follows the nnU-Net preprocessing): [Download](https://mega.nz/file/G0J3CQjY#tNw1yFa_6I7rDDvpu4cIJxEepWCVIk9J08i5tIln47Q)

## Training
```bash
cd training_scripts
./run_training_acdc.sh
```
**Before running the code above please adjust the script's paths.**


## Inference 
```bash
cd evaluation_scripts
./run_evaluation_acdc.sh
```
**Before running the code above please adjust the script's paths.**