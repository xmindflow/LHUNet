# Model weights and Visulaizations

You can download the learned weights in the following and their results.
   Dataset   | Visulization         | Weights
  -----------|----------------|----------------
   BRaTS 2018  | [[Download]()]| [[Download]()] 
   LA (fold 0)      | [[Download]()]| [[Download]()] 
   Pancreas (fold 0)      | [[Download]()]| [[Download]()] 

# BRaTS 2018 dataset
Download the dataset: [Download](https://mega.nz/file/ShpgECbC#rpZr_lWFr5ZIk8vXe7AVCFEPLZIFxFg7NnQ-Vk16LrM)

## Training 
```bash
python src/train_test.py -c brats-2018_lhunet_sgd
```
**Before running the code above please adjust the config's paths.** 

## Inference
Make sure in the config file you set the `just_test` variable to `true` and give the path to the model you want to test in the `ckpt_path`.
```bash
python src/train_test.py -c brats-2018_lhunet_sgd
```

# Pancreas and LA dataset
Download the pre-processed datasets with their splits: [Pancreas](https://mega.nz/file/21p3ATLY#IZuAzvqXD8CymZZibN2oqLhfK0nZrBx8hWyk76SZRNk), [LA](https://mega.nz/file/K84Q3RKK#XDKPoSeYerwPJC7mcVyiTOM-Ydfv3TckDnAKkhpEVdY)

## Training
```bash
python src/train_test.py -c {dataset}_lhunet_sgd -f {fold_number}
```
instead of {dataset} you should write `pancreas` or `LA` according to which dataset you want to train. regarding the {fold_number} for the Pancreas dataset it ranges from 0 till 3 (four-fold cross validation) and for the LA dataset it ranges from 0 till 4 (five-fold cross validation)
**Before running the code above please adjust the config's paths.** 

## Inference
Make sure in the config file you set the `just_test` variable to `true` and give the path to the model you want to test in the `ckpt_path`.
```bash
python src/train_test.py -c {dataset}_lhunet_sgd -f {fold_number}
```