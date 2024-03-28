# Model weights and Visulaizations

You can download the learned weights and their results in the following links.
   Dataset   | Visulization         | Weights
  -----------|----------------|----------------
   BRaTS 2018  | [[Download](https://mega.nz/folder/L4oQmAzL#OxnzxfbQZ8Oanr0iaKJ0qw)]| [[Download](https://mega.nz/file/O1w02BAS#BNK2p2QZ2x2MDE5HKEe7Xwms7J5DMPlyYBMnUEjjwQ8)] 
   LA (fold 0)      | [[Download](https://mega.nz/folder/SpImhTRD#Rcyh4ernRwpWuma9EOLh0g)]| [[Download](https://mega.nz/file/TkIG0DDD#bTwRrWe_ht3hoitKWsH88LePoN0vokmBHYo8Xhsk8wU)] 
   Pancreas (fold 0)      | [[Download](https://mega.nz/folder/2sAEiBSb#_apiwqhLrCT-RMDm5Qsg4g)]| [[Download](https://mega.nz/file/z84ijCzA#ZLW7_-qqpfY0qlwoJAbu6bD0hv29iiRBjuFpFe4Sdsc)] 

# BRaTS 2018 dataset
Download the dataset: [Download](https://mega.nz/file/ShpgECbC#rpZr_lWFr5ZIk8vXe7AVCFEPLZIFxFg7NnQ-Vk16LrM)

## Training 
```bash
python src/train_test.py -c brats-2018_lhunet_sgd
```
**Before running the code above please adjust the config's paths.** 

## Inference
Make sure in the config file you set the `test_mode` variable to `true` and give the path to the model you want to test in the `ckpt_path`.
```bash
python src/train_test.py -c brats-2018_lhunet_sgd
```

# Pancreas and LA dataset
Download the pre-processed datasets with their splits: [Pancreas](https://mega.nz/file/21p3ATLY#IZuAzvqXD8CymZZibN2oqLhfK0nZrBx8hWyk76SZRNk), [LA](https://mega.nz/file/K84Q3RKK#XDKPoSeYerwPJC7mcVyiTOM-Ydfv3TckDnAKkhpEVdY)

## Training
```bash
python src/train_test.py -c {dataset}_lhunet_sgd -f {fold_number}
```
instead of `{dataset}` you should write `pancreas` or `la` according to which dataset you want to train. regarding the `{fold_number}` for the Pancreas dataset it ranges from 0 till 3 (four-fold cross validation) and for the LA dataset it ranges from 0 till 4 (five-fold cross validation)
**Before running the code above please adjust the config's paths.** 

## Inference
Make sure in the config file you set the `just_test` variable to `true` and give the path to the model you want to test in the `ckpt_path`.
```bash
python src/train_test.py -c {dataset}_lhunet_sgd -f {fold_number}
```