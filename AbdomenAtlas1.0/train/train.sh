#!/bin/bash
backbone=lhunet
datapath="/media/say26747/EC2426FA2426C782/8k_CT_scan/AbdomenAtlas" # change path accordingly
continue_path="${HOME}/out/AbdomenAtlas1.0.lhunet.cache.lr0.01.epoch.1000/lhunet.pth"  # change path accordingly

python train.py --data_root_path $datapath \
 --num_workers 5 --log_name AbdomenAtlas1.0.$backbone.cache.lr0.01.epoch.1000 --backbone $backbone --lr 0.01 \
  --batch_size 2 --max_epoch 1000 --cache_dataset #--continue_training True --continue_path $continue_path 