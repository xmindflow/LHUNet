#!/bin/bash
pretrainpath='../weight/lhunet.pth'
savepath=./AbdomenAtlasDemoPredict # change path accordingly
datarootpath='/home/say26747/Downloads/test_examples/AbdomenAtlasTest' # change path accordingly    

python inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize 

### run the metric calculation (you can comment it out if you don't want to run it)
python inference_metrics_multiprocess.py --seg_path $datarootpath --pred_path $savepath 