import os
from glob import glob
import shutil

join = os.path.join

current_path = os.getcwd()
nnunet_path= os.path.join(current_path, 'nnUNet')
source_path = os.path.join(current_path, "src")

########### Copying files to nnUNet directory ###########
# Copying Training files
trainer_files= glob(os.path.join(source_path,"*Trainer.py"))
for file in trainer_files:
    shutil.copy(file, join(nnunet_path, "nnunetv2", "training", "nnUNetTrainer", os.path.basename(file)))

######### Copying Model files ###########
model_folder = os.path.join(source_path, "lhunet")
### copy the folder to nnUNet model_sharing
model_destination = os.path.join(nnunet_path, "nnunetv2", "model_sharing", "lhunet")
if not os.path.exists(model_destination):
    os.makedirs(model_destination)
shutil.copytree(model_folder, model_destination, dirs_exist_ok=True)


####### copy the run file and prediction file #######
run_file = os.path.join(source_path, "run_training.py")
shutil.copy(run_file, join(nnunet_path, "nnunetv2", "run", os.path.basename(run_file)))

prediction_file = os.path.join(source_path, "predict_from_raw_data.py")
shutil.copy(prediction_file, join(nnunet_path, "nnunetv2", "inference", os.path.basename(prediction_file)))

print("Files copied successfully to nnUNet directory.")