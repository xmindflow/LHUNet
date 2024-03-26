import os
import warnings
import pytorch_lightning as pl
import yaml
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
import argparse

from loader.dataloaders import get_dataloaders
from utils import load_config, print_config
from models.get_models import get_model
from lightning_module_brats import SemanticSegmentation3D as brats_module
from lightning_module_pancreas import SemanticSegmentation3D as la_heart_module


warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("-c", "--config", type=str, required=True, help="config file")
    parser.add_argument(
        "-f", "--fold", type=int, required=False, default=-1, help="fold"
    )  # only for pancrease and LA dataset
    return parser.parse_args()


def set_seed(seed=0):
    pl.seed_everything(seed)


def get_base_directory():
    return (
        os.getcwd()
        if os.path.basename(os.getcwd()) != "src"
        else os.path.dirname(os.getcwd())
    )


def configure_logger(config, parent_dir):
    path = os.path.join(parent_dir, "tb_logs")
    return TensorBoardLogger(path, name=config["model"]["name"])


def configure_trainer(config, logger):
    brats_dataset = True
    if "brats" not in config["dataset"]["name"].split("_")[0]:
        brats_dataset = False
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_total_dice",
    #     dirpath=logger.log_dir,
    #     filename=f"{config['model']['name']}-{{epoch:02d}}-{{val_total_dice:.6f}}",
    #     save_top_k=3,
    #     mode="max",
    #     save_last=True,
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss" if brats_dataset else None,
        dirpath=logger.log_dir,
        filename=f"{config['model']['name']}-{{epoch:02d}}-{{val_loss:.6f}}",
        save_top_k=3 if brats_dataset else 1,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    check_val_every_n_epoch = config.get("check_val_every_n_epoch", 1)
    check_val_every_n_epoch = (
        check_val_every_n_epoch if brats_dataset else config["training"]["epochs"]
    )
    callbacks = [checkpoint_callback, lr_monitor]
    return Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config["training"]["epochs"],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu",
        devices=1,
    )


def get_module(config):
    if "brats" in config["dataset"]["name"].split("_")[0]:
        return brats_module
    else:
        return la_heart_module


def main():
    ######################## parse the arguments ########################
    args = parse_arguments()
    
    ######################## load the configuration file ########################
    set_seed()
    BASE_DIR = get_base_directory()
    CONFIG_NAME = (
        f"{args.config.split('_')[0]}/{args.config.split('_')[1]}/{args.config}.yaml"
    )    
    CONFIG_FILE_PATH = os.path.join(BASE_DIR, "configs", CONFIG_NAME)

    config = load_config(CONFIG_FILE_PATH)
    
    if "brats" not in config["dataset"]["name"].split("_")[0]:
        if args.fold == -1:
            raise ValueError(
                "Fold must be determined for pancreas and la_heart dataset!"
            )
        config["fold"] = args.fold
        
    print_config(config)

    tr_loader, vl_loader, te_loader = get_dataloaders(config, ["tr", "vl", "te"])
    
    ######################### get the model and configure the trainer ########################
    network = get_model(config)
    lightning_module = get_module(config)

    logger = configure_logger(config, BASE_DIR)
    trainer = configure_trainer(config, logger)

    just_test = config["checkpoints"]["test_mode"]
    auto_lr_find = config.get("find_lr", False)
    ckpt_path = None
    if config["checkpoints"]["continue_training"]:
        ckpt_path = config["checkpoints"]["ckpt_path"]

    if not just_test:
        model = lightning_module(config, model=network)

        if auto_lr_find:
            Tuner(trainer).lr_find(
                model, train_dataloaders=tr_loader, val_dataloaders=vl_loader
            )
            config["training"]["optimizer"]["params"]["lr"] = model.lr

            # lr_find_files = glob.glob(".lr_find*")
            # if lr_find_files:
            #     os.remove(lr_find_files[0])
            with open(os.path.join(logger.log_dir, "hpram.yaml"), "w") as yaml_file:
                yaml.dump(config, yaml_file)
            if ckpt_path:
                if not os.path.exists(ckpt_path):
                    raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
                print(f"Try to resume from {ckpt_path}")

            trainer.fit(
                model,
                train_dataloaders=tr_loader,
                val_dataloaders=vl_loader,
                ckpt_path=ckpt_path,
            )
        else:
            os.makedirs(logger.log_dir, exist_ok=True)
            with open(os.path.join(logger.log_dir, "hpram.yaml"), "w") as yaml_file:
                yaml.dump(config, yaml_file)
            trainer.fit(
                model,
                train_dataloaders=tr_loader,
                val_dataloaders=vl_loader,
                ckpt_path=ckpt_path,
            )

        print(f"testing {CONFIG_NAME}")
        trainer.test(
            model, dataloaders=te_loader, ckpt_path="best"
        )  # uses the best model to do the test

    else:
        ckpt_path = config["checkpoints"]["ckpt_path"]
        if not os.path.exists(ckpt_path):
            raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
        print(f"Try to test from {ckpt_path}")
        try:
            model = lightning_module.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                map_location="cpu",
                config=config,
                model=network,
            )
        except:
            checkpoint = torch.load(ckpt_path)
            network.load_state_dict(checkpoint["MODEL_STATE"], strict=False)
            model = lightning_module(config, model=network)

        print("loaded the checkpoint.")
        trainer.test(model, dataloaders=te_loader)


if __name__ == "__main__":
    main()
