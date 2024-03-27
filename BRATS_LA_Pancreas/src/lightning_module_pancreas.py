import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import nibabel as nib
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer
from optimizers import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Tuple, Dict
from medpy import metric
from fvcore.nn import FlopCountAnalysis
from typing import Tuple


class SemanticSegmentation3D(pl.LightningModule):
    def __init__(self, config: dict, model=None):
        super(SemanticSegmentation3D, self).__init__()
        self.config = config
        self.model = model.cuda()
        lambda_dice = config["training"]["criterion"]["params"].get("dice_weight", 0.5)
        lambda_ce = config["training"]["criterion"]["params"].get("bce_weight", 0.5)
        self.global_loss_weight = config["training"]["criterion"]["params"].get(
            "global_weight", 1.0
        )
        self.criterion_dice_ce = DiceCELoss(
            sigmoid=False,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            to_onehot_y=True,
            include_background=False, 
        ).to(self.device)

        ## Initialize metrics for each type and mode
        modes = ["tr", "vl", "te"]
        self.modes_dict = {"tr": "train", "vl": "val", "te": "test"}
        self.metric_types = ["Dice", "Jaccard", "HD95", "ASD"]

        self.metrics = {}
        for mode in modes:
            self.metrics[mode] = {}
            for type_ in self.metric_types:
                self.metrics[mode][type_] = []

        # calculate the params and FLOPs
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        # self.log("n_parameters_M", round(n_parameters * 1e-6, 2))

        input_res = (1, 96, 96, 96)
        input = torch.ones(()).new_empty(
            (1, *input_res),
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )

        flops = FlopCountAnalysis(self.model, input)
        total_flops = flops.total()
        print(f"MAdds: {round(total_flops * 1e-9, 2)} G")
        # self.log("GFLOPS", round(total_flops * 1e-9, 2))

        self.lr = self.config["training"]["optimizer"]["params"]["lr"]
        self.log_pictures = config["checkpoints"]["log_pictures"]
        self.save_nifty = config["checkpoints"].get("save_nifty", True)
        self.test_batch_size = config["data_loader"]["test"]["batch_size"]
        if self.test_batch_size != config["data_loader"]["validation"]["batch_size"]:
            raise ValueError("Test batch size must be equal to validation batch size")

        self.use_sliding_window = config.get("use_sliding_window", False)
        if self.use_sliding_window:
            roi_size = config["dataset"]["crop_size"]
            self.slider = SlidingWindowInferer(
                roi_size=roi_size,
                # sw_batch_size=config["data_loader"]["train"]["batch_size"],
                sw_batch_size=20,
                **config["sliding_window_params"],
            )
        else:
            print(
                "NOT USING SLIDING WINDOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        self.model_name = config["model"]["name"].split("_")[0]
        self.deep_supervision = False
        if hasattr(self.model, "do_ds"):
            self.deep_supervision = self.model.do_ds
        self.save_hyperparameters(config)

    def forward(self, x):
        return self.model(x)

    def _extract_data(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = batch["image"].float()
        msks = batch["label"].type(torch.uint8)
        return imgs, msks

    def on_epoch_end(self, stage: str):
        if stage == "tr":
            return
        for type_ in self.metric_types:
            metric = np.mean(self.metrics[stage][type_])
            # metric = torch.stack(self.metrics[stage][type_], dim=0).mean()
            self.log(f"{self.modes_dict[stage]}_{type_}", metric)
            self.metrics[stage][type_] = []

    def on_train_epoch_start(self) -> None:
        if self.use_sliding_window:
            if self.deep_supervision:
                self.model.do_ds = True

    def on_validation_epoch_start(self) -> None:
        if self.use_sliding_window:
            if self.deep_supervision:
                self.model.do_ds = False

    def on_test_epoch_start(self) -> None:
        if self.use_sliding_window:
            if self.deep_supervision:
                self.model.do_ds = False

    def _shared_step(self, batch, stage: str, batch_idx=None) -> torch.Tensor:
        imgs, gts = self._extract_data(batch)
        preds = self._forward_pass(imgs, stage)

        loss, preds = self._calculate_losses(preds, gts)

        # self._update_metrics(preds, gts, stage)
        self._log_losses(loss, stage)

        if stage == "te" or stage == "vl":
            self._cal_metrics(preds, gts, stage)
            # self._save_nifty_or_picture(batch, imgs, preds, gts, batch_idx)

        return loss

    # batch have values of size (batch ,modalities, channels, height, width)
    def training_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "tr", batch_idx)}

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("tr")

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._shared_step(batch, "vl", batch_idx)}

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("vl")

    def test_step(self, batch, batch_idx):
        return {"test_loss": self._shared_step(batch, "te", batch_idx)}

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end("te")

    def configure_optimizers(self):
        optimizer_cls = getattr(
            torch.optim, self.config["training"]["optimizer"]["name"]
        )
        del self.config["training"]["optimizer"]["params"]["lr"]
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.lr,
            **self.config["training"]["optimizer"]["params"],
        )
        scheduler_cls = globals().get(
            self.config["training"]["scheduler"]["name"], None
        )
        if scheduler_cls is None:
            scheduler = None
        else:
            scheduler = scheduler_cls(
                optimizer, **self.config["training"]["scheduler"]["params"]
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "lr_scheduler",
            },
        }

    def _calculate_losses(
        self,
        preds: Union[torch.Tensor, list, tuple],
        gts: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(preds, (list, tuple)):  # TODO: for the supervision case
            # just doing it for the SegResNetVAE case, where the output is a list of 2 elements
            weights = self._cal_loss_weights(preds)
            loss_dict = self._cal_loss_for_supervision(preds, gts, weights)
            preds = preds[0]
        else:
            loss_dict = self._cal_losses(preds, gts)

        return loss_dict, preds

    def _cal_losses(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
    ) -> dict:
        loss = self._cal_global_loss(preds, gts)
        return loss

    def _cal_loss_weights(self, preds: list | tuple) -> list:
        weights = np.array([1 / (2**i) for i in range(len(preds))])
        weights = weights / weights.sum()
        return weights

    def _cal_global_loss(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        loss = self.criterion_dice_ce(
            preds.float(), gts.float()
        )
        return loss

    def _log_losses(self, losses: torch.Tensor, stage: str) -> None:
        self.log(
            f"{self.modes_dict[stage]}_loss",
            losses,
            on_epoch=True,
            prog_bar=True,
        )

    def _cal_metrics(self, preds: torch.Tensor, gts: torch.Tensor, stage: str) -> None:
        # no need to permute the dimensions
        preds = preds.permute(0, 1, 3, 4, 2).cpu().numpy()
        preds = np.argmax(preds, axis=1)
        gts = gts.permute(0, 1, 3, 4, 2).squeeze(1).cpu().numpy()
        for i in range(preds.shape[0]):
            metrics = self._cal_metrics_per_subject(preds[i], gts[i], stage)
            for metric, type in zip(metrics, self.metric_types):
                self.metrics[stage][type].append(metric)

    def _cal_metrics_per_subject(self, pred, gt, stage: str) -> None:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        if np.count_nonzero(pred) == 0:
            hd = 0.0
            asd = 0.0
        else:
            hd = metric.binary.hd95(pred, gt)
            asd = metric.binary.asd(pred, gt)

        return [dice, jc, hd, asd]

    def _forward_pass(self, imgs: torch.Tensor, stage: str) -> torch.Tensor | list:
        if self.use_sliding_window:
            if stage == "tr":
                preds = self(imgs)
            else:
                preds = self.slider(imgs, self.model)
        else:
            preds = self(imgs)
        return preds
