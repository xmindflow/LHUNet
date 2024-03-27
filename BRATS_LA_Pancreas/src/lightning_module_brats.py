import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import get_binary_metrics
import numpy as np
import nibabel as nib
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer
from optimizers import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Tuple, Dict
from fvcore.nn import FlopCountAnalysis
from typing import Tuple


class SemanticSegmentation3D(pl.LightningModule):
    def __init__(self, config: dict, model=None):
        super(SemanticSegmentation3D, self).__init__()
        ############### initilizing the model and the config ################
        self.config = config
        self.model = model.cuda()
        ############### initilizing the loss function ################
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
        ).to(self.device)

        ############### initilizing the metrics ################
        modes = ["tr", "vl", "te"]
        self.modes_dict = {"tr": "train", "vl": "val", "te": "test"}

        self.types = ["wt", "tc", "et"]
        self.metrics = {}
        for mode in modes:
            self.metrics[mode] = {}
            for type_ in self.types:
                metric_name = f"metrics_{self.modes_dict[mode]}_{type_}"
                self.metrics[mode][type_] = (
                    get_binary_metrics(mode=mode)
                    .clone(prefix=f"{metric_name}/")
                    .to(self.device)
                )

        ################# calculation of number of params and FLOPS ################
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        # self.log("n_parameters_M", round(n_parameters * 1e-6, 2))
        size = config["dataset"]["input_size"]
        input_res = (4, size[2], size[0], size[1])
        input = torch.ones(()).new_empty(
            (1, *input_res),
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )

        flops = FlopCountAnalysis(self.model, input)
        total_flops = flops.total()
        print(f"MAdds: {round(total_flops * 1e-9, 2)} G")

        self.lr = self.config["training"]["optimizer"]["params"]["lr"]
        self.save_nifty = config["checkpoints"].get("save_nifty", True)
        self.test_batch_size = config["data_loader"]["test"]["batch_size"]
        if self.test_batch_size != config["data_loader"]["validation"]["batch_size"]:
            raise ValueError("Test batch size must be equal to validation batch size")

        self.use_sliding_window = config.get("use_sliding_window", False)
        if self.use_sliding_window:
            roi_size = config["dataset"]["input_size"]
            # need to chaneg the location to channel first
            roi_size = [roi_size[2], roi_size[0], roi_size[1]]
            self.slider = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=config["data_loader"]["train"]["batch_size"],
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
        imgs = batch["volume"].float()
        msks = batch["seg-volume"].type(torch.uint8)
        return imgs, msks

    def on_epoch_end(self, stage: str):
        total_dice = []
        for type_ in self.types:
            metric = self.metrics[stage][type_].compute()
            self.log_dict({f"{k}": v for k, v in metric.items()})
            total_dice.append(metric[f"metrics_{self.modes_dict[stage]}_{type_}/Dice"])
            self.metrics[stage][type_].reset()
        if stage == "vl":
            self.log("val_total_dice", sum(total_dice) / len(total_dice))

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

        self._update_metrics(preds, gts, stage)
        self._log_losses(loss, stage)

        if stage == "te":
            self._save_nifty(batch, imgs, preds, gts, batch_idx)

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
        loss = self.criterion_dice_ce(preds.float(), gts.float())
        return loss

    def _log_losses(self, losses: torch.Tensor, stage: str) -> None:
        self.log(
            f"{self.modes_dict[stage]}_loss",
            losses,
            on_epoch=True,
            prog_bar=True,
        )

    def _update_metrics(
        self, preds: torch.Tensor, gts: torch.Tensor, stage: str
    ) -> None:
        metrics = self.metrics[stage]
        preds_list, gts_list = self._contstruct_wt_tc_et(preds, gts)
        for index, type_ in enumerate(self.types):
            pred = preds_list[index].float()
            gt = gts_list[index].type(torch.uint8)
            metrics[type_].to(self.device)
            metrics[type_].update(pred, gt)

    def _save_nifty(self, batch, imgs, preds, gts, batch_idx):
        save_dir = self.logger.log_dir
        if self.save_nifty:
            for patient_idx in range(imgs.shape[0]):
                self.save_nifty_from_logits_preds(
                    batch["patient_name"][patient_idx],
                    batch["affinity"][patient_idx],
                    preds[patient_idx],
                    gts[patient_idx],
                    save_dir,
                )

    def save_nifty_from_logits_preds(
        self,
        name: str,
        affinity: torch.Tensor,
        preds: torch.Tensor,
        masks: torch.Tensor,
        save_dir: str,
    ) -> None:
        """
        inputs are 4D tensors (channel,depth,height,width)
        """
        affinity = affinity.detach().cpu().numpy()
        # not necessary to permute the preds and masks
        preds = preds.permute(0, 2, 3, 1)
        masks = masks.permute(0, 2, 3, 1).squeeze().type(torch.uint8)
        preds_labels = (
            torch.argmax(F.softmax(preds, dim=0), dim=0).squeeze().type(torch.uint8)
        )
        mask_img = nib.Nifti1Image(masks.detach().cpu().numpy(), affine=affinity)
        preds_img = nib.Nifti1Image(
            preds_labels.detach().cpu().numpy(), affine=affinity
        )
        name_dir = os.path.join(save_dir, "nifty predictions", name)
        os.makedirs(name_dir, exist_ok=True)
        nib.save(mask_img, os.path.join(name_dir, f"{name}-seg.nii.gz"))
        nib.save(preds_img, os.path.join(name_dir, f"{name}-pred.nii.gz"))

    def _contstruct_wt_tc_et(
        self, preds: torch.Tensor, gts: torch.Tensor
    ) -> Tuple[list, list]:
        preds_list = []
        gts_list = []
        preds_labels = torch.argmax(F.softmax(preds, dim=1), dim=1).unsqueeze(1)
        for type in ["wt", "tc", "et"]:
            constructor = getattr(self, f"_construct_{type}")
            preds_list.append(constructor(preds_labels))
            gts_list.append(constructor(gts))
        return preds_list, gts_list

    def _construct_et(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor == 3).type(torch.uint8)

    def _construct_tc(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor == 3) | (tensor == 1)).type(torch.uint8)

    def _construct_wt(self, tensor: torch.Tensor) -> torch.Tensor:
        return (((tensor == 3) | (tensor == 1)) | (tensor == 2)).type(torch.uint8)

    def _forward_pass(self, imgs: torch.Tensor, stage: str) -> torch.Tensor | list:
        if self.use_sliding_window:
            if stage == "tr":
                preds = self(imgs)
            else:
                preds = self.slider(imgs, self.model)
        else:
            preds = self(imgs)
        return preds
