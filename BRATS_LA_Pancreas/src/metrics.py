import torchmetrics
import torch
import numpy as np
import torch.nn.functional as F
from medpy.metric.binary import hd95
from monai.metrics import HausdorffDistanceMetric


device = "cuda" if torch.cuda.is_available() else "cpu"


class MONAIHausdorffDistance(torchmetrics.Metric):
    def __init__(self, include_background=False, percentile=95):
        super().__init__()
        self.include_background = include_background
        self.percentile = percentile
        self.monai_hd95 = HausdorffDistanceMetric(
            include_background=include_background, percentile=percentile
        )

    def update(self, y_pred, y_true):
        # y_preds and y can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]). (we are using batch first)
        y_pred = y_pred.permute(0, 1, 3, 4, 2)
        y_true = y_true.permute(0, 1, 3, 4, 2)
        # Convert tensors to one-hot format
        y_pred_binary = (F.sigmoid(y_pred) > 0.5).type(torch.int8)
        if y_pred.shape[1] == 1:
            y_pred_one_hot = torch.cat([~y_pred_binary, y_pred_binary], dim=1)
            y_true_one_hot = torch.cat([~y_true, y_true], dim=1)
        else:
            raise ValueError("y_pred must be a single channel tensor")
        # Compute the HD95 using MONAI metric
        self.monai_hd95(y_pred_one_hot, y_true_one_hot)

    def compute(self):
        # Return the computed HD95 value from MONAI
        return self.monai_hd95.aggregate()

    def reset(self):
        # Reset the MONAI metric
        self.monai_hd95.reset()


def get_binary_metrics(mode="tr", *args, **kwargs):
    if mode not in ["tr", "te", "vl"]:
        raise ValueError("mode must be in ['tr','te','vl']")
    if mode == "te":
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(task="binary"),
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.Dice(multiclass=False),
                torchmetrics.Precision(task="binary"),
                torchmetrics.Specificity(task="binary"),
                torchmetrics.Recall(task="binary"),
                # IoU
                torchmetrics.JaccardIndex(task="binary"),
                MONAIHausdorffDistance(),  # only use it for test set
            ],
            prefix="metrics/",
        )
    else:
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(task="binary"),
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.Dice(multiclass=False),
                torchmetrics.Precision(task="binary"),
                torchmetrics.Specificity(task="binary"),
                torchmetrics.Recall(task="binary"),
                # IoU
                torchmetrics.JaccardIndex(task="binary"),
            ],
            prefix="metrics/",
        )
    return metrics
