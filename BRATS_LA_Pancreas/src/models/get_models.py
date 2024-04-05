from monai.networks.nets import SwinUNETR, UNETR, SegResNetVAE
import torch


def get_lhunet_model(config):
    from models.dim3.lhunet.models.v7 import LHUNet as model

    return model(**config["model"]["params"])


MODEL_FACTORY = {
    "lhunet": get_lhunet_model,
}


def get_model(config):
    model_name = (
        config["model"]["name"].lower().split("_")[0]
    )  # Get the base name (e.g., unet from unet_variant1)
    if model_name in MODEL_FACTORY:
        return MODEL_FACTORY[model_name](config)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
