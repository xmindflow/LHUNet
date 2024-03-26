from monai.networks.nets import SwinUNETR, UNETR, SegResNetVAE
import torch



def get_lhunet_model(config):
    from models.dim3.lhunet.models.v7 import LHUNet as model

    return model(**config["model"]["params"])



MODEL_FACTORY = {
    # "unet": get_unet,
    # "transunet": get_transunet,
    # "missformer": get_missformer,
    # "multiresunet": get_multiresunet,
    # "resunet": get_resunet,
    # "uctransnet": get_uctransnet,
    # "attunet": get_attunet,
    # "unet3d": get_unet3d,
    # "resunet3d": get_resunet3d,
    # "resunetse3d": get_resunetse3d,
    # "mainmodel": get_main_model,
    # "mainmodel-bridge": get_main_bridge_model,
    # "transunet3d": get_transunet3d,
    # "swinunetr": get_swinunetr,
    # "swinunetr3d": get_swinunetr,
    # "swinunetr3d-v2": get_swinunetr,
    # "unetr3d": get_unetr,
    # "segresnetvae3d": get_segresnetvae,
    # "nnformer3d": get_nnformer,
    # "unetrpp3d": get_unetrpp,
    # "dlk-former": d_lka_net_synapse,
    # "vnet": get_vnet,
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
