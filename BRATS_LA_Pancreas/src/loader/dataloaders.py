from .brats2018 import brats2018_loader
from .brats2021 import brats2021_loader
from .la_heart import la_heart_loader


def get_dataloaders(config, mode_or_modes):
    ds_name = config.get("dataset", {}).get("name")

    if not ds_name:
        raise ValueError("You must determine dataset name!")

    get_ds_loader = globals().get(f"{ds_name.lower()}")

    if not get_ds_loader:
        raise ValueError(f"<{ds_name.lower()}> loader not implemented yet!")

    _data = get_ds_loader(config, verbose=True)

    if isinstance(mode_or_modes, list):
        return [_data[mode.lower()]["loader"] for mode in mode_or_modes]

    if isinstance(mode_or_modes, str):
        return _data[mode_or_modes.lower()]["loader"]

    raise ValueError("<mode_or_modes> must be either a string or a list of strings!")
