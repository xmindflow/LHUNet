class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


import torch.nn.functional as F
import os


def _print(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{bcolors.ENDC}"

    if "bold" in p.lower():
        pre += bcolors.BOLD
    elif "underline" in p.lower():
        pre += bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += bcolors.HEADER

    if "warning" in p.lower():
        pre += bcolors.WARNING
    elif "error" in p.lower():
        pre += bcolors.FAIL
    elif "ok" in p.lower():
        pre += bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += bcolors.OKBLUE
        else:
            pre += bcolors.OKCYAN

    print(f"{pre}{string}{bcolors.ENDC}")


import yaml


def load_config(config_filepath):
    try:
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            expanded_config = expand_env_vars_in_data(config)
            return expanded_config
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)


def expand_env_vars_in_data(data):
    """
    Recursively walk through the data structure,
    expanding environment variables in string values.
    """
    if isinstance(data, dict):
        # For dictionaries, apply expansion to each value
        return {key: expand_env_vars_in_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        # For lists, apply expansion to each element
        return [expand_env_vars_in_data(element) for element in data]
    elif isinstance(data, str):
        # For strings, expand environment variables
        return os.path.expandvars(data)
    else:
        # For all other data types, return as is
        return data


import json


def print_config(config, logger=None):
    conf_str = json.dumps(config, indent=2)
    if logger:
        logger.info(f"\n{' Config '.join(2*[10*'>>',])}\n{conf_str}\n{28*'>>'}")
    else:
        _print("Config:", "info_underline")
        print(conf_str)
        print(30 * "~-", "\n")
