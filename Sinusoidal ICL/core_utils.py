import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import os

import wandb
import json

_COLAB = False


def colab_setup(mount_point='/content/drive'):
    """
    Prepares the Colab environment: mounts Google Drive if needed.

    Args:
        mount_point: Where to mount Google Drive (default: '/content/drive')
    """
    try:
        from google.colab import drive
        if not os.path.exists(mount_point):
            drive.mount(mount_point)
        print("[✅] Google Drive mounted.")
    except ImportError:
        print("[⚠️] Not running in Colab. Skipping Drive mount.")

def wandb_setup(project: str, name: str, config: dict = None, entity: str = None):
    """
    Initializes Weights & Biases for a run.
    
    Args:
        project: WandB project name
        name:    Run name
        config:  Dictionary of hyperparameters
        entity:  Your wandb username or team (optional)
    Returns:
        cfg: wandb.config (dot-accessible config object)
    """
    wandb.login()
    wandb.init(project=project, name=name, config=config, entity=entity)
    return wandb.config

def save_model(model: torch.nn.Module,
                         folder: str,
                         filename: str,
                         mount_point: str = '/content/drive'):
    """
    Saves model.state_dict() into Google Drive.

    Args:
        model:        your torch.nn.Module
        gdrive_folder: path inside "My Drive" (e.g. 'models/checkpoints')
        filename:     filename to save as (e.g. 'resnet50.pth')
        mount_point:  where Drive is mounted (Colab default '/content/drive')
    """
    # 1. Mount Drive if needed
    if _COLAB and not os.path.exists(mount_point):
        drive.mount(mount_point)

    # 2. Build the full path under "My Drive"
    full_folder = os.path.join(mount_point, 'My Drive', folder)
    os.makedirs(full_folder, exist_ok=True)

    # 3. Save the state dict
    target_path = os.path.join(full_folder, filename)
    torch.save(model.state_dict(), target_path)

    print(f"[✅] Saved state_dict to {target_path}")

def save_config(cfg: dict, base_dir: str = "sinusoidal_icl/checkpoints", mount_point: str = '/content/drive') -> None:
    """
    Saves the training config as a JSON file to:
        <base_dir>/<cfg['name']>/config.json

    Args:
        cfg (dict): The configuration dictionary to save.
        base_dir (str): The base directory under which the config is saved.
            Default is "sinusoidal_icl/checkpoints".
    """
    if _COLAB and not os.path.exists(mount_point):
        drive.mount(mount_point)

    target_dir = os.path.join(mount_point, 'My Drive', base_dir)
    os.makedirs(target_dir, exist_ok=True)

    config_path = os.path.join(target_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[💾] Saved config to {config_path}")