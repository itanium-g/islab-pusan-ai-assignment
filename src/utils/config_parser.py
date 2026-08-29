"""
YAML Configuration Parser & Validator.
Handles hierarchical configuration merging, validation, and CLI overrides.
"""

import os
import yaml
from copy import deepcopy

DEFAULT_CONFIG = {
    "experiment": {
        "name": "drone_detection_experiment",
        "output_dir": "runs/train",
        "seed": 42
    },
    "dataset": {
        "train_manifest": "data/splits/train.txt",
        "val_manifest": "data/splits/val.txt",
        "test_manifest": "data/splits/test.txt",
        "img_size": 640,
        "num_classes": 1,
        "class_names": ["drone"],
        "use_cached": True,
        "cached_dir": "data/cached_640"
    },
    "model": {
        "name": "DroneNet-FPN-Attention",
        "type": "fpn_attn",
        "in_channels": 3,
        "num_classes": 1,
        "base_channels": 32,
        "strides": [4, 8, 16]
    },
    "training": {
        "epochs": 40,
        "batch_size": 16,
        "num_workers": 4,
        "learning_rate": 0.001,
        "min_learning_rate": 0.00001,
        "warmup_epochs": 3,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "amp": True,
        "grad_accum_steps": 1,
        "loss_weights": {
            "obj": 1.0,
            "box": 2.5,
            "cls": 0.5
        }
    },
    "logging": {
        "wandb": False,
        "wandb_project": "drone-detection-islab",
        "wandb_entity": None,
        "tensorboard": True,
        "log_interval": 10,
        "eval_interval": 1
    }
}

def merge_dicts(base: dict, update: dict) -> dict:
    """
    Recursively merge update dict into base dict.
    """
    merged = deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged

def load_config(config_path: str, overrides: dict = None) -> dict:
    """
    Load YAML configuration file and merge with defaults and optional overrides.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
        
    merged_cfg = merge_dicts(DEFAULT_CONFIG, user_cfg)
    
    if overrides:
        merged_cfg = merge_dicts(merged_cfg, overrides)
        
    return merged_cfg

def save_config(config: dict, output_path: str):
    """
    Save configuration dict to YAML file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
