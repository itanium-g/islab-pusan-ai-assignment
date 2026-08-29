"""
Unified Experiment Logger.
Supports Console formatting, TensorBoard logging, and Weights & Biases (W&B) tracking.
"""

import os
import sys
import logging
from typing import Dict, Any
import numpy as np
import torch

class ExperimentLogger:
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_cfg: dict = None,
        wandb_project: str = None,
        wandb_name: str = None,
        config: dict = None,
        rank: int = 0,
        **kwargs
    ):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.rank = rank
        self.writer = None
        self.wandb_run = None
        
        # Build wandb config dict if individual args passed
        if wandb_cfg is None:
            wandb_cfg = {}
        if wandb_project:
            wandb_cfg["project"] = wandb_project
        if wandb_name:
            wandb_cfg["name"] = wandb_name
        if config:
            wandb_cfg["config"] = config
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup standard Python logging
        self.logger = logging.getLogger("DroneDetector")
        self.logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="a")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler (only for main process)
        if rank == 0:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # TensorBoard
            if self.use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
                    self.info("TensorBoard logging initialized.")
                except ImportError:
                    self.warning("TensorBoard not found. TensorBoard logging disabled.")
                    self.use_tensorboard = False
                    
            # Weights & Biases
            if self.use_wandb and wandb_cfg:
                try:
                    import wandb
                    project = wandb_cfg.get("project", "drone-detection-islab")
                    entity = wandb_cfg.get("entity", None)
                    name = wandb_cfg.get("name", os.path.basename(log_dir))
                    self.wandb_run = wandb.init(
                        project=project,
                        entity=entity,
                        name=name,
                        config=wandb_cfg.get("full_config", {}),
                        reinit=True
                    )
                    self.info(f"W&B logging initialized. Dashboard URL: {self.wandb_run.url}")
                except Exception as e:
                    self.warning(f"Failed to initialize W&B: {e}. W&B logging disabled.")
                    self.use_wandb = False

    def info(self, msg: str):
        if self.rank == 0:
            self.logger.info(msg)

    def warning(self, msg: str):
        if self.rank == 0:
            self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """
        Log scalars to TensorBoard and W&B, gracefully ignoring non-scalar arrays/curves.
        """
        if self.rank != 0:
            return
            
        formatted_metrics = {}
        for k, v in metrics.items():
            scalar_val = None
            if isinstance(v, (int, float)):
                scalar_val = float(v)
            elif isinstance(v, (np.floating, np.integer)):
                scalar_val = float(v)
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                if v.ndim == 0 or (hasattr(v, "numel") and v.numel() == 1):
                    scalar_val = float(v.item())
                elif hasattr(v, "size") and v.size == 1:
                    scalar_val = float(v.flatten()[0])
            
            if scalar_val is None:
                continue  # Skip PR curves, matrices, lists, non-scalar arrays
                
            tag = f"{prefix}/{k}" if prefix else k
            formatted_metrics[tag] = scalar_val
            
            if self.writer:
                self.writer.add_scalar(tag, scalar_val, global_step=step)
                
        if self.wandb_run:
            try:
                import wandb
                wandb.log(formatted_metrics, step=step)
            except Exception:
                pass

    def log_image(self, tag: str, image_tensor, step: int):
        """
        Log image grid to TensorBoard / W&B.
        """
        if self.rank != 0:
            return
            
        if self.writer:
            self.writer.add_image(tag, image_tensor, global_step=step)
            
        if self.wandb_run:
            try:
                import wandb
                # If tensor is (C, H, W), convert to numpy
                if hasattr(image_tensor, "permute"):
                    np_img = image_tensor.permute(1, 2, 0).cpu().numpy()
                else:
                    np_img = image_tensor
                wandb.log({tag: wandb.Image(np_img)}, step=step)
            except Exception:
                pass

    def close(self):
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
