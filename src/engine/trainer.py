"""
OOP Training Engine for Drone Object Detection.
Supports Single-GPU, Multi-GPU DistributedDataParallel (DDP), Automatic Mixed Precision (AMP),
Cosine Annealing with Linear Warmup, Gradient Accumulation, and W&B/TensorBoard Logging.
"""

import os
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .evaluator import Evaluator
from src.utils.logger import ExperimentLogger
from src.utils.visualization import plot_batch_predictions

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: ExperimentLogger = None,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.device = device or (torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu"))
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup DDP wrapper if multi-GPU
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank] if self.device.type == "cuda" else None,
                output_device=rank if self.device.type == "cuda" else None,
                find_unused_parameters=True
            )
            
        t_cfg = config.get("training", {})
        self.epochs = t_cfg.get("epochs", 40)
        self.grad_accum_steps = t_cfg.get("grad_accum_steps", 1)
        self.use_amp = t_cfg.get("amp", True) and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Setup Optimizer
        lr = t_cfg.get("learning_rate", 0.001)
        weight_decay = t_cfg.get("weight_decay", 0.0005)
        opt_name = t_cfg.get("optimizer", "AdamW")
        
        if opt_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
        # Setup Learning Rate Scheduler (Warmup + Cosine Annealing)
        warmup_epochs = t_cfg.get("warmup_epochs", 3)
        min_lr = t_cfg.get("min_learning_rate", 1e-5)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, self.epochs - warmup_epochs))
                return max(min_lr / lr, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926)).item()))
                
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        # Logging & Output Directories
        self.output_dir = os.path.join(config.get("experiment", {}).get("output_dir", "runs/train"), config.get("experiment", {}).get("name", "exp"))
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        self.logger = logger or ExperimentLogger(self.output_dir, rank=rank)
        self.evaluator = Evaluator(
            model=self.model.module if hasattr(self.model, "module") else self.model,
            dataloader=self.val_loader,
            device=self.device
        )
        
        self.best_map50 = 0.0
        self.start_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
            
        total_loss = 0.0
        total_obj_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.is_main_process)
        self.optimizer.zero_grad()
        
        for step, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                loss_dict = self.model(images, targets)
                loss = loss_dict["loss"] / self.grad_accum_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss_dict["loss"].item()
            total_obj_loss += loss_dict["obj_loss"].item()
            total_box_loss += loss_dict["box_loss"].item()
            total_cls_loss += loss_dict["cls_loss"].item()
            
            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss_dict['loss'].item():.4f}",
                    "box": f"{loss_dict['box_loss'].item():.4f}",
                    "obj": f"{loss_dict['obj_loss'].item():.4f}",
                    "lr": f"{current_lr:.6f}"
                })
                
        self.scheduler.step()
        
        epoch_metrics = {
            "train_loss": total_loss / max(1, num_batches),
            "train_obj_loss": total_obj_loss / max(1, num_batches),
            "train_box_loss": total_box_loss / max(1, num_batches),
            "train_cls_loss": total_cls_loss / max(1, num_batches),
            "lr": self.optimizer.param_groups[0]["lr"]
        }
        return epoch_metrics

    def fit(self):
        self.logger.info(f"Starting training: {self.config.get('model', {}).get('name', 'Model')} for {self.epochs} epochs.")
        
        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(epoch)
            
            # Validation at each epoch
            if self.is_main_process:
                val_metrics = self.evaluator.evaluate()
                
                # Log metrics
                combined_metrics = {**train_metrics, **val_metrics}
                self.logger.log_metrics(combined_metrics, step=epoch + 1)
                
                self.logger.info(
                    f"[Epoch {epoch+1:02d}/{self.epochs:02d}] "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val AP@0.5: {val_metrics['ap50']*100:.2f}% | "
                    f"Val mAP@0.5:0.95: {val_metrics['map50_95']*100:.2f}% | "
                    f"Precision: {val_metrics['precision']*100:.2f}% | "
                    f"Recall: {val_metrics['recall']*100:.2f}% | "
                    f"FPS: {val_metrics['fps']:.1f}"
                )
                
                # Save checkpoints
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "ap50": val_metrics["ap50"],
                    "map50_95": val_metrics["map50_95"],
                    "config": self.config
                }
                
                # Save last checkpoint
                last_ckpt_path = os.path.join(self.checkpoint_dir, "last.pth")
                torch.save(checkpoint, last_ckpt_path)
                
                # Save best checkpoint based on AP@0.5
                if val_metrics["ap50"] > self.best_map50:
                    self.best_map50 = val_metrics["ap50"]
                    best_ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    torch.save(checkpoint, best_ckpt_path)
                    self.logger.info(f"==> New Best Model Saved with Val AP@0.5: {self.best_map50*100:.2f}%")
                    
        total_time_hours = (time.time() - start_time) / 3600.0
        self.logger.info(f"Training completed in {total_time_hours:.2f} hours. Best Val AP@0.5: {self.best_map50*100:.2f}%")
        self.logger.close()

    def train(self):
        """
        Alias for fit() method.
        """
        return self.fit()
