"""
Main Training Entrypoint for Drone Detection.
Supports Single-GPU, Multi-GPU DistributedDataParallel (DDP), Custom Config Overrides, and W&B logging.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config_parser import load_config
from src.utils.logger import ExperimentLogger
from src.data.dataset import DroneDataset
from src.data.dataloader import build_dataloader
from src.models.detector import build_detector
from src.engine.trainer import Trainer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_safe_device(rank: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        test_t = torch.zeros(1, device=f"cuda:{rank}") + 1
        _ = test_t.item()
        return torch.device(f"cuda:{rank}")
    except Exception as e:
        print(f"[WARNING] CUDA device cuda:{rank} ({torch.cuda.get_device_name(rank)}) is not compatible with this PyTorch build ({e}). Falling back to multi-threaded CPU.")
        torch.set_num_threads(os.cpu_count() or 4)
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Train Vanilla Drone Detection Model from Scratch.")
    parser.add_argument("--config", type=str, default="configs/model3_fpn_attn.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)")
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel multi-GPU training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for torchrun DDP")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases cloud logging")
    parser.add_argument("--wandb-project", type=str, default="drone-detection-islab", help="W&B project name")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 batch to verify forward/loss passes")
    
    args = parser.parse_args()
    
    # 1. Setup DDP Environment if requested
    rank = 0
    world_size = 1
    if args.ddp or "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        
    is_main_process = (rank == 0)
    
    # 2. Load Configuration
    config = load_config(args.config)
    
    # Apply CLI Overrides
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
        
    set_seed(config.get("seed", 42) + rank)
    
    # Determine Device
    if args.device:
        device = torch.device(args.device)
    elif args.ddp:
        device = torch.device(f"cuda:{rank}")
    else:
        device = get_safe_device(0)
        
    # 3. Setup Logger
    exp_name = config.get("experiment", {}).get("name", "drone_exp")
    output_dir = os.path.join(config.get("experiment", {}).get("output_dir", "runs/train"), exp_name)
    
    logger = ExperimentLogger(
        log_dir=output_dir,
        use_wandb=args.wandb and is_main_process,
        wandb_project=args.wandb_project,
        wandb_name=exp_name,
        config=config,
        rank=rank
    )
    
    if is_main_process:
        logger.info(f"=== Starting Drone Detection Pipeline ===")
        logger.info(f"Loaded config: {args.config}")
        logger.info(f"World Size: {world_size} | Local Rank: {rank} | Device: {device}")
        
    # 4. Build Datasets & DataLoaders
    img_size = config.get("dataset", {}).get("img_size", 640)
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("dataset", {}).get("num_workers", 4)
    use_cached = config.get("dataset", {}).get("use_cached", True)
    cached_dir = config.get("dataset", {}).get("cached_dir", "data/cached_640")
    
    train_manifest = config.get("dataset", {}).get("train_manifest", "data/splits/train.txt")
    val_manifest = config.get("dataset", {}).get("val_manifest", "data/splits/val.txt")
    
    # Fallback to dynamic split if manifests don't exist yet
    if not os.path.exists(train_manifest):
        if is_main_process:
            logger.info("Manifest files not found. Generating automatic sequence-aware split...")
            from scripts.split_dataset import create_splits
            create_splits(
                dataset_dir=config.get("dataset", {}).get("raw_dir", "curated_datasets/obj_det_base"),
                output_dir="data/splits"
            )
        if dist.is_initialized():
            dist.barrier()
            
    train_dataset = DroneDataset(
        manifest_path=train_manifest,
        img_size=img_size,
        is_train=True,
        use_cached=use_cached,
        cached_dir=cached_dir
    )
    val_dataset = DroneDataset(
        manifest_path=val_manifest,
        img_size=img_size,
        is_train=False,
        use_cached=use_cached,
        cached_dir=cached_dir
    )
    
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        is_distributed=(world_size > 1)
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        is_distributed=False
    )
    
    if is_main_process:
        logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        
    # 5. Build Model from Scratch
    model = build_detector(config.get("model", {}))
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if is_main_process:
        logger.info(f"Model: {model.name} | Total Parameters: {num_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
        
    if args.dry_run:
        if is_main_process:
            logger.info("Executing dry-run sanity check...")
            model.to(device)
            sample_imgs, sample_tgts, _ = next(iter(train_loader))
            sample_imgs = sample_imgs.to(device)
            loss_dict = model(sample_imgs, sample_tgts)
            logger.info(f"Dry-run forward pass successful! Initial loss: {loss_dict['loss'].item():.4f}")
            logger.info("Dry-run test complete. Exiting.")
        return
        
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        rank=rank,
        world_size=world_size,
        device=device
    )
    
    # 7. Start Training Loop
    trainer.train()
    
    # Clean up DDP
    if dist.is_initialized():
        dist.destroy_process_group()
        
    if is_main_process:
        logger.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
