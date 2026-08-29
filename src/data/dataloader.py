"""
DataLoader Factory & Batch Collator.
Handles variable numbers of bounding boxes per sample and supports Multi-GPU DistributedSampler.
"""

from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, DistributedSampler
from .dataset import DroneDataset

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    """
    Custom collate function to handle variable-length ground-truth boxes per image.
    batch: List of (image_tensor, boxes_tensor, metadata)
    Returns:
        images: (B, 3, H, W) Tensor
        targets: List of (N_i, 5) Tensors [cls, cx, cy, w, h]
        metadatas: List of metadata dicts
    """
    images = []
    targets = []
    metadatas = []
    
    for img, boxes, meta in batch:
        images.append(img)
        targets.append(boxes)
        metadatas.append(meta)
        
    images = torch.stack(images, dim=0)
    return images, targets, metadatas

def build_dataloader(
    dataset: DroneDataset,
    batch_size: int = 16,
    is_train: bool = True,
    shuffle: bool = None,
    num_workers: int = 4,
    distributed: bool = False,
    is_distributed: bool = None,
    pin_memory: bool = True,
    drop_last: bool = None
) -> DataLoader:
    """
    Create an optimized DataLoader instance with optional DDP DistributedSampler.
    Supports both is_train/shuffle and distributed/is_distributed argument aliases.
    """
    if is_distributed is not None:
        distributed = is_distributed
    if shuffle is None:
        shuffle = is_train
    if drop_last is None:
        drop_last = is_train
        
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        loader_shuffle = False
    else:
        loader_shuffle = shuffle
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=(num_workers > 0)
    )
    return loader
