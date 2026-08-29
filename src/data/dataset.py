"""
PyTorch Dataset for Drone Object Detection.
Supports raw high-resolution images, pre-resized cached images,
in-memory caching, and YOLO annotation parsing with robust path resolution.
"""

import os
import glob
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import letterbox_image_and_boxes, get_train_transforms, get_val_transforms

class DroneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str = None,
        data_dir: str = None,
        img_size: int = 640,
        is_train: bool = True,
        use_cached: bool = True,
        cached_dir: str = "data/cached_640",
        cache_in_ram: bool = False,
        transforms = None
    ):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.is_train = is_train
        self.use_cached = use_cached
        self.cached_dir = cached_dir
        self.cache_in_ram = cache_in_ram
        self.transforms = transforms or (get_train_transforms(self.img_size) if is_train else get_val_transforms(self.img_size))
        
        self.samples = []
        self.ram_cache: Dict[int, Tuple[Image.Image, np.ndarray]] = {}
        
        # Load samples from manifest or directory
        if manifest_path and os.path.exists(manifest_path):
            manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
            with open(manifest_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "," in line:
                        img_p, lbl_p = line.split(",", 1)
                        img_p = self._resolve_path(img_p.strip(), manifest_dir)
                        lbl_p = self._resolve_path(lbl_p.strip(), manifest_dir)
                        if os.path.exists(img_p):
                            self.samples.append((img_p, lbl_p))
        elif data_dir and os.path.exists(data_dir):
            txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
            for tf in txt_files:
                base = os.path.splitext(tf)[0]
                img_p = f"{base}.png" if os.path.exists(f"{base}.png") else f"{base}.jpg"
                if os.path.exists(img_p):
                    self.samples.append((img_p, tf))
                    
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for Dataset with manifest={manifest_path}, data_dir={data_dir}")

    def _resolve_path(self, path: str, base_dir: str) -> str:
        """Resolve path from current working directory or relative to manifest base dir."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        
        # Try relative to manifest directory
        rel_to_manifest = os.path.join(base_dir, path)
        if os.path.exists(rel_to_manifest):
            return os.path.abspath(rel_to_manifest)
            
        # Try in curated_datasets/obj_det_base
        basename = os.path.basename(path)
        curated_path = os.path.join("curated_datasets", "obj_det_base", basename)
        if os.path.exists(curated_path):
            return os.path.abspath(curated_path)
            
        return path

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_sample(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        img_path, lbl_path = self.samples[index]
        
        # Check if preprocessed cached version is requested & available
        if self.use_cached and self.cached_dir and os.path.exists(self.cached_dir):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cached_img = os.path.join(self.cached_dir, "images", f"{base_name}.jpg")
            cached_lbl = os.path.join(self.cached_dir, "labels", f"{base_name}.txt")
            
            if os.path.exists(cached_img) and os.path.exists(cached_lbl):
                with Image.open(cached_img) as img:
                    image = img.convert("RGB")
                    
                boxes = []
                with open(cached_lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = float(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            boxes.append([cls_id, cx, cy, w, h])
                boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
                return image, boxes

        # Otherwise, load raw full-resolution image and letterbox on the fly
        with Image.open(img_path) as img:
            raw_image = img.convert("RGB")
            
        raw_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = float(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        raw_boxes.append([cls_id, cx, cy, w, h])
                        
        raw_boxes = np.array(raw_boxes, dtype=np.float32) if raw_boxes else np.zeros((0, 5), dtype=np.float32)
        padded_image, adjusted_boxes = letterbox_image_and_boxes(raw_image, raw_boxes, self.img_size)
        return padded_image, adjusted_boxes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.cache_in_ram and index in self.ram_cache:
            image, boxes = self.ram_cache[index]
            image = image.copy()
            boxes = boxes.copy()
        else:
            image, boxes = self._load_raw_sample(index)
            if self.cache_in_ram:
                self.ram_cache[index] = (image, boxes)
                
        # Apply dynamic augmentations
        img_tensor, boxes_tensor = self.transforms(image, boxes)
        
        metadata = {
            "index": index,
            "img_path": self.samples[index][0],
            "lbl_path": self.samples[index][1]
        }
        
        return img_tensor, boxes_tensor, metadata
