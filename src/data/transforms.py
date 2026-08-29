"""
Data Transforms & Augmentations for Object Detection.
Includes letterbox resizing, bounding box coordinate transforms,
photometric distortions (simulating fog/sun conditions), and tensor conversion.
"""

import random
from typing import Tuple, List
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

def letterbox_image_and_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    target_size: Tuple[int, int] = (640, 640)
) -> Tuple[Image.Image, np.ndarray]:
    """
    Resize image to target_size with neutral grey padding (114),
    and adjust normalized [cls, cx, cy, w, h] bounding box coordinates.
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded_image = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded_image.paste(resized_image, (pad_x, pad_y))
    
    if boxes is None or len(boxes) == 0:
        return padded_image, np.zeros((0, 5), dtype=np.float32)
        
    adjusted_boxes = boxes.copy().astype(np.float32)
    # boxes format: [cls, cx, cy, w, h] (normalized 0..1)
    cls_ids = adjusted_boxes[:, 0]
    cx_orig = adjusted_boxes[:, 1] * orig_w
    cy_orig = adjusted_boxes[:, 2] * orig_h
    w_orig = adjusted_boxes[:, 3] * orig_w
    h_orig = adjusted_boxes[:, 4] * orig_h
    
    cx_new = cx_orig * scale + pad_x
    cy_new = cy_orig * scale + pad_y
    w_new = w_orig * scale
    h_new = h_orig * scale
    
    norm_cx = np.clip(cx_new / target_w, 0.0, 1.0)
    norm_cy = np.clip(cy_new / target_h, 0.0, 1.0)
    norm_w = np.clip(w_new / target_w, 0.0, 1.0)
    norm_h = np.clip(h_new / target_h, 0.0, 1.0)
    
    adjusted_boxes = np.stack([cls_ids, norm_cx, norm_cy, norm_w, norm_h], axis=1)
    return padded_image, adjusted_boxes

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, boxes: np.ndarray):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, boxes: np.ndarray):
        if random.random() < self.p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                # Invert cx: cx_new = 1.0 - cx
                boxes = boxes.copy()
                boxes[:, 1] = 1.0 - boxes[:, 1]
        return image, boxes

class PhotometricDistortion:
    """
    Simulate solar glare, fog, brightness, contrast, and saturation variations.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, boxes: np.ndarray):
        if random.random() < self.p:
            # 1. Brightness
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)
            
            # 2. Contrast
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)
            
            # 3. Color (Saturation)
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Color(image).enhance(factor)
            
            # 4. Blur (Atmospheric haze simulation)
            if random.random() < 0.2:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
                
        return image, boxes

class ToTensor:
    def __call__(self, image: Image.Image, boxes: np.ndarray):
        # Convert PIL image to Float Tensor (C, H, W) normalized to [0, 1]
        img_np = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        
        boxes_tensor = torch.from_numpy(boxes).float() if (boxes is not None and len(boxes) > 0) else torch.zeros((0, 5), dtype=torch.float32)
        return img_tensor, boxes_tensor

def get_train_transforms(target_size: Tuple[int, int] = (640, 640)):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        PhotometricDistortion(p=0.6),
        ToTensor()
    ])

def get_val_transforms(target_size: Tuple[int, int] = (640, 640)):
    return Compose([
        ToTensor()
    ])
