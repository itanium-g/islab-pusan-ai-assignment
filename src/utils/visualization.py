"""
Visualization Utilities.
Functions for drawing bounding boxes, creating side-by-side ground-truth vs prediction grids,
and generating evaluation curves (Precision-Recall, F1-Confidence, Confusion Matrix).
"""

import os
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Palette for classes
CLASS_COLORS = {
    0: (0, 255, 0),     # Green for Drone
    1: (255, 0, 0),     # Red
    2: (0, 0, 255),     # Blue
    3: (255, 255, 0),   # Yellow
}

def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray = None,
    scores: np.ndarray = None,
    class_names: List[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Draw bounding boxes on an RGB numpy image or PIL Image.
    boxes: (N, 4) in [x1, y1, x2, y2] format, or (N, 6) in [x1, y1, x2, y2, score, cls].
    """
    line_thickness = kwargs.get("thickness", line_thickness)
    is_pil_input = isinstance(image, Image.Image)
    if not is_pil_input:
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image.copy()
        
    draw = ImageDraw.Draw(pil_img)
    
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = map(int, box[:4])
        
        if len(box) >= 6:
            score_val = float(box[4])
            cls_id = int(box[5])
        elif len(box) == 5:
            score_val = float(box[4])
            cls_id = int(labels[i]) if labels is not None else 0
        else:
            score_val = float(scores[i]) if (scores is not None and len(scores) > i) else None
            cls_id = int(labels[i]) if labels is not None else 0
            
        box_color = CLASS_COLORS.get(cls_id, color)
        
        # Draw rectangle
        for t in range(line_thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=box_color)
            
        # Draw label & score tag
        label_text = class_names[cls_id] if (class_names and cls_id < len(class_names)) else f"class_{cls_id}"
        if score_val is not None:
            label_text += f" {score_val:.2f}"
            
        # Background tag
        text_bbox = draw.textbbox((x1, max(0, y1 - 12)), label_text)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x1 + 1, max(0, y1 - 12)), label_text, fill=(0, 0, 0))
        
    if is_pil_input:
        return pil_img
    return np.array(pil_img)

def plot_batch_predictions(
    images: torch.Tensor,
    targets: List[torch.Tensor],
    predictions: List[torch.Tensor],
    class_names: List[str] = None,
    max_images: int = 4,
    save_path: str = None
) -> np.ndarray:
    """
    Generate side-by-side Ground Truth vs Prediction visual comparison grid for a batch.
    images: (B, 3, H, W) normalized tensor
    targets: List of (N_i, 5) [cls, cx, cy, w, h] normalized
    predictions: List of (M_i, 6) [x1, y1, x2, y2, score, cls] in pixel coordinates
    """
    B, _, H, W = images.shape
    num_display = min(B, max_images)
    
    fig, axes = plt.subplots(num_display, 2, figsize=(12, 4 * num_display))
    if num_display == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(num_display):
        # Convert image tensor to numpy RGB
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # 1. Ground truth
        gt_boxes = []
        gt_labels = []
        if i < len(targets) and targets[i] is not None and len(targets[i]) > 0:
            tgt = targets[i].cpu().numpy()
            for row in tgt:
                cls_id = int(row[0])
                cx, cy, w, h = row[1:5]
                # convert normalized cxcywh to pixel xyxy
                x1 = (cx - 0.5 * w) * W
                y1 = (cy - 0.5 * h) * H
                x2 = (cx + 0.5 * w) * W
                y2 = (cy + 0.5 * h) * H
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(cls_id)
                
        gt_img = draw_bounding_boxes(
            img_np.copy(),
            np.array(gt_boxes) if gt_boxes else np.zeros((0, 4)),
            labels=np.array(gt_labels) if gt_labels else None,
            class_names=class_names,
            color=(0, 255, 0)
        )
        
        # 2. Prediction
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        if i < len(predictions) and predictions[i] is not None and len(predictions[i]) > 0:
            pred = predictions[i].cpu().numpy()
            pred_boxes = pred[:, :4]
            pred_scores = pred[:, 4]
            pred_labels = pred[:, 5].astype(int)
            
        pred_img = draw_bounding_boxes(
            img_np.copy(),
            np.array(pred_boxes) if len(pred_boxes) > 0 else np.zeros((0, 4)),
            labels=np.array(pred_labels) if len(pred_labels) > 0 else None,
            scores=np.array(pred_scores) if len(pred_scores) > 0 else None,
            class_names=class_names,
            color=(255, 0, 0)
        )
        
        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title(f"Sample {i+1}: Ground Truth (Green)")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(pred_img)
        axes[i, 1].set_title(f"Sample {i+1}: Model Prediction (Red)")
        axes[i, 1].axis("off")
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        
    plt.close(fig)
    return fig

def plot_pr_curve(recalls: np.ndarray = None, precisions: np.ndarray = None, ap: float = 0.0, save_path: str = None, **kwargs):
    """
    Plot Precision-Recall curve with robust fallbacks.
    """
    if recalls is None:
        recalls = kwargs.get("recall", np.array([0.0, 1.0]))
    if precisions is None:
        precisions = kwargs.get("precision", np.array([1.0, 0.0]))
    if "ap" in kwargs:
        ap = kwargs["ap"]
        
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    if len(recalls) == 0:
        recalls = np.array([0.0, 1.0])
    if len(precisions) == 0:
        precisions = np.array([0.0, 0.0])
        
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color="blue", lw=2, label=f"Drone Precision-Recall (AP@0.5 = {ap*100:.2f}%)")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
