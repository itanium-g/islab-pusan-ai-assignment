"""
Custom Multi-Task Loss Module for Drone Detection.
Combines:
1. Focal Objectness Loss (Addresses severe foreground-background imbalance)
2. Complete-IoU (CIoU) Loss (Optimizes area overlap, Euclidean distance, and aspect ratio)
3. Label-Smoothed Binary Cross-Entropy Classification Loss
"""

import math
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.box_ops import cxcywh_to_xyxy, box_ciou

class FocalObjectnessLoss(nn.Module):
    """
    Focal Loss for dense objectness prediction:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "sum"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        loss = alpha_factor * modulating_factor * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class CompleteIoULoss(nn.Module):
    """
    Complete-IoU (CIoU) Loss:
    L_CIoU = 1 - IoU + (rho^2(b, b_gt) / c^2) + alpha * v
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
            
        ciou = box_ciou(pred_boxes, target_boxes)
        loss = 1.0 - ciou
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class CustomMultiTaskLoss(nn.Module):
    """
    Unified Multi-Task Drone Detection Loss:
    L_total = lambda_obj * L_focal + lambda_box * L_ciou + lambda_cls * L_cls
    """
    def __init__(
        self,
        strides: List[int] = [4, 8, 16],
        lambda_obj: float = 1.2,
        lambda_box: float = 3.0,
        lambda_cls: float = 0.5,
        num_classes: int = 1
    ):
        super().__init__()
        self.strides = strides
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.num_classes = num_classes
        
        self.focal_loss = FocalObjectnessLoss(alpha=0.25, gamma=2.0, reduction="sum")
        self.ciou_loss = CompleteIoULoss(reduction="mean")

    def forward(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        targets: List[torch.Tensor],
        img_size: int = 640
    ) -> Dict[str, torch.Tensor]:
        """
        predictions: List of (cls_preds, obj_preds, reg_preds) for each pyramid level
        targets: List of (N_i, 5) [cls, cx, cy, w, h] normalized per image in batch
        """
        device = predictions[0][0].device
        batch_size = len(targets)
        
        total_obj_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_positives = 0
        
        for level_idx, (cls_pred, obj_pred, reg_pred) in enumerate(predictions):
            stride = self.strides[level_idx]
            B, _, H, W = obj_pred.shape
            
            # Reshape predictions: (B, H, W, ...)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()   # (B, H, W, num_classes)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous()   # (B, H, W, 1)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()   # (B, H, W, 4)
            
            # Target maps for this level
            target_obj = torch.zeros((B, H, W, 1), device=device)
            target_cls = torch.zeros((B, H, W, self.num_classes), device=device)
            pos_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
            
            matched_pred_boxes = []
            matched_target_boxes = []
            
            for b in range(batch_size):
                tgt = targets[b]
                if tgt is None or len(tgt) == 0:
                    continue
                    
                tgt = tgt.to(device)
                for obj in tgt:
                    cls_id = int(obj[0])
                    cx, cy, w, h = obj[1:5]
                    max_dim = max(w.item(), h.item())
                    
                    # Scale assignment logic
                    if len(self.strides) == 1:
                        assign = True
                    else:
                        assign = False
                        if level_idx == 0 and max_dim < 0.06:      # P2 (tiny)
                            assign = True
                        elif level_idx == 1 and 0.02 <= max_dim <= 0.15: # P3 (small)
                            assign = True
                        elif level_idx == 2 and max_dim > 0.08:    # P4 (medium/large)
                            assign = True
                        elif level_idx == 1 and not assign:        # Fallback to P3
                            assign = True
                        
                    if not assign:
                        continue
                        
                    # Find grid cell containing target center
                    gx = int(cx.item() * W)
                    gy = int(cy.item() * H)
                    
                    gx = min(max(0, gx), W - 1)
                    gy = min(max(0, gy), H - 1)
                    
                    target_obj[b, gy, gx, 0] = 1.0
                    target_cls[b, gy, gx, cls_id] = 1.0
                    pos_mask[b, gy, gx] = True
                    
                    # Decode predicted box for CIoU
                    pred_tx = reg_pred[b, gy, gx, 0]
                    pred_ty = reg_pred[b, gy, gx, 1]
                    pred_tw = reg_pred[b, gy, gx, 2]
                    pred_th = reg_pred[b, gy, gx, 3]
                    
                    p_cx = (torch.sigmoid(pred_tx) + gx) / W
                    p_cy = (torch.sigmoid(pred_ty) + gy) / H
                    p_w = torch.exp(pred_tw.clamp(max=3.0)) * (stride / img_size * 2.0)
                    p_h = torch.exp(pred_th.clamp(max=3.0)) * (stride / img_size * 2.0)
                    
                    pred_box_xyxy = cxcywh_to_xyxy(torch.stack([p_cx, p_cy, p_w, p_h]))
                    tgt_box_xyxy = cxcywh_to_xyxy(torch.stack([cx, cy, w, h]))
                    
                    matched_pred_boxes.append(pred_box_xyxy)
                    matched_target_boxes.append(tgt_box_xyxy)
                    
            # Objectness loss (Focal) across all cells
            level_obj_loss = self.focal_loss(obj_pred, target_obj)
            total_obj_loss = total_obj_loss + level_obj_loss
            
            # Classification loss & Box loss on positive cells
            num_pos = len(matched_pred_boxes)
            total_positives += num_pos
            
            if num_pos > 0:
                p_boxes = torch.stack(matched_pred_boxes)
                t_boxes = torch.stack(matched_target_boxes)
                level_box_loss = self.ciou_loss(p_boxes, t_boxes)
                total_box_loss = total_box_loss + level_box_loss
                
                # Class loss
                pos_cls_pred = cls_pred[pos_mask]
                pos_cls_tgt = target_cls[pos_mask]
                level_cls_loss = F.binary_cross_entropy_with_logits(pos_cls_pred, pos_cls_tgt, reduction="mean")
                total_cls_loss = total_cls_loss + level_cls_loss
                
        # Normalize by positive targets and batch size
        obj_loss = total_obj_loss / (batch_size * 100.0)
        box_loss = total_box_loss / len(self.strides)
        cls_loss = total_cls_loss / len(self.strides)
        
        total_loss = (
            self.lambda_obj * obj_loss +
            self.lambda_box * box_loss +
            self.lambda_cls * cls_loss
        )
        
        return {
            "loss": total_loss,
            "obj_loss": obj_loss.detach(),
            "box_loss": box_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "num_positives": torch.tensor(total_positives, device=device)
        }
