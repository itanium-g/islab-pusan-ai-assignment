"""
DroneDetector OOP Model Assembly & Model Registry.
Integrates Backbones, FPN Necks, Decoupled Heads, and Loss Functions
into unified end-to-end models with seamless forward and inference methods.
"""

from typing import List, Tuple, Dict, Any, Union
import torch
import torch.nn as nn

from .backbone import DroneBackbone, VanillaCNNBackbone
from .neck import HighResFPN
from .head import DecoupledDetectionHead, MultiScaleDetectionHead
from .loss import CustomMultiTaskLoss
from src.utils.box_ops import non_max_suppression, cxcywh_to_xyxy

class DroneDetector(nn.Module):
    """
    Modular Object Detector for Drone Localization and Classification.
    """
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        loss_fn: nn.Module = None,
        strides: List[int] = [4, 8, 16],
        num_classes: int = 1,
        name: str = "DroneDetector"
    ):
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.strides = strides
        self.num_classes = num_classes
        self.loss_fn = loss_fn or CustomMultiTaskLoss(strides=strides, num_classes=num_classes)

    def forward(
        self,
        images: torch.Tensor,
        targets: List[torch.Tensor] = None
    ) -> Union[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        In training mode (targets provided): returns loss dictionary.
        In eval/inference mode: returns decoded prediction bounding boxes [x1, y1, x2, y2, score, class_id].
        """
        B, C, H, W = images.shape
        
        # 1. Feature Extraction via Backbone
        backbone_feats = self.backbone(images)
        
        # 2. Feature Pyramid Fusion via Neck (if present)
        if self.neck is not None:
            pyramid_feats = self.neck(backbone_feats)
        else:
            pyramid_feats = backbone_feats if isinstance(backbone_feats, (list, tuple)) else [backbone_feats]
            
        # 3. Detection Head Projections
        predictions = self.head(pyramid_feats)
        
        # 4. Training vs Inference Return
        if self.training and targets is not None:
            loss_dict = self.loss_fn(predictions, targets, img_size=max(H, W))
            return loss_dict
            
        # 5. Decode Inference Predictions
        return self._decode_predictions(predictions, img_size=(W, H))

    def _decode_predictions(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Decode raw grid predictions into normalized [x1, y1, x2, y2, score, class_id].
        """
        W, H = img_size
        all_detections = []
        
        for level_idx, (cls_pred, obj_pred, reg_pred) in enumerate(predictions):
            stride = self.strides[level_idx] if level_idx < len(self.strides) else self.strides[-1]
            B, _, fH, fW = obj_pred.shape
            device = obj_pred.device
            
            # Reshape: (B, fH*fW, ...)
            cls_probs = torch.sigmoid(cls_pred).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            obj_probs = torch.sigmoid(obj_pred).permute(0, 2, 3, 1).reshape(B, -1, 1)
            reg_offsets = reg_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            
            # Confidence score = obj_prob * max(cls_prob)
            max_cls_scores, class_ids = torch.max(cls_probs, dim=-1, keepdim=True)
            scores = obj_probs * max_cls_scores
            
            # Grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(fH, device=device, dtype=torch.float32),
                torch.arange(fW, device=device, dtype=torch.float32),
                indexing="ij"
            )
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)
            
            # Decode centers & scales
            pred_cx = (torch.sigmoid(reg_offsets[..., 0]) + grid_x) / fW * W
            pred_cy = (torch.sigmoid(reg_offsets[..., 1]) + grid_y) / fH * H
            pred_w = torch.exp(reg_offsets[..., 2].clamp(max=3.0)) * (stride / max(W, H) * 2.0) * W
            pred_h = torch.exp(reg_offsets[..., 3].clamp(max=3.0)) * (stride / max(W, H) * 2.0) * H
            
            cxcywh = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
            xyxy = cxcywh_to_xyxy(cxcywh)
            
            level_dets = torch.cat([xyxy, scores, class_ids.float()], dim=-1)
            all_detections.append(level_dets)
            
        # Concat predictions across all pyramid levels: (B, total_anchors, 6)
        concat_detections = torch.cat(all_detections, dim=1)
        return concat_detections

def build_detector(model_cfg: Dict[str, Any]) -> DroneDetector:
    """
    Model Factory & Registry.
    Builds Model 1 (Baseline), Model 2 (FPN), or Model 3 (FPN + Attention).
    """
    model_type = model_cfg.get("type", "fpn_attn").lower()
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("num_classes", 1)
    base_channels = model_cfg.get("base_channels", 32)
    
    if model_type == "baseline":
        backbone = VanillaCNNBackbone(in_channels=in_channels, base_channels=base_channels)
        head = MultiScaleDetectionHead(in_channels=base_channels * 4, num_classes=num_classes, strides=[8])
        detector = DroneDetector(
            backbone=backbone,
            neck=None,
            head=head,
            strides=[8],
            num_classes=num_classes,
            name="VanillaDroneNet-Base"
        )
        
    elif model_type == "fpn":
        backbone = DroneBackbone(in_channels=in_channels, base_channels=base_channels, use_rfb=False)
        neck = HighResFPN(in_channels_list=backbone.out_channels, out_channels=base_channels * 2, use_attention=False)
        head = MultiScaleDetectionHead(in_channels=base_channels * 2, num_classes=num_classes, strides=[4, 8, 16])
        detector = DroneDetector(
            backbone=backbone,
            neck=neck,
            head=head,
            strides=[4, 8, 16],
            num_classes=num_classes,
            name="DroneNet-FPN"
        )
        
    elif model_type in ["fpn_attn", "best", "drone_net_attn"]:
        backbone = DroneBackbone(in_channels=in_channels, base_channels=base_channels, use_rfb=True)
        neck = HighResFPN(in_channels_list=backbone.out_channels, out_channels=base_channels * 2, use_attention=True)
        head = MultiScaleDetectionHead(in_channels=base_channels * 2, num_classes=num_classes, strides=[4, 8, 16])
        detector = DroneDetector(
            backbone=backbone,
            neck=neck,
            head=head,
            strides=[4, 8, 16],
            num_classes=num_classes,
            name="DroneNet-FPN-Attention"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return detector
