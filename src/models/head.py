"""
Decoupled Multi-Scale Detection Heads for Small UAV Detection.
Separates classification, objectness, and bounding box regression branches.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from .backbone import ConvBNAct

class DecoupledDetectionHead(nn.Module):
    """
    Decoupled Head on a single feature pyramid level.
    Uses separate convolutional branches for regression and classification/objectness
    to eliminate task-interference gradients.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        num_anchors: int = 1,
        stride: int = 8
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.stride = stride
        
        # Classification branch
        self.cls_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        )
        
        # Objectness branch
        self.obj_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1)
        )
        
        # Bounding box regression branch: (tx, ty, tw, th)
        self.reg_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        )
        
        self._init_biases()

    def _init_biases(self):
        # Prior probability for focal loss: bias = -log((1-pi)/pi) with pi=0.01 -> ~ -4.6
        nn.init.constant_(self.obj_branch[-1].bias, -4.59)
        nn.init.constant_(self.cls_branch[-1].bias, -4.59)
        nn.init.zeros_(self.reg_branch[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_preds: (B, num_anchors * num_classes, H, W)
            obj_preds: (B, num_anchors * 1, H, W)
            reg_preds: (B, num_anchors * 4, H, W)
        """
        cls_preds = self.cls_branch(x)
        obj_preds = self.obj_branch(x)
        reg_preds = self.reg_branch(x)
        return cls_preds, obj_preds, reg_preds

class MultiScaleDetectionHead(nn.Module):
    """
    Coordinates decoupled heads across multiple pyramid levels (P2, P3, P4).
    """
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 1,
        strides: List[int] = [4, 8, 16]
    ):
        super().__init__()
        self.strides = strides
        self.heads = nn.ModuleList([
            DecoupledDetectionHead(in_channels, num_classes=num_classes, num_anchors=1, stride=s)
            for s in strides
        ])

    def forward(self, pyramid_features: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        outputs = []
        for feat, head in zip(pyramid_features, self.heads):
            outputs.append(head(feat))
        return outputs
