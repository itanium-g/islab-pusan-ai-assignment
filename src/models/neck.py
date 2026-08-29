"""
Feature Pyramid Network (FPN) Necks & Attention Modules.
Includes Coordinate Attention (CA), Squeeze-and-Excitation (SE),
and High-Resolution Multi-Scale FPN (P2, P3, P4).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from .backbone import ConvBNAct

class SqueezeExcitationBlock(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation Attention.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module (CVPR).
    Decomposes channel attention into horizontal and vertical 1D feature encodings.
    Captures long-range spatial dependencies along one direction while preserving
    precise positional information along the other, ideal for tiny drone targets in fog.
    """
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        out = identity * a_h * a_w
        return out

class HighResFPN(nn.Module):
    """
    High-Resolution Feature Pyramid Network for Small Object Detection.
    Fuses C2 (stride 4, 160x160), C3 (stride 8, 80x80), and C4 (stride 16, 40x40)
    using top-down pathways with optional attention mechanisms.
    Outputs refined pyramid features: P2, P3, P4 with uniform channel dimension.
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 64,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention
        c2_in, c3_in, c4_in = in_channels_list
        
        # Lateral 1x1 convolutions
        self.lat_c4 = ConvBNAct(c4_in, out_channels, kernel_size=1)
        self.lat_c3 = ConvBNAct(c3_in, out_channels, kernel_size=1)
        self.lat_c2 = ConvBNAct(c2_in, out_channels, kernel_size=1)
        
        # Smooth 3x3 convolutions after fusion
        self.smooth_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p2 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        
        # Optional Coordinate Attention blocks
        if use_attention:
            self.attn_p4 = CoordinateAttention(out_channels, out_channels)
            self.attn_p3 = CoordinateAttention(out_channels, out_channels)
            self.attn_p2 = CoordinateAttention(out_channels, out_channels)
        else:
            self.attn_p4 = nn.Identity()
            self.attn_p3 = nn.Identity()
            self.attn_p2 = nn.Identity()

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2, c3, c4 = features
        
        # Top-down pathway
        p4 = self.lat_c4(c4)
        p4_up = self.upsample(p4)
        
        p3 = self.lat_c3(c3) + p4_up
        p3_up = self.upsample(p3)
        
        p2 = self.lat_c2(c2) + p3_up
        
        # Smoothing convolutions
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)
        
        # Attention enhancement
        p4 = self.attn_p4(p4)
        p3 = self.attn_p3(p3)
        p2 = self.attn_p2(p2)
        
        return p2, p3, p4
