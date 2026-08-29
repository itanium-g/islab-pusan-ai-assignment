"""
Custom From-Scratch Convolutional Backbones for Small UAV Detection.
Zero pretrained weights. Implements ConvBNAct, Residual Blocks,
Receptive Field Blocks (RFB), and Multi-Scale Feature Extractors.
"""

from typing import List, Tuple
import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    """
    Standard Convolution + BatchNorm2d + Activation block with Kaiming He initialization.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        act: str = "silu"
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
            
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    Lightweight Residual Block: Conv3x3 -> Conv3x3 with identity shortcut.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: float = 1.0):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNAct(mid_channels, out_channels, kernel_size=3, stride=1, act="none")
        self.act = nn.SiLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=stride, act="none")
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.act(out + res)

class ReceptiveFieldBlock(nn.Module):
    """
    Receptive Field Block (RFB) with multi-branch dilated convolutions.
    Expands receptive field without spatial downsampling, ideal for tiny drone targets.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        inter_channels = in_channels // 4
        
        # Branch 0: 1x1 conv
        self.branch0 = nn.Sequential(
            ConvBNAct(in_channels, inter_channels, kernel_size=1)
        )
        
        # Branch 1: 1x1 conv -> 3x3 conv (rate 1)
        self.branch1 = nn.Sequential(
            ConvBNAct(in_channels, inter_channels, kernel_size=1),
            ConvBNAct(inter_channels, inter_channels, kernel_size=3, padding=1)
        )
        
        # Branch 2: 1x1 conv -> 3x3 conv (rate 2)
        self.branch2 = nn.Sequential(
            ConvBNAct(in_channels, inter_channels, kernel_size=1),
            ConvBNAct(inter_channels, inter_channels, kernel_size=3, padding=2, dilation=2)
        )
        
        # Branch 3: 1x1 conv -> 3x3 conv (rate 3)
        self.branch3 = nn.Sequential(
            ConvBNAct(in_channels, inter_channels, kernel_size=1),
            ConvBNAct(inter_channels, inter_channels, kernel_size=3, padding=3, dilation=3)
        )
        
        self.out_conv = ConvBNAct(inter_channels * 4, out_channels, kernel_size=1, act="none")
        self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, act="none") if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b0, b1, b2, b3], dim=1)
        out = self.out_conv(concat)
        return self.act(out + self.shortcut(x))

class VanillaCNNBackbone(nn.Module):
    """
    Model 1 Baseline Backbone: Plain sequential CNN with max pooling.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, base_channels, kernel_size=3, stride=1),
            ConvBNAct(base_channels, base_channels, kernel_size=3, stride=2)  # 1/2
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels, base_channels * 2, kernel_size=3, stride=2),  # 1/4
            ConvBNAct(base_channels * 2, base_channels * 2, kernel_size=3, stride=1)
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 4, kernel_size=3, stride=2),  # 1/8
            ConvBNAct(base_channels * 4, base_channels * 4, kernel_size=3, stride=1)
        )
        self.out_channels = [base_channels * 4]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return [x]  # Returns P3 feature map (1/8 scale)

class DroneBackbone(nn.Module):
    """
    High-Performance Multi-Scale Residual Backbone with optional RFB module.
    Outputs:
        C2: Stride 4 (160x160 for 640x640 input)
        C3: Stride 8 (80x80 for 640x640 input)
        C4: Stride 16 (40x40 for 640x640 input)
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, use_rfb: bool = True):
        super().__init__()
        self.use_rfb = use_rfb
        c = base_channels
        
        # Stem (Stride 2)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c, kernel_size=3, stride=1),
            ConvBNAct(c, c, kernel_size=3, stride=2)
        )
        
        # Stage 2: Stride 4 (C2)
        self.stage2 = nn.Sequential(
            ResidualBlock(c, c * 2, stride=2),
            ResidualBlock(c * 2, c * 2, stride=1)
        )
        
        # Stage 3: Stride 8 (C3)
        self.stage3 = nn.Sequential(
            ResidualBlock(c * 2, c * 4, stride=2),
            ResidualBlock(c * 4, c * 4, stride=1),
            ResidualBlock(c * 4, c * 4, stride=1)
        )
        
        # Stage 4: Stride 16 (C4)
        stage4_layers = [
            ResidualBlock(c * 4, c * 8, stride=2),
            ResidualBlock(c * 8, c * 8, stride=1)
        ]
        if use_rfb:
            stage4_layers.append(ReceptiveFieldBlock(c * 8, c * 8))
        self.stage4 = nn.Sequential(*stage4_layers)
        
        self.out_channels = [c * 2, c * 4, c * 8]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)        # (B, C, H/2, W/2)
        c2 = self.stage2(x)      # (B, 2C, H/4, W/4)
        c3 = self.stage3(c2)     # (B, 4C, H/8, W/8)
        c4 = self.stage4(c3)     # (B, 8C, H/16, W/16)
        return c2, c3, c4
