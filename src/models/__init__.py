"""
Models subpackage for Drone Object Detection.
Contains scratch backbones, FPN necks, decoupled detection heads, custom multi-task loss, and detector assembly.
"""

from .backbone import DroneBackbone, VanillaCNNBackbone, ReceptiveFieldBlock
from .neck import HighResFPN, CoordinateAttention, SqueezeExcitationBlock
from .head import DecoupledDetectionHead, MultiScaleDetectionHead
from .loss import CustomMultiTaskLoss, FocalObjectnessLoss, CompleteIoULoss
from .detector import DroneDetector, build_detector
