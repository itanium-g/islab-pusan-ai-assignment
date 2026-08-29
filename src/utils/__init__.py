"""
Utilities package for bounding box operations, logging, config parsing, and visualization.
"""

from .box_ops import (
    box_iou,
    box_ciou,
    box_giou,
    xywh_to_xyxy,
    xyxy_to_xywh,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
    non_max_suppression,
    encode_boxes,
    decode_boxes
)
from .config_parser import load_config
from .logger import ExperimentLogger
from .visualization import draw_bounding_boxes, plot_batch_predictions
