"""
Vectorized Bounding Box Operations & Geometry Module.
Includes Coordinate Transforms, IoU, GIoU, CIoU, Bounding Box Encoding/Decoding,
and Vectorized Non-Maximum Suppression (NMS).
"""

import math
import torch
import torchvision

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (cx, cy, w, h) to (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (x1, y1, x2, y2) to (cx, cy, w, h).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (x1, y1, w, h) to (x1, y1, x2, y2).
    """
    x1, y1, w, h = boxes.unbind(-1)
    x2 = x1 + w
    y2 = y1 + h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (x1, y1, x2, y2) to (x1, y1, w, h).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x1, y1, w, h], dim=-1)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate pairwise Intersection over Union (IoU) between two sets of boxes (xyxy).
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) matrix of IoU values.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union

def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Generalized IoU (GIoU) between corresponding pairs of boxes (xyxy).
    boxes1: (N, 4)
    boxes2: (N, 4)
    Returns: (N,) GIoU values.
    """
    # Intersection
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    iou = inter / union
    
    # Smallest enclosing box
    enc_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enc_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enc_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enc_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + eps
    
    return iou - (enc_area - union) / enc_area

def box_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Complete IoU (CIoU) between corresponding pairs of boxes (xyxy format).
    Optimizes overlap area, center point Euclidean distance, and aspect ratio consistency.
    boxes1: (N, 4) - predicted boxes
    boxes2: (N, 4) - ground truth boxes
    Returns: (N,) CIoU values in range [-1, 1]
    """
    # 1. Standard IoU
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    w1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0)
    h1 = (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    w2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0)
    h2 = (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + eps
    iou = inter / union
    
    # 2. Center distance penalty
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) * 0.5
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) * 0.5
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) * 0.5
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) * 0.5
    
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # 3. Enclosing diagonal distance squared
    enc_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enc_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enc_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enc_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps
    
    # 4. Aspect ratio penalty
    atan1 = torch.atan(w1 / (h1 + eps))
    atan2 = torch.atan(w2 / (h2 + eps))
    v = (4.0 / (math.pi ** 2)) * torch.pow(atan2 - atan1, 2)
    
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
        
    ciou = iou - (rho2 / c2) - (alpha * v)
    return ciou

def encode_boxes(gt_boxes: torch.Tensor, anchor_grid: torch.Tensor) -> torch.Tensor:
    """
    Encode ground truth boxes (cxcywh normalized) relative to anchor grid.
    """
    # Grid offset: (cx_gt - cx_grid), (cy_gt - cy_grid), log(w_gt / w_grid), log(h_gt / h_grid)
    grid_xy = anchor_grid[..., :2]
    grid_wh = anchor_grid[..., 2:]
    
    tx = gt_boxes[..., 0:2] - grid_xy
    tw = torch.log(gt_boxes[..., 2:4] / grid_wh.clamp(min=1e-6) + 1e-6)
    return torch.cat([tx, tw], dim=-1)

def decode_boxes(pred_offsets: torch.Tensor, anchor_grid: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted offsets (tx, ty, tw, th) relative to anchor grid into cxcywh normalized.
    """
    grid_xy = anchor_grid[..., :2]
    grid_wh = anchor_grid[..., 2:]
    
    pred_xy = torch.sigmoid(pred_offsets[..., 0:2]) + grid_xy - 0.5
    pred_wh = torch.exp(pred_offsets[..., 2:4].clamp(max=4.0)) * grid_wh
    return torch.cat([pred_xy, pred_wh], dim=-1)

def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 100
) -> list:
    """
    Perform Non-Maximum Suppression (NMS) on inference predictions.
    prediction: Tensor of shape (B, N, 6) where each row is (x1, y1, x2, y2, obj_conf, class_conf/class_id)
    Returns: List of length B, each element is a Tensor of shape (K, 6) [x1, y1, x2, y2, score, class_id]
    """
    output = []
    
    for i, pred in enumerate(prediction):
        # Filter by confidence
        # pred shape: (N, 4 + 1 + num_classes) or (N, 6)
        if pred.shape[-1] > 6:
            boxes = pred[:, :4]
            obj_conf = pred[:, 4:5]
            cls_scores = pred[:, 5:]
            scores, class_ids = torch.max(cls_scores, dim=1, keepdim=True)
            scores = scores * obj_conf
            detections = torch.cat([boxes, scores, class_ids.float()], dim=1)
        else:
            detections = pred
            
        # Confidence threshold filter
        mask = detections[:, 4] > conf_thres
        detections = detections[mask]
        
        if detections.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue
            
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5]
        
        # Batched NMS (class-specific)
        offsets = class_ids * 4096.0
        boxes_for_nms = boxes + offsets[:, None]
        
        keep_indices = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)
        keep_indices = keep_indices[:max_det]
        
        output.append(detections[keep_indices])
        
    return output
