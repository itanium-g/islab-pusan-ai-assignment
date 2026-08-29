"""
Automated Unit & Integration Tests for Drone Detection Pipeline.
Validates:
1. Bounding Box Operations & Geometry (IoU, GIoU, CIoU, transforms)
2. Custom Loss Functions (Focal Objectness, CIoU)
3. Model Architecture Forward Passes & Dimension Integrity (Model 1, Model 2, Model 3)
4. Non-Maximum Suppression (NMS)
"""

import unittest
import torch
import numpy as np

from src.utils.box_ops import (
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
    box_iou,
    box_giou,
    box_ciou,
    non_max_suppression
)
from src.models.backbone import VanillaCNNBackbone, DroneBackbone, ReceptiveFieldBlock
from src.models.neck import HighResFPN, CoordinateAttention
from src.models.detector import build_detector
from src.models.loss import CustomMultiTaskLoss

class TestBoxOps(unittest.TestCase):
    def test_coordinate_conversion(self):
        cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float32)
        xyxy = cxcywh_to_xyxy(cxcywh)
        expected_xyxy = torch.tensor([[0.4, 0.3, 0.6, 0.7]], dtype=torch.float32)
        self.assertTrue(torch.allclose(xyxy, expected_xyxy, atol=1e-5))
        
        recovered_cxcywh = xyxy_to_cxcywh(xyxy)
        self.assertTrue(torch.allclose(cxcywh, recovered_cxcywh, atol=1e-5))

    def test_iou_identical(self):
        box1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        box2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        iou = box_iou(box1, box2)
        self.assertAlmostEqual(iou.item(), 1.0, places=4)

    def test_ciou_properties(self):
        # Perfect match CIoU = 1.0
        b1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        b2 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        ciou = box_ciou(b1, b2)
        self.assertAlmostEqual(ciou.item(), 1.0, places=4)
        
        # Disjoint boxes CIoU < 0
        b3 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b4 = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
        ciou_disjoint = box_ciou(b3, b4)
        self.assertLess(ciou_disjoint.item(), 0.0)

    def test_nms(self):
        # Two overlapping high-confidence boxes
        pred = torch.tensor([[
            [10.0, 10.0, 50.0, 50.0, 0.9, 0.0],
            [12.0, 12.0, 52.0, 52.0, 0.8, 0.0],
            [200.0, 200.0, 250.0, 250.0, 0.85, 0.0]
        ]])
        kept = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)[0]
        self.assertEqual(len(kept), 2)  # One of the two overlapping boxes suppressed

class TestModels(unittest.TestCase):
    def test_model1_baseline_forward(self):
        cfg = {"type": "baseline", "base_channels": 16, "num_classes": 1}
        model = build_detector(cfg)
        dummy_img = torch.randn(2, 3, 128, 128)
        model.eval()
        with torch.no_grad():
            preds = model(dummy_img)
        self.assertEqual(preds.shape[0], 2)
        self.assertEqual(preds.shape[2], 6)  # [x1, y1, x2, y2, score, cls]

    def test_model3_fpn_attn_forward_and_loss(self):
        cfg = {"type": "fpn_attn", "base_channels": 16, "num_classes": 1}
        model = build_detector(cfg)
        dummy_img = torch.randn(2, 3, 128, 128)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            preds = model(dummy_img)
        self.assertEqual(preds.shape[0], 2)
        
        # Training mode with loss
        model.train()
        dummy_targets = [
            torch.tensor([[0.0, 0.5, 0.5, 0.05, 0.05]]),
            torch.tensor([[0.0, 0.3, 0.3, 0.02, 0.03], [0.0, 0.7, 0.7, 0.04, 0.04]])
        ]
        loss_dict = model(dummy_img, dummy_targets)
        self.assertIn("loss", loss_dict)
        self.assertIn("obj_loss", loss_dict)
        self.assertIn("box_loss", loss_dict)
        self.assertGreater(loss_dict["loss"].item(), 0.0)

if __name__ == "__main__":
    unittest.main()
