"""
Comprehensive Object Detection Evaluation Engine.
Computes mAP@0.5, mAP@0.75, mAP@0.5:0.95 (COCO metric), Precision, Recall, F1 score,
Inference Latency (FPS), and Environmental Domain Breakdown (Foggy vs Sunny, City/Forest/Lake).
"""

import time
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.box_ops import non_max_suppression, box_iou, cxcywh_to_xyxy

class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: torch.device = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, generate_plots: bool = False, max_batches: int = None) -> Dict[str, Any]:
        self.model.eval()
        
        all_detections = []
        all_ground_truths = []
        all_metas = []
        
        latencies = []
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        
        batch_count = 0
        for images, targets, metadatas in tqdm(self.dataloader, desc="Evaluating", leave=False):
            if max_batches and batch_count >= max_batches:
                break
            batch_count += 1
            
            images = images.to(self.device)
            B, _, H, W = images.shape
            
            # Measure inference time
            start_time = time.perf_counter()
            raw_predictions = self.model(images)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            inference_time = (time.perf_counter() - start_time) / B
            latencies.append(inference_time)
            
            # Apply NMS
            nms_predictions = non_max_suppression(
                raw_predictions,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres
            )
            
            for b in range(B):
                pred_boxes = nms_predictions[b].cpu()
                gt_boxes_norm = targets[b].cpu()
                
                # Convert normalized ground truth [cls, cx, cy, w, h] to pixel [x1, y1, x2, y2, cls]
                if len(gt_boxes_norm) > 0:
                    cls_ids = gt_boxes_norm[:, 0:1]
                    cxcywh = gt_boxes_norm[:, 1:5]
                    # scale by image dimensions
                    cxcywh_px = cxcywh * torch.tensor([W, H, W, H])
                    xyxy_px = cxcywh_to_xyxy(cxcywh_px)
                    gt_boxes_px = torch.cat([xyxy_px, cls_ids], dim=-1)
                else:
                    gt_boxes_px = torch.zeros((0, 5))
                    
                all_detections.append(pred_boxes)
                all_ground_truths.append(gt_boxes_px)
                all_metas.append(metadatas[b])

        # Compute Metrics across IoU thresholds
        metrics = self._compute_ap_metrics(all_detections, all_ground_truths, iou_thresholds)
        
        # Latency & FPS
        avg_latency = np.mean(latencies) * 1000.0  # ms
        fps = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0.0
        metrics["latency_ms"] = avg_latency
        metrics["fps"] = fps
        
        # Environmental breakdown (Foggy vs Sunny)
        domain_breakdown = self._compute_domain_breakdown(all_detections, all_ground_truths, all_metas)
        metrics["domains"] = domain_breakdown
        
        return metrics

    def _compute_ap_metrics(
        self,
        all_detections: List[torch.Tensor],
        all_ground_truths: List[torch.Tensor],
        iou_thresholds: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute AP@0.5, AP@0.75, and mAP@0.5:0.95.
        """
        # Flatten all predictions with image IDs
        pred_records = []
        for img_id, dets in enumerate(all_detections):
            if len(dets) == 0:
                continue
            for det in dets:
                x1, y1, x2, y2, score, cls_id = det.tolist()
                pred_records.append({
                    "img_id": img_id,
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "cls_id": int(cls_id)
                })
                
        # Sort all predictions by confidence descending
        pred_records = sorted(pred_records, key=lambda x: x["score"], reverse=True)
        total_gts = sum(len(gt) for gt in all_ground_truths)
        
        if total_gts == 0:
            return {"ap50": 0.0, "ap75": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        aps = []
        precisions_at_50 = []
        recalls_at_50 = []
        
        for iou_thresh in iou_thresholds:
            detected_gts = {img_id: np.zeros(len(all_ground_truths[img_id]), dtype=bool) for img_id in range(len(all_ground_truths))}
            
            tp = np.zeros(len(pred_records))
            fp = np.zeros(len(pred_records))
            
            for p_idx, pred in enumerate(pred_records):
                img_id = pred["img_id"]
                p_box = torch.tensor(pred["box"])[None, :]
                gts = all_ground_truths[img_id]
                
                if len(gts) == 0:
                    fp[p_idx] = 1.0
                    continue
                    
                gt_boxes = gts[:, :4]
                ious = box_iou(p_box, gt_boxes).squeeze(0).numpy()
                best_gt_idx = np.argmax(ious)
                best_iou = ious[best_gt_idx]
                
                if best_iou >= iou_thresh:
                    if not detected_gts[img_id][best_gt_idx]:
                        tp[p_idx] = 1.0
                        detected_gts[img_id][best_gt_idx] = True
                    else:
                        fp[p_idx] = 1.0  # Duplicate detection
                else:
                    fp[p_idx] = 1.0
                    
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            recall = cum_tp / max(1, total_gts)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
            
            # 11-point interpolated AP
            ap = self._voc_ap(recall, precision)
            aps.append(ap)
            
            if np.isclose(iou_thresh, 0.50):
                precisions_at_50 = precision
                recalls_at_50 = recall
                
        ap50 = aps[0]
        ap75 = aps[5] if len(aps) > 5 else 0.0
        map50_95 = np.mean(aps)
        
        # Max F1 score at IoU 0.50
        if len(precisions_at_50) > 0 and len(recalls_at_50) > 0:
            f1_scores = 2 * (precisions_at_50 * recalls_at_50) / (precisions_at_50 + recalls_at_50 + 1e-7)
            best_idx = np.argmax(f1_scores)
            p_final = float(precisions_at_50[best_idx])
            r_final = float(recalls_at_50[best_idx])
            f1_final = float(f1_scores[best_idx])
        else:
            p_final, r_final, f1_final = 0.0, 0.0, 0.0
            
        return {
            "ap50": float(ap50),
            "map50": float(ap50),
            "mAP50": float(ap50),
            "mAP_50": float(ap50),
            "ap75": float(ap75),
            "map75": float(ap75),
            "mAP75": float(ap75),
            "map50_95": float(map50_95),
            "mAP50_95": float(map50_95),
            "precision": p_final,
            "recall": r_final,
            "f1": f1_final,
            "recalls": recalls_at_50,
            "precisions": precisions_at_50,
            "recalls_curve": recalls_at_50,
            "precisions_curve": precisions_at_50
        }

    def _voc_ap(self, rec: np.ndarray, prec: np.ndarray) -> float:
        """
        11-point VOC Average Precision calculation.
        """
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return float(ap)

    def _compute_domain_breakdown(
        self,
        all_detections: List[torch.Tensor],
        all_ground_truths: List[torch.Tensor],
        all_metas: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate AP@0.5 separately on Foggy vs Sunny and City vs Forest vs Lake.
        """
        domain_subsets = {"foggy": ([], []), "sunny": ([], []), "city": ([], []), "forest": ([], []), "lake": ([], [])}
        
        for dets, gts, meta in zip(all_detections, all_ground_truths, all_metas):
            p = meta.get("img_path", "").lower()
            if "foggy" in p:
                domain_subsets["foggy"][0].append(dets)
                domain_subsets["foggy"][1].append(gts)
            if "sunny" in p:
                domain_subsets["sunny"][0].append(dets)
                domain_subsets["sunny"][1].append(gts)
            if "city" in p:
                domain_subsets["city"][0].append(dets)
                domain_subsets["city"][1].append(gts)
            if "forest" in p:
                domain_subsets["forest"][0].append(dets)
                domain_subsets["forest"][1].append(gts)
            if "lake" in p:
                domain_subsets["lake"][0].append(dets)
                domain_subsets["lake"][1].append(gts)
                
        results = {}
        for dom, (d_list, g_list) in domain_subsets.items():
            if len(d_list) > 0 and sum(len(g) for g in g_list) > 0:
                res = self._compute_ap_metrics(d_list, g_list, np.array([0.50]))
                results[f"{dom}_ap50"] = res["ap50"]
            else:
                results[f"{dom}_ap50"] = 0.0
                
        return results
