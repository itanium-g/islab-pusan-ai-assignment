"""
Comprehensive Evaluation & Benchmarking Entrypoint.
Evaluates model checkpoints against the test set, generates Precision-Recall curves,
computes COCO metrics (mAP@0.5, mAP@0.75, mAP@0.5:0.95), latency (FPS), and domain breakdowns.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tabulate import tabulate

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config_parser import load_config
from src.data.dataset import DroneDataset
from src.data.dataloader import build_dataloader
from src.models.detector import build_detector
from src.engine.evaluator import Evaluator
from src.utils.visualization import plot_pr_curve

def get_safe_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        test_t = torch.zeros(1, device="cuda:0") + 1
        _ = test_t.item()
        return torch.device("cuda:0")
    except Exception as e:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Drone Detection Models.")
    parser.add_argument("--config", type=str, default="configs/model3_fpn_attn.yaml", help="Path to model config")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint weights")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--output-dir", type=str, default="runs/eval", help="Directory to save evaluation reports and plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Config
    config = load_config(args.config)
    device = get_safe_device()
    print(f"Running evaluation on Device: {device}")
    
    # 2. Build Dataset
    manifest_map = {
        "train": config.get("dataset", {}).get("train_manifest", "data/splits/train.txt"),
        "val": config.get("dataset", {}).get("val_manifest", "data/splits/val.txt"),
        "test": config.get("dataset", {}).get("test_manifest", "data/splits/test.txt")
    }
    manifest_path = manifest_map.get(args.split)
    
    dataset = DroneDataset(
        manifest_path=manifest_path,
        img_size=config.get("dataset", {}).get("img_size", 640),
        is_train=False,
        use_cached=config.get("dataset", {}).get("use_cached", True),
        cached_dir=config.get("dataset", {}).get("cached_dir", "data/cached_640")
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        is_distributed=False
    )
    
    # 3. Build Model & Load Weights
    model = build_detector(config.get("model", {}))
    if os.path.exists(args.weights):
        print(f"Loading checkpoint weights from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {args.weights}. Evaluating randomly initialized model.")
        
    model.to(device)
    model.eval()
    
    # 4. Run Evaluation
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
    
    metrics = evaluator.evaluate()
    
    # 5. Format Metric Output Table
    table_data = [
        ["Model Architecture", model.name],
        ["Dataset Split", args.split.upper()],
        ["Total Evaluated Frames", metrics.get("num_samples", len(dataset))],
        ["Precision", f"{metrics.get('precision', 0.0)*100:.2f}%"],
        ["Recall", f"{metrics.get('recall', 0.0)*100:.2f}%"],
        ["F1 Score", f"{metrics.get('f1', 0.0):.4f}"],
        ["mAP @ IoU=0.50 (AP50)", f"{metrics.get('map50', 0.0)*100:.2f}%"],
        ["mAP @ IoU=0.75 (AP75)", f"{metrics.get('map75', 0.0)*100:.2f}%"],
        ["mAP @ IoU=0.50:0.95 (COCO)", f"{metrics.get('map50_95', 0.0)*100:.2f}%"],
        ["Inference Latency", f"{metrics.get('latency_ms', 0.0):.2f} ms / frame"],
        ["Throughput (FPS)", f"{metrics.get('fps', 0.0):.1f} FPS"]
    ]
    
    print("\n" + "="*50)
    print(f"  BENCHMARK REPORT: {model.name}")
    print("="*50)
    print(tabulate(table_data, headers=["Metric / Parameter", "Value"], tablefmt="fancy_grid"))
    
    # 6. Save Precision-Recall Curve
    pr_curve_path = os.path.join(args.output_dir, f"{model.name}_pr_curve.png")
    plot_pr_curve(recalls=metrics.get("recalls", []), precisions=metrics.get("precisions", []), ap=metrics.get("mAP50", 0.0), save_path=pr_curve_path)
    print(f"\nPrecision-Recall curve saved to: {pr_curve_path}")

if __name__ == "__main__":
    main()
