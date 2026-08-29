"""
Inference & Visual Detection Script.
Runs trained drone detector models on individual images, folders, or video streams
and outputs annotated visualization renders.
"""

import os
import sys
import argparse
import glob
import time
from PIL import Image
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config_parser import load_config
from src.models.detector import build_detector
from src.data.transforms import letterbox_image_and_boxes, ToTensor
from src.utils.box_ops import non_max_suppression
from src.utils.visualization import draw_bounding_boxes

def get_safe_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        test_t = torch.zeros(1, device="cuda:0") + 1
        _ = test_t.item()
        return torch.device("cuda:0")
    except Exception as e:
        return torch.device("cpu")

def run_inference_on_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    class_names: list = ["Drone"]
):
    with Image.open(image_path) as img:
        original_img = img.convert("RGB")
        
    orig_w, orig_h = original_img.size
    
    # 1. Letterbox resize image
    padded_img, _ = letterbox_image_and_boxes(original_img, None, (img_size, img_size))
    
    # 2. Convert to Tensor
    to_tensor = ToTensor()
    img_tensor, _ = to_tensor(padded_img, None)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 3. Model Forward Pass
    start_time = time.perf_counter()
    with torch.no_grad():
        predictions = model(img_tensor) # (1, total_anchors, 6)
        detections = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=iou_thres)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # 4. Map boxes back to original image scale
    scale = min(img_size / orig_w, img_size / orig_h)
    pad_x = (img_size - orig_w * scale) / 2
    pad_y = (img_size - orig_h * scale) / 2
    
    annotated_img = original_img.copy()
    if detections is not None and len(detections) > 0:
        det_boxes = []
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det.tolist()
            # Un-pad & un-scale
            rx1 = (x1 - pad_x) / scale
            ry1 = (y1 - pad_y) / scale
            rx2 = (x2 - pad_x) / scale
            ry2 = (y2 - pad_y) / scale
            det_boxes.append([rx1, ry1, rx2, ry2, score, cls_id])
            
        annotated_img = draw_bounding_boxes(
            image=annotated_img,
            boxes=det_boxes,
            class_names=class_names,
            color=(0, 255, 0),
            thickness=3
        )
        
    return annotated_img, len(detections) if detections is not None else 0, latency_ms

def main():
    parser = argparse.ArgumentParser(description="Run Visual Drone Detection Inference.")
    parser.add_argument("--config", type=str, default="configs/model3_fpn_attn.yaml", help="Path to model config")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth", help="Path to trained weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image, directory of images, or video")
    parser.add_argument("--output-dir", type=str, default="runs/inference", help="Directory to save annotated renders")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence score threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference resolution")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = get_safe_device()
    print(f"Loading model on device: {device}")
    
    # 1. Load Model
    cfg = load_config(args.config)
    model = build_detector(cfg.get("model", {}))
    
    if os.path.exists(args.weights):
        print(f"Loading checkpoint weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else (checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {args.weights}. Using initialized architecture.")
        
    model.to(device)
    model.eval()
    
    # 2. Collect image sources
    if os.path.isfile(args.source):
        img_paths = [args.source]
    elif os.path.isdir(args.source):
        img_paths = glob.glob(os.path.join(args.source, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(args.source, "*.[pP][nN][gG]"))
    else:
        raise ValueError(f"Source {args.source} not found.")
        
    print(f"Processing {len(img_paths)} images...")
    total_latency = 0.0
    
    for idx, img_path in enumerate(img_paths):
        out_img, num_dets, lat = run_inference_on_image(
            image_path=img_path,
            model=model,
            device=device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
        total_latency += lat
        
        base_name = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, f"det_{base_name}")
        out_img.save(out_path)
        
        if idx < 5 or idx % 50 == 0:
            print(f"[{idx+1}/{len(img_paths)}] {base_name}: Detected {num_dets} drones ({lat:.1f} ms) -> {out_path}")
            
    avg_fps = (1000.0 * len(img_paths)) / max(1.0, total_latency)
    print(f"\nInference complete! Average Latency: {total_latency/max(1, len(img_paths)):.2f} ms ({avg_fps:.1f} FPS)")
    print(f"All annotated images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
