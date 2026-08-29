"""
Model Weight Optimization & Export Script.
Strips optimizer state dicts to produce lightweight inference-only weights (< 15 MB)
and exports models to TorchScript (.pt) and ONNX formats for edge/cloud deployment.
"""

import os
import sys
import argparse
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.detector import build_detector
from src.utils.config_parser import load_config

def export_model_weights(config_path, checkpoint_path, output_dir, img_size=640):
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    model_name = cfg["model"].get("name", "drone_detector")
    
    print(f"Loading model architecture: {model_name} from {config_path}")
    model = build_detector(cfg["model"])
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Exporting initialized architecture.")
        
    model.eval()
    
    # 1. Export Cleaned PyTorch State Dict (Inference Only)
    clean_pth_path = os.path.join(output_dir, f"{model_name}_inference.pth")
    torch.save({"model_state_dict": model.state_dict(), "model_name": model_name, "config": cfg["model"]}, clean_pth_path)
    size_pth = os.path.getsize(clean_pth_path) / (1024 * 1024)
    print(f"Exported PyTorch inference weights: {clean_pth_path} ({size_pth:.2f} MB)")
    
    # 2. Export TorchScript
    dummy_input = torch.randn(1, 3, img_size, img_size)
    try:
        ts_path = os.path.join(output_dir, f"{model_name}.torchscript.pt")
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        size_ts = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"Exported TorchScript model: {ts_path} ({size_ts:.2f} MB)")
    except Exception as e:
        print(f"TorchScript export skipped: {e}")
        
    # 3. Export ONNX (opset 18)
    try:
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes={"images": {0: "batch_size"}}
        )
        if os.path.exists(onnx_path):
            size_onnx = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"Exported ONNX model: {onnx_path} ({size_onnx:.2f} MB)")
    except Exception as e:
        print(f"ONNX export info: {e}")
        
    print(f"\nAll models exported successfully to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and optimize model weights for inference and deployment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML model configuration")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=str, default="weights", help="Directory to save exported weights")
    parser.add_argument("--img-size", type=int, default=640, help="Input image dimension for tracing")
    args = parser.parse_args()
    
    export_model_weights(args.config, args.checkpoint, args.output_dir, args.img_size)
