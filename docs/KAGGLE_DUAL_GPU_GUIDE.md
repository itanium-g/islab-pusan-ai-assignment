# ☁️ Kaggle Dual-GPU Distributed Training Guide

This guide details how to launch, monitor, and retrieve results from the automated **Dual NVIDIA Tesla T4 DDP** training pipeline on Kaggle.

---

## 1. Overview & Cloud Architecture

- **Kernel URL**: [https://www.kaggle.com/code/itanium/drone-detection-islab-dual-gpu](https://www.kaggle.com/code/itanium/drone-detection-islab-dual-gpu)
- **Accelerator**: `NvidiaTeslaT4` (Dual Tesla T4 GPUs, 2x 16 GB VRAM)
- **Engine**: PyTorch `torchrun` DistributedDataParallel (DDP) with Automatic Mixed Precision (AMP)
- **Dataset Source**: Automatically downloaded from Google Drive (`19L9yUP62xMESJMw6srf5HGcL8s5b0gv8`) and extracted to `/tmp`

---

## 2. Command-Line Interface (CLI) Execution

### 2.1 Rebuild Self-Contained Notebook
```powershell
python scripts/build_kaggle_notebook.py
```

### 2.2 Push & Launch on Kaggle Dual Tesla T4
```powershell
.\venv\Scripts\kaggle.exe kernels push -p notebooks --accelerator NvidiaTeslaT4
```

### 2.3 Monitor Live Execution Status
```powershell
.\venv\Scripts\kaggle.exe kernels status itanium/drone-detection-islab-dual-gpu
```

### 2.4 Download Generated Output Artifacts
Once status reaches `"complete"`, download all model weights, evaluation plots, and the compiled IEEE Paper PDF:
```powershell
.\venv\Scripts\kaggle.exe kernels output itanium/drone-detection-islab-dual-gpu -p kaggle_output
```

---

## 3. Generated Output Deliverables

Upon completion, Kaggle produces the following lightweight outputs (< 15 MB total):
- `weights/DroneNet-FPN-Attention_inference.pth` (Stripped inference weights, 16.6 MB)
- `weights/DroneNet-FPN-Attention.onnx` (ONNX computational graph)
- `weights/DroneNet-FPN-Attention.torchscript.pt` (TorchScript traced model)
- `runs/eval/` (Evaluation metrics, PR curves, Confusion Matrices)
- `paper/Drone_Detection_Paper.pdf` (Compiled IEEE Conference Paper)
