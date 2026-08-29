# 🚀 Getting Started Guide

This document covers installation, environment setup, local training, evaluation, inference, and Docker containerization for the DroneNet-FPN-Attention repository.

---

## 1. Environment Setup

### Option A: Local Python Virtual Environment (Windows PowerShell / Linux / macOS)
```bash
# 1. Create Python virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS / WSL:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
```

### Option B: WSL2 Ubuntu Environment
```bash
wsl -d Ubuntu
cd /mnt/c/Users/<YOUR_USERNAME>/projects/islab-pusan-ai-assignment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Option C: Docker & Docker Compose
```bash
# Build CUDA container and run training
docker compose -f docker/docker-compose.yml up train

# Run evaluation container
docker compose -f docker/docker-compose.yml up evaluate
```

---

## 2. Dataset Setup & Preprocessing

The raw dataset contains 2,400 multi-weather UAV frames (120 flight sequences).

### 2.1 Sequence-Aware Dataset Split
```bash
python scripts/split_dataset.py --dataset-dir curated_datasets/obj_det_base --output-dir data/splits
```

### 2.2 High-Speed 640x640 Pre-Caching
```bash
python scripts/preprocess_dataset.py --src-dir curated_datasets/obj_det_base --dest-dir data/cached_640 --img-size 640 --workers 8
```

---

## 3. Model Training

### Train Proposed Best Model (Model 3: DroneNet-FPN-Attention)
```bash
# Single GPU / Local CPU
python train.py --config configs/model3_fpn_attn.yaml --epochs 40 --batch-size 16

# Multi-GPU DistributedDataParallel (DDP)
torchrun --nproc_per_node=2 train.py --config configs/model3_fpn_attn.yaml --ddp
```

### Train Baseline Models (Ablation Comparison)
```bash
# Model 1: Vanilla Base CNN (Single Scale P3)
python train.py --config configs/model1_baseline.yaml --epochs 30 --batch-size 16

# Model 2: DroneNet-FPN (Multi-Scale P2/P3/P4)
python train.py --config configs/model2_fpn.yaml --epochs 35 --batch-size 16
```

---

## 4. Evaluation & Benchmarking

```bash
python evaluate.py --config configs/model3_fpn_attn.yaml --weights runs/train/model3_fpn_attention_best/checkpoints/best_model.pth --split test
```

---

## 5. Inference & Visualization

```bash
python infer.py --weights weights/DroneNet-FPN-Attention_inference.pth --source curated_datasets/obj_det_base/augmented_raw_dataset_city_foggy_city_foggy_0_1_sequence.164_step0.camera.png --output-dir runs/infer --conf-thresh 0.35
```

---

## 6. Weight Export & Paper Compilation

```bash
# Export lightweight stripped weights (< 15 MB), ONNX, and TorchScript
python scripts/export_weights.py --config configs/model3_fpn_attn.yaml --checkpoint runs/train/model3_fpn_attention_best/checkpoints/best_model.pth --output-dir weights

# Compile IEEE Conference Paper PDF
python scripts/compile_paper.py
```
