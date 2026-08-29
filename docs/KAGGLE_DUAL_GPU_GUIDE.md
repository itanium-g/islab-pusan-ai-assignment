# ☁️ Kaggle Dual-GPU Execution Guide

This guide details how to run the DroneNet-FPN-Attention training, evaluation, and paper compilation pipeline on Kaggle using Dual NVIDIA Tesla T4 GPUs (DDP) with zero manual dataset uploading.

---

## 1. Automated Dataset Download

The pipeline automatically fetches the full 2,400-frame AirSim UAV dataset directly from Google Drive:
- **Google Drive File ID**: `19L9yUP62xMESJMw6srf5HGcL8s5b0gv8`
- The script uses `gdown` with chunked unzipping to `/tmp/obj_det_base/`, ensuring the dataset is ephemeral and does not inflate the downloadable Kaggle output archive (< 15 MB).

---

## 2. Multi-GPU DistributedDataParallel (DDP) Architecture

The training engine initializes multi-GPU synchronization via `torchrun` and PyTorch DDP:
- **Backend**: `nccl` (NVIDIA Collective Communications Library)
- **World Size**: 2 GPUs (Dual Tesla T4)
- **Batch Size per GPU**: 8 (Effective global batch size: 16)
- **Automatic Mixed Precision (AMP)**: `torch.cuda.amp.GradScaler` enabled
- **Unused Parameter Discovery**: `find_unused_parameters=True` enabled to accommodate multi-scale FPN branches where target scales vary per batch.

```
                  +-----------------------------------+
                  |         torchrun launcher         |
                  +-----------------+-----------------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
        +--------------------+            +--------------------+
        | Rank 0 (Tesla T4)  |            | Rank 1 (Tesla T4)  |
        | Batch: 8 images    |            | Batch: 8 images    |
        | Master Logger (TB) |            | Worker Process     |
        +---------+----------+            +---------+----------+
                  |                                 |
                  +<====== NCCL All-Reduce ========>+
```

---

## 3. Kaggle Metadata & Accelerator Configuration

The kernel configuration is managed via `notebooks/kernel-metadata.json`:

```json
{
  "id": "itanium/drone-detection-islab-dual-gpu",
  "title": "Drone Detection ISLab Dual GPU",
  "code_file": "kaggle_training.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

> **Note on Accelerator**: Setting `"machine_shape": "NvidiaTeslaT4"` guarantees that Kaggle allocates **Dual T4 GPUs** upon every push.

---

## 4. Kaggle CLI Commands & Workflow

### 4.1 Authentication
If using API Token (`kaggle.json`):
Place `kaggle.json` inside `~/.kaggle/kaggle.json` (or `C:\Users\<USER>\.kaggle\kaggle.json`).

### 4.2 Push & Execute
```powershell
# Rebuild the standalone notebook
python scripts/build_kaggle_notebook.py

# Push and launch execution on Kaggle
.\venv\Scripts\kaggle.exe kernels push -p notebooks --accelerator NvidiaTeslaT4
```

### 4.3 Check Live Status
```powershell
.\venv\Scripts\kaggle.exe kernels status itanium/drone-detection-islab-dual-gpu
```

### 4.4 Download Output Artifacts
Once status reaches `"complete"`, download all trained model weights, ONNX exports, evaluation curves, and the compiled IEEE Paper PDF:
```powershell
.\venv\Scripts\kaggle.exe kernels output itanium/drone-detection-islab-dual-gpu -p kaggle_output
```

---

## 5. Generated Output Artifacts

Upon completion, Kaggle produces the following lightweight outputs (< 15 MB total):
- `weights/DroneNet-FPN-Attention_inference.pth` (Stripped inference weights, ~16.6 MB)
- `weights/DroneNet-FPN-Attention.onnx` (ONNX computational graph)
- `weights/DroneNet-FPN-Attention.torchscript.pt` (TorchScript tracing)
- `runs/eval/` (Evaluation metrics, PR curves, Confusion Matrices)
- `paper/Drone_Detection_Paper.pdf` (Compiled IEEE Conference Paper)
