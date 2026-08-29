@echo off
echo ========================================================
echo Launching Multi-GPU DistributedDataParallel (DDP) Training
echo ========================================================

set NUM_GPUS=2
set CONFIG=configs/model3_fpn_attn.yaml

if not "%1"=="" set NUM_GPUS=%1
if not "%2"=="" set CONFIG=%2

echo Number of GPUs: %NUM_GPUS%
echo Config File   : %CONFIG%

REM Use torchrun to launch DDP training
torchrun --nproc_per_node=%NUM_GPUS% train.py --config %CONFIG% --ddp

if %errorlevel% neq 0 (
    echo Multi-GPU training encountered an error.
    exit /b 1
)

echo ========================================================
echo Multi-GPU training finished successfully!
echo ========================================================
