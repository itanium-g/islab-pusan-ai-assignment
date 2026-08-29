@echo off
echo ========================================================
echo Initializing Git LFS for Large Model Weights & Data
echo ========================================================

REM Check if git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in PATH.
    exit /b 1
)

REM Check if git lfs is available
git lfs version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git LFS is not installed. Please install it from https://git-lfs.github.com/
    exit /b 1
)

echo Initializing Git LFS...
git lfs install

echo Ensuring tracking attributes are applied...
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.zip"
git lfs track "*.tar.gz"

echo Adding .gitattributes...
git add .gitattributes

echo ========================================================
echo Git LFS configured successfully!
echo To pull all LFS artifacts after cloning:
echo   git lfs pull
echo ========================================================
