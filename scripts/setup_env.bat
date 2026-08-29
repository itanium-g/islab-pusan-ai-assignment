@echo off
echo ========================================================
echo Setting up Python Virtual Environment for Drone Detector
echo ========================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist "venv" (
    echo Creating virtual environment in .\venv ...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment 'venv' already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt

echo ========================================================
echo Setup complete! To activate your environment in terminal:
echo   .\venv\Scripts\activate
echo ========================================================
