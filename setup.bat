@echo off
rem VocalTrace - one-shot environment bootstrap (Windows).
rem Creates a virtual environment, installs core dependencies, smoke test.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo == VocalTrace setup ==

rem --- 0. Python version ----------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo error: python not found on PATH. Install Python 3.9+ from python.org first.
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info ^>= (3,9) else 1)"
if errorlevel 1 (
    echo error: Python 3.9+ required.
    exit /b 1
)

rem --- 1. Virtual environment -----------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo -- reusing existing .venv
) else (
    echo -- creating virtual environment in .venv
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
set "PY=.venv\Scripts\python"

rem --- 2. Dependencies --------------------------------------------------------
echo -- installing dependencies (this can take a few minutes)
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

rem --- 3. Smoke test ----------------------------------------------------------
echo -- running smoke test
"%PY%" -c "import numpy as np; from vocaltrace.features import FeatureExtractor; fe = FeatureExtractor(); v = fe.to_flat_vector(fe.extract_all(np.random.default_rng(42).normal(0, 0.1, 16000).astype('float32'))); print('VocalTrace ready, flat vector dim =', v.size)"
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Try:
echo   .venv\Scripts\python main.py detect --file data\fake\sample1.wav
echo   .venv\Scripts\python -m pytest tests/ -q
exit /b 0
