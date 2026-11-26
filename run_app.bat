@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY_CMD="

:: 1. Try Windows Python Launcher (py)
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 -c "import sys" >nul 2>&1
    if !errorlevel!==0 set "PY_CMD=py -3"
)

:: 2. Try python in PATH (ignore Microsoft Store stub)
if not defined PY_CMD (
    python -c "import sys; print(sys.executable)" >nul 2>&1
    if !errorlevel!==0 set "PY_CMD=python"
)

:: 3. Check standard per-user Python install folders
if not defined PY_CMD (
    for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" set PY_CMD="%%D\python.exe"
    )
)

:: 4. If still not found, offer automatic install via winget
if not defined PY_CMD (
    echo Python 3 was not found on this PC.
    where winget >nul 2>&1
    if %errorlevel%==0 (
        echo Installing Python 3.12 via winget...
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
            if exist "%%D\python.exe" set PY_CMD="%%D\python.exe"
        )
    )
)

if not defined PY_CMD (
    echo.
    echo Could not find or auto-install Python 3.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo IMPORTANT: Check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

:: Check FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    if not exist "ffmpeg.exe" (
        echo FFmpeg not found. Attempting install via winget...
        where winget >nul 2>&1
        if %errorlevel%==0 (
            winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        ) else (
            echo Note: Install FFmpeg or place ffmpeg.exe in this folder before trimming.
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment and installing dependencies...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    ".venv\Scripts\pip.exe" install --upgrade pip
    ".venv\Scripts\pip.exe" install -r requirements.txt
)

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" app.py
) else (
    echo Setup failed: .venv\Scripts\pythonw.exe was not created.
    pause
)
