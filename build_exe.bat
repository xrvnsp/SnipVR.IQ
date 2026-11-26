@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
.venv\Scripts\pip install -r requirements.txt pyinstaller
.venv\Scripts\pyinstaller --noconfirm --onefile --windowed --name "SnipVR.IQ" --icon "app_icon.ico" --add-data "app_icon.ico;." --add-data "default_model.npz;." app.py
echo.
echo Build complete! Find your EXE in dist\SnipVR.IQ.exe
pause
