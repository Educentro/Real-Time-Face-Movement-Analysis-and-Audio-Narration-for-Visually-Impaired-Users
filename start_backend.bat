@echo off
setlocal
cd /d "%~dp0"
echo Starting Flask backend...
python app.py
echo.
echo Backend stopped. Press any key to close.
pause >nul
