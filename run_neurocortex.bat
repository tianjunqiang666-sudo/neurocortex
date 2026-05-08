@echo off
chcp 65001 >nul
title NeuroCortex AI - Startup Launcher
echo ==================================================
echo   NeuroCortex AI v2.0 - Startup Launcher
echo ==================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add to PATH.
    pause
    exit /b
)

:: Start API
echo [1/3] Starting API Service (Port 8000)...
start /b uvicorn neurocortex.api.server:app --port 8000 --log-level info

:: Wait
timeout /t 5 >nul

:: Start Dashboard
echo [2/3] Starting Dashboard (Streamlit)...
start /b streamlit run neurocortex/ui/config_dashboard.py

:: Tips
echo [3/3] Ready!
echo.
echo --------------------------------------------------
echo   - Dashboard: http://localhost:8501
echo   - Visualizer: http://localhost:8000/visualizer
echo   - CLI Chat: python -m neurocortex.main
echo --------------------------------------------------
echo.
echo Press any key to stop services...
pause >nul

:: Cleanup
taskkill /f /im uvicorn.exe >nul 2>&1
taskkill /f /im streamlit.exe >nul 2>&1
echo Stopped.
