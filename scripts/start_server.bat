@echo off
echo Starting AI Horse Racing Predictor System...
echo.
cd /d "C:\AI_Horse_Racing"
call venv\Scripts\activate.bat
python horse_racing_server.py
pause
