@echo off
echo ============================================
echo AI Horse Racing Predictor System - Setup
echo ============================================
echo.
echo This will install all required components...
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Installing Python 3.9...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe' -OutFile 'python_installer.exe'"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
) else (
    echo Python is already installed.
)

echo.
echo Creating project directory structure...
mkdir "C:\AI_Horse_Racing" 2>nul
mkdir "C:\AI_Horse_Racing\data" 2>nul
mkdir "C:\AI_Horse_Racing\models" 2>nul
mkdir "C:\AI_Horse_Racing\temp" 2>nul

echo.
echo Installing Python packages...
pip install --upgrade pip
pip install numpy pandas scikit-learn flask flask-cors tensorflow-cpu joblib requests beautifulsoup4 lxml

echo.
echo Creating virtual environment...
python -m venv "C:\AI_Horse_Racing\venv"

echo.
echo Downloading sample data and models...
cd "C:\AI_Horse_Racing"
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/datasets/horse-racing/master/data/races.csv' -OutFile 'data\races.csv'"
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/datasets/horse-racing/master/data/horses.csv' -OutFile 'data\horses.csv'"

echo.
echo Creating main application files...
REM This will create the single HTML page and backend server
copy setup_system.bat create_app.bat

echo.
echo Setup complete!
echo.
echo To start the system, run:
echo 1. Double-click "start_server.bat" (will be created next)
echo 2. Open browser to: http://localhost:5000
echo.
pause
