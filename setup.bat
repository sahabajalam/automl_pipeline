@echo off
echo Starting Autonomous ML Pipeline Setup...
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

echo Python detected, installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Setup complete! You can now run:
echo   python main.py --data-path data/sample/titanic.csv --target-column Survived
echo.
echo Or test the system:
echo   python test_pipeline.py
echo.
pause
