@echo off

REM This checks the Python version because the .keras model uses python 3.10.18
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.10.18 first.
    pause
    exit /b
)

REM python -m venv wzj_cancer_predictor_env
REM call wzj_env\Scripts\activate.bat

REM This section installs the required libraries for the .keras model to work, specifically tensorflow==2.19.0 keras==3.10.0
python -m pip install --upgrade pip
pip install flask tensorflow==2.19.0 keras==3.10.0 opencv-python-headless numpy

REM This starts the python script
python app.py

pause
