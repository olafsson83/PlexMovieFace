@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Creating a Python 3.11 virtual environment for this project...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not find Python 3.11 via the "py" launcher.
        echo Install it from https://www.python.org/downloads/release/python-3119/
        echo ^(or run: winget install --id Python.Python.3.11^)
        echo then re-run this file.
        pause
        exit /b 1
    )
)

echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Something went wrong installing dependencies.
    pause
    exit /b 1
)

.venv\Scripts\python.exe setup.py
pause
