@echo off
REM ---------------------------------------------------------------------
REM AI Semiconductor Process Optimizer - one-click launcher (Windows)
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] creating virtual environment .venv ...
    python -m venv .venv || goto :error
)

echo [2/4] installing requirements ...
call .venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt || goto :error

if not exist "models\surrogate_bundle.joblib" (
    echo [3/4] training surrogate models ^(first run only, ~10-20 min^) ...
    call .venv\Scripts\python.exe -W ignore train_model.py || goto :error
) else (
    echo [3/4] model artifact found - skipping training
)

echo [4/4] starting server at http://127.0.0.1:8000
start "" http://127.0.0.1:8000
call .venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
goto :eof

:error
echo.
echo *** failed. See the message above. ***
pause
