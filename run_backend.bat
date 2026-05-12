@echo off
setlocal

REM Force UTF-8 codepage to avoid mojibake.
chcp 65001 >nul

cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment python not found: %PYTHON_EXE%
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
REM NOTE:
REM - --reload restarts worker on file changes (dev-only) and causes model cold starts.
REM - default: reload is OFF so models stay warm in memory.
REM - enable: set UVICORN_RELOAD=1
if "%UVICORN_RELOAD%"=="1" (
    "%PYTHON_EXE%" -m uvicorn web_app:app --reload --host 127.0.0.1 --port 8000
) else (
    "%PYTHON_EXE%" -m uvicorn web_app:app --host 127.0.0.1 --port 8000
)
