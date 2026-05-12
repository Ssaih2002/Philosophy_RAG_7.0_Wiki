@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Virtual environment python not found: %PYTHON_EXE%
    exit /b 1
)
echo Philosophy RAG: static frontend on http://127.0.0.1:5173/  (Ctrl+C to stop)
"%PYTHON_EXE%" -m http.server 5173 --bind 127.0.0.1
