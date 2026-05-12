@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found: %PYTHON_EXE%
    echo Please run start_app.bat first to create .venv.
    pause
    exit /b 1
)

echo [1/2] Ingest single-doc temp profile (tmp) from data_single ...
"%PYTHON_EXE%" ingest_single_tmp.py --profile tmp --data-dir data_single --embedding-model BAAI/bge-m3
if errorlevel 1 goto :error

echo [2/2] Merge tmp profile into quality profile ...
"%PYTHON_EXE%" merge_profile.py tmp quality
if errorlevel 1 goto :error

echo.
echo Done. tmp profile has been merged into quality.
echo You can now ask questions in quality profile.
pause
exit /b 0

:error
echo.
echo Failed. Please review errors above.
pause
exit /b 1

