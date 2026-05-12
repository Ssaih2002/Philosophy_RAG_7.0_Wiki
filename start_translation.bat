@echo off
setlocal

REM Force UTF-8 codepage to avoid mojibake.
chcp 65001 >nul

REM Ensure script runs from project root.
cd /d "%~dp0"

REM Keep Hugging Face cache inside the project, same as the main launcher.
set "HF_HOME=%~dp0data\hf_cache"
set "TRANSFORMERS_CACHE=%HF_HOME%"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

echo [1/5] Checking virtual environment...
set "CREATED_VENV=0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found, creating with Python 3.11...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.11 -m venv .venv 2>nul
        if errorlevel 1 py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto :error
    set "CREATED_VENV=1"
)

set "PYTHON_EXE=.venv\Scripts\python.exe"

echo [2/5] Checking dependencies (requirements.txt)...
set "REQ_HASH_FILE=.venv\.req_hash"
set "CUR_REQ_HASH="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 'requirements.txt').Hash"`) do set "CUR_REQ_HASH=%%H"
if "%CUR_REQ_HASH%"=="" (
    echo Failed to compute requirements hash.
    goto :error
)

set "OLD_REQ_HASH="
if exist "%REQ_HASH_FILE%" (
    set /p OLD_REQ_HASH=<"%REQ_HASH_FILE%"
)

if "%CREATED_VENV%"=="1" (
    echo New venv created: will install dependencies.
    goto :install_deps
)
if not "%CUR_REQ_HASH%"=="%OLD_REQ_HASH%" (
    echo requirements.txt changed: will install dependencies.
    goto :install_deps
)

echo Dependencies unchanged: skipping pip install.
goto :after_install

:install_deps
echo Installing dependencies...
"%PYTHON_EXE%" -m ensurepip --upgrade >nul 2>nul
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto :error
>"%REQ_HASH_FILE%" echo %CUR_REQ_HASH%

:after_install
if exist "tools\ensure_torch_accel.py" (
    "%PYTHON_EXE%" "tools\ensure_torch_accel.py"
)

echo [3/5] Starting backend (uvicorn)...
if not exist "run_backend.bat" (
    echo run_backend.bat not found.
    goto :error
)
REM Translation does not need the RAG reranker warmup; skipping it makes backend startup faster.
start "Philosophy Translation Backend" cmd /k "cd /d ""%~dp0"" && set PRELOAD_RERANKER=0 && call run_backend.bat"

echo [4/5] Starting static translation frontend server (http.server on port 5173)...
if not exist "run_frontend_static.bat" (
    echo run_frontend_static.bat not found.
    goto :error
)
start "Philosophy Translation Frontend (static)" cmd /k "cd /d ""%~dp0"" && call run_frontend_static.bat"

echo [5/5] Opening translation workbench...
timeout /t 2 /nobreak >nul
if exist "translation_frontend.html" (
    start "" "http://127.0.0.1:5173/translation_frontend.html?v=%RANDOM%%RANDOM%"
) else (
    echo translation_frontend.html not found.
    goto :error
)

echo.
echo Translation startup complete.
echo - API backend:       http://127.0.0.1:8000
echo - Translation page:  http://127.0.0.1:5173/translation_frontend.html
echo.
echo Two extra windows were opened: Backend ^(uvicorn^) and Translation Frontend ^(http.server^).
echo Close those windows to stop the servers.
exit /b 0

:error
echo.
echo Translation startup failed. Please check the error log above.
pause
exit /b 1
