@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   CrispEmbed Quick Benchmark
echo ============================================

:: Find binary (prefer GPU builds)
set "BIN="
if exist "build-cuda\crispembed.exe" set "BIN=build-cuda\crispembed.exe"
if exist "build-vulkan\crispembed.exe" if not defined BIN set "BIN=build-vulkan\crispembed.exe"
if exist "build\crispembed.exe" if not defined BIN set "BIN=build\crispembed.exe"

if not defined BIN (
    echo [ERROR] No crispembed.exe found. Build first.
    exit /b 1
)

set "SRV="
if exist "build-cuda\crispembed-server.exe" set "SRV=build-cuda\crispembed-server.exe"
if exist "build-vulkan\crispembed-server.exe" if not defined SRV set "SRV=build-vulkan\crispembed-server.exe"
if exist "build\crispembed-server.exe" if not defined SRV set "SRV=build\crispembed-server.exe"

:: Use model from argument or find one
set "MODEL=%~1"
if "%MODEL%"=="" (
    for %%f in (*.gguf) do (
        set "MODEL=%%f"
        goto :found_model
    )
    for %%f in (models\*.gguf) do (
        set "MODEL=%%f"
        goto :found_model
    )
    echo [ERROR] No .gguf model found. Pass as argument: benchmark.bat model.gguf
    exit /b 1
)
:found_model

echo Binary: !BIN!
echo Model:  !MODEL!
echo.

:: Run benchmark via PowerShell
powershell -ExecutionPolicy Bypass -File benchmark.ps1 -Model "!MODEL!" -NRuns 50

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FALLBACK] PowerShell script failed. Running simple CLI benchmark...
    echo.

    :: Simple timing loop
    set "TEXT=The quick brown fox jumps over the lazy dog"
    echo Warming up...
    !BIN! -m "!MODEL!" "!TEXT!" >nul 2>nul

    echo Running 20 iterations...
    set "START=%TIME%"
    for /l %%i in (1,1,20) do (
        !BIN! -m "!MODEL!" "!TEXT!" >nul 2>nul
    )
    set "END=%TIME%"
    echo Start: !START!
    echo End:   !END!
    echo (Manual timing - calculate ms per text from timestamps)
)
