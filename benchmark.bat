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
if exist "build\Release\crispembed.exe" if not defined BIN set "BIN=build\Release\crispembed.exe"

if not defined BIN (
    echo [ERROR] No crispembed.exe found. Build first with:
    echo   build-windows.bat   (CPU)
    echo   build-cuda.bat      (NVIDIA GPU)
    echo   build-vulkan.bat    (Vulkan GPU)
    exit /b 1
)

:: Use model from argument or find one
set "MODEL=%~1"
if "%MODEL%"=="" goto :find_model
:: Check if argument is a file
if exist "%MODEL%" goto :have_model
:: Check if it's a name — search for matching .gguf
for %%f in (*%MODEL%*.gguf) do (
    set "MODEL=%%f"
    goto :have_model
)
for %%f in (models\*%MODEL%*.gguf) do (
    set "MODEL=%%f"
    goto :have_model
)
echo [ERROR] Model not found: %MODEL%
echo   Specify a .gguf file path: benchmark.bat path\to\model.gguf
echo   Download models from: https://huggingface.co/cstr
exit /b 1

:find_model
for %%f in (*.gguf) do (
    set "MODEL=%%f"
    goto :have_model
)
for %%f in (models\*.gguf) do (
    set "MODEL=%%f"
    goto :have_model
)
:: No local file found; use auto-download with model name
set "MODEL=all-MiniLM-L6-v2"
echo [INFO] No local .gguf found. Using auto-download: !MODEL!
goto :have_model

:have_model
echo Binary: !BIN!
echo Model:  !MODEL!
echo.

:: Quick smoke test
echo [1/3] Smoke test...
!BIN! -m "!MODEL!" "hello" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Model failed to load. Check the path and try again.
    echo   If you see "ggml.dll not found", rebuild with:
    echo     build-cuda.bat    (includes -DBUILD_SHARED_LIBS=OFF now)
    exit /b 1
)

:: Run PowerShell benchmark
echo [2/3] Running PowerShell benchmark...
powershell -ExecutionPolicy Bypass -File benchmark.ps1 -Model "!MODEL!" -NRuns 50 2>nul

if %ERRORLEVEL% neq 0 (
    echo.
    echo [3/3] PowerShell failed. Running simple CLI timing...
    echo.

    set "TEXT=The quick brown fox jumps over the lazy dog near the river bank"
    echo Warming up...
    for /l %%i in (1,1,3) do !BIN! -m "!MODEL!" "!TEXT!" >nul 2>nul

    :: Use PowerShell for accurate timing since cmd has only second resolution
    powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); for($i=0;$i -lt 20;$i++){$null=& '!BIN!' -m '!MODEL!' '!TEXT!' 2>$null}; $sw.Stop(); Write-Host ('CLI: {0:F1}ms/text  {1:F0} texts/s' -f ($sw.ElapsedMilliseconds/20), (20000/$sw.ElapsedMilliseconds))"
)
