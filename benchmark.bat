@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   CrispEmbed Quick Benchmark
echo ============================================

:: Find binary — simple sequential checks
set "BIN="
if exist "build-cuda\crispembed.exe" set "BIN=build-cuda\crispembed.exe"
if "!BIN!"=="" if exist "build-vulkan\crispembed.exe" set "BIN=build-vulkan\crispembed.exe"
if "!BIN!"=="" if exist "build\crispembed.exe" set "BIN=build\crispembed.exe"
if "!BIN!"=="" if exist "build\Release\crispembed.exe" set "BIN=build\Release\crispembed.exe"
if "!BIN!"=="" if exist "build-cuda\Release\crispembed.exe" set "BIN=build-cuda\Release\crispembed.exe"

if "!BIN!"=="" (
    echo [ERROR] No crispembed.exe found. Build first with:
    echo   build-windows.bat   (CPU only)
    echo   build-cuda.bat      (NVIDIA GPU)
    echo   build-vulkan.bat    (Vulkan GPU)
    exit /b 1
)

echo Binary: !BIN!

:: Resolve model argument
set "MODEL=%~1"

if "!MODEL!"=="" (
    :: Try to find a local .gguf
    for %%f in (*.gguf) do (
        set "MODEL=%%f"
        goto :have_model
    )
    :: No local file — default to auto-download
    set "MODEL=all-MiniLM-L6-v2"
    echo [INFO] No .gguf found locally. Will auto-download: !MODEL!
)

:have_model
echo Model:  !MODEL!
echo.

:: Smoke test
echo [1/2] Smoke test...
!BIN! -m "!MODEL!" "hello" 2>nul | findstr /r "." >nul
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Model failed to produce output.
    echo   Try: !BIN! -m "!MODEL!" "hello"
    exit /b 1
)
echo   OK
echo.

:: Benchmark
echo [2/2] Running benchmark...
echo.
powershell -ExecutionPolicy Bypass -File benchmark.ps1 -Model "!MODEL!" -NRuns 50

if !ERRORLEVEL! neq 0 (
    echo.
    echo [FALLBACK] PowerShell failed. Simple CLI timing:
    powershell -Command "$n=20; $sw=[Diagnostics.Stopwatch]::StartNew(); for($i=0;$i -lt $n;$i++){$null = & '!BIN!' -m '!MODEL!' 'The quick brown fox' 2>$null}; $sw.Stop(); $ms=$sw.ElapsedMilliseconds/$n; Write-Host ('  {0:F1}ms/text  {1:F0} texts/s' -f $ms, (1000/$ms))"
)
