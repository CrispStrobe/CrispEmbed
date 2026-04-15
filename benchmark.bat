@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   CrispEmbed Quick Benchmark
echo ============================================

:: Find binary (prefer GPU builds, search recursively)
set "BIN="
for %%d in (build-cuda build-vulkan build) do (
    if exist "%%d\crispembed.exe" (
        if not defined BIN set "BIN=%%d\crispembed.exe"
    )
    if not defined BIN (
        if exist "%%d\Release\crispembed.exe" set "BIN=%%d\Release\crispembed.exe"
    )
    if not defined BIN (
        if exist "%%d\bin\crispembed.exe" set "BIN=%%d\bin\crispembed.exe"
    )
)

:: Also search with where
if not defined BIN (
    for /f "delims=" %%f in ('where /r . crispembed.exe 2^>nul') do (
        if not defined BIN set "BIN=%%f"
    )
)

if not defined BIN (
    echo [ERROR] No crispembed.exe found. Build first with:
    echo   build-windows.bat   (CPU only)
    echo   build-cuda.bat      (NVIDIA GPU)
    echo   build-vulkan.bat    (Vulkan GPU)
    exit /b 1
)

echo Binary: !BIN!

:: Find server
set "SRV="
for %%d in (build-cuda build-vulkan build) do (
    if exist "%%d\crispembed-server.exe" (
        if not defined SRV set "SRV=%%d\crispembed-server.exe"
    )
    if not defined SRV (
        if exist "%%d\Release\crispembed-server.exe" set "SRV=%%d\Release\crispembed-server.exe"
    )
)

:: Resolve model argument
set "MODEL=%~1"

:: If no argument, try to find a local .gguf, else use auto-download name
if "%MODEL%"=="" (
    for %%f in (*.gguf) do (
        set "MODEL=%%f"
        goto :have_model
    )
    for /r models %%f in (*.gguf) do (
        set "MODEL=%%f"
        goto :have_model
    )
    :: No local file — use model name for auto-download
    set "MODEL=all-MiniLM-L6-v2"
    echo [INFO] No local .gguf found. Will auto-download: !MODEL!
)

:: Check if MODEL is a file path or a model name
if exist "!MODEL!" goto :have_model

:: Not a file — might be a model name, let CrispEmbed auto-download it
echo [INFO] Model "!MODEL!" is not a local file. CrispEmbed will auto-download.

:have_model
echo Model:  !MODEL!
echo.

:: Smoke test
echo [1/2] Smoke test...
!BIN! -m "!MODEL!" "hello" 2>nul | findstr /r "." >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Model failed to load or produce output.
    echo   Check: !BIN! -m "!MODEL!" "hello"
    exit /b 1
)
echo   OK

:: Try PowerShell benchmark
echo [2/2] Running benchmark...
echo.
powershell -ExecutionPolicy Bypass -File benchmark.ps1 -Model "!MODEL!" -NRuns 50

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FALLBACK] PowerShell benchmark failed. Running simple CLI timing...
    echo.
    powershell -Command "$n=20; $sw=[Diagnostics.Stopwatch]::StartNew(); for($i=0;$i -lt $n;$i++){$null = & '!BIN!' -m '!MODEL!' 'The quick brown fox' 2>$null}; $sw.Stop(); $ms=$sw.ElapsedMilliseconds/$n; Write-Host ('  CLI: {0:F1}ms/text  {1:F0} texts/s (includes model load)' -f $ms, (1000/$ms))"
)
