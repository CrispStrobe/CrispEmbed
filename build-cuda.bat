@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   CrispEmbed - Windows Build (CUDA GPU)
echo ============================================

:: Check ggml submodule
if not exist "ggml\CMakeLists.txt" (
    echo [INFO] Initializing ggml submodule...
    git submodule update --init --recursive
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to initialize submodule. Run: git submodule update --init --recursive
        exit /b 1
    )
)

:: Find vswhere.exe
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!vswhere!" (
    echo [ERROR] vswhere.exe not found. Please install Visual Studio 2022 Build Tools.
    exit /b 1
)

:: Find VS Installation Path
for /f "usebackq tokens=*" %%i in (`"!vswhere!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "vs_path=%%i"
)

if not defined vs_path (
    echo [ERROR] Visual Studio C++ build tools not found.
    exit /b 1
)

:: Check for CUDA
where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] nvcc not found. Please install CUDA Toolkit.
    echo   Download: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)

:: Initialize MSVC Environment
set "vcvars=!vs_path!\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!vcvars!" (
    echo [ERROR] vcvars64.bat not found at !vcvars!
    exit /b 1
)

echo [INFO] Initializing MSVC environment...
call "!vcvars!"

:: Configure
:: Clean old shared lib build if present (avoids ggml.dll not found errors)
if exist "build-cuda\ggml.dll" (
    echo [INFO] Cleaning old shared lib build...
    rmdir /s /q build-cuda 2>nul
)

echo [INFO] Configuring with CUDA + Ninja...
cmake -G Ninja -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_LLAMAFILE=ON -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF %*

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

:: Build
echo [INFO] Building CrispEmbed with CUDA...
cmake --build build-cuda --config Release

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo [SUCCESS] CUDA build complete!
echo   CLI:      build-cuda\crispembed.exe
echo   Server:   build-cuda\crispembed-server.exe
echo   Quantize: build-cuda\crispembed-quantize.exe
echo.
echo Quick test:
echo   build-cuda\crispembed.exe -m model.gguf "Hello world"
echo.
echo Benchmark:
echo   benchmark.bat model.gguf
echo   powershell -File benchmark.ps1 -Model model.gguf
