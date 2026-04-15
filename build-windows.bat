@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   CrispEmbed - Windows Build (CPU)
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
    echo   Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
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

:: Initialize MSVC Environment
set "vcvars=!vs_path!\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!vcvars!" (
    echo [ERROR] vcvars64.bat not found at !vcvars!
    exit /b 1
)

echo [INFO] Initializing MSVC environment...
call "!vcvars!"

:: Configure
:: Clean old shared lib build if present
if exist "build\ggml.dll" (
    echo [INFO] Cleaning old shared lib build...
    rmdir /s /q build 2>nul
)

echo [INFO] Configuring with Ninja...
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF %*

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed.
    echo   If Ninja is not installed, try: cmake -B build -DCMAKE_BUILD_TYPE=Release %*
    exit /b 1
)

:: Build
echo [INFO] Building CrispEmbed...
cmake --build build --config Release

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo [SUCCESS] Build complete!
echo   CLI:      build\crispembed.exe
echo   Server:   build\crispembed-server.exe
echo   Quantize: build\crispembed-quantize.exe
echo.
echo Usage: build\crispembed.exe -m model.gguf "Hello world"
