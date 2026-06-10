#!/bin/bash
# CrispEmbed WASM Build Script — math OCR for browser use.
#
# Usage:
#   ./build-wasm.sh                    # default build
#   ./build-wasm.sh --clean            # remove build-wasm/ first
#   ./build-wasm.sh --simd             # enable WASM SIMD128 (default: on)
#   ./build-wasm.sh --no-simd          # disable WASM SIMD128
#   ./build-wasm.sh -- -DFOO=BAR      # extra cmake flags
#
# Prerequisites:
#   - Emscripten SDK activated (source emsdk_env.sh)
#
# Output:
#   build-wasm/crispembed_ocr.js       Emscripten JS loader
#   build-wasm/crispembed_ocr.wasm     WebAssembly binary

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="build-wasm"
CLEAN=false
SIMD=ON
CMAKE_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)    CLEAN=true; shift ;;
        --simd)     SIMD=ON; shift ;;
        --no-simd)  SIMD=OFF; shift ;;
        --)         shift; CMAKE_EXTRA=("$@"); break ;;
        *)          CMAKE_EXTRA+=("$1"); shift ;;
    esac
done

# Check emcc is available
if ! command -v emcc &>/dev/null; then
    echo "[ERROR] emcc not found. Activate Emscripten SDK first:"
    echo "  source /mnt/volume1/emsdk/emsdk_env.sh"
    exit 1
fi

echo "============================================"
echo "  CrispEmbed - WASM Build (math OCR)"
echo "============================================"

# Check ggml submodule
if [ ! -f "$SCRIPT_DIR/ggml/CMakeLists.txt" ]; then
    echo "[INFO] Initializing ggml submodule..."
    cd "$SCRIPT_DIR" && git submodule update --init --recursive
fi

# Clean if requested
if [ "$CLEAN" = true ] && [ -d "$SCRIPT_DIR/$BUILD_DIR" ]; then
    echo "[INFO] Cleaning $BUILD_DIR..."
    rm -rf "$SCRIPT_DIR/$BUILD_DIR"
fi

# Exported C functions (with _ prefix per Emscripten convention)
EXPORTED_FUNCS="[\
'_wasm_ocr_version',\
'_wasm_ocr_init',\
'_wasm_ocr_recognize_gray',\
'_wasm_ocr_recognize',\
'_wasm_ocr_free',\
'_malloc',\
'_free',\
'_main'\
]"

EXPORTED_RUNTIME="[\
'ccall','cwrap','FS','MEMFS','getValue','setValue','UTF8ToString','stringToUTF8','lengthBytesUTF8'\
]"

# SIMD flags
SIMD_FLAGS=""
if [ "$SIMD" = "ON" ]; then
    SIMD_FLAGS="-msimd128"
    echo "[INFO] WASM SIMD128 enabled"
fi

# Configure
echo "[INFO] Configuring with emcmake..."
cd "$SCRIPT_DIR"
emcmake cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_OPENMP=OFF \
    -DCRISPEMBED_BUILD_SHARED=OFF \
    -DCRISPEMBED_WASM=ON \
    -DCMAKE_C_FLAGS="$SIMD_FLAGS -pthread" \
    -DCMAKE_CXX_FLAGS="$SIMD_FLAGS -pthread" \
    -DCMAKE_EXE_LINKER_FLAGS="\
-sEXPORTED_FUNCTIONS=$EXPORTED_FUNCS \
-sEXPORTED_RUNTIME_METHODS=$EXPORTED_RUNTIME \
-sALLOW_MEMORY_GROWTH=1 \
-sINITIAL_MEMORY=67108864 \
-sSTACK_SIZE=1048576 \
-sMODULARIZE=1 \
-sEXPORT_NAME=CrispEmbedOCR \
-sENVIRONMENT=web \
-sFILESYSTEM=1 \
-sWASM_BIGINT=1 \
-sNO_EXIT_RUNTIME=1 \
-pthread \
-sPTHREAD_POOL_SIZE=0 \
$SIMD_FLAGS \
" \
    "${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}"

# Build
echo "[INFO] Building..."
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || echo 4) --target crispembed-wasm

echo ""
echo "[SUCCESS] WASM build complete!"
echo "  JS loader: $BUILD_DIR/crispembed_ocr.js"
echo "  WASM:      $BUILD_DIR/crispembed_ocr.wasm"
ls -lh "$BUILD_DIR/crispembed_ocr.js" "$BUILD_DIR/crispembed_ocr.wasm" 2>/dev/null || true
echo ""
echo "Copy to CrispCalc:  cp $BUILD_DIR/crispembed_ocr.{js,wasm} ../CrispCalc/web/"
