#!/bin/bash
# CrispEmbed macOS Build Script
#
# Usage:
#   ./build-macos.sh              # Metal + Accelerate (default)
#   ./build-macos.sh --cpu        # CPU only, no Metal
#   ./build-macos.sh --clean      # Remove build/ first
#   ./build-macos.sh --shared     # Also build shared lib for Python
#   ./build-macos.sh -- -DFOO=BAR # Extra cmake flags after --

set -e

BUILD_DIR="build"
METAL=ON
SHARED=OFF
CLEAN=false
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)    METAL=OFF; shift ;;
        --clean)  CLEAN=true; shift ;;
        --shared) SHARED=ON; shift ;;
        -j)       JOBS="$2"; shift 2 ;;
        --)       shift; CMAKE_EXTRA=("$@"); break ;;
        *)        CMAKE_EXTRA+=("$1"); shift ;;
    esac
done

LABEL="Metal"
[ "$METAL" = "OFF" ] && LABEL="CPU"

echo "============================================"
echo "  CrispEmbed - macOS Build ($LABEL)"
echo "============================================"

# Check ggml submodule
if [ ! -f "ggml/CMakeLists.txt" ]; then
    echo "[INFO] Initializing ggml submodule..."
    git submodule update --init --recursive
fi

# Check cmake
if ! command -v cmake &>/dev/null; then
    echo "[ERROR] cmake not found. Install via: brew install cmake"
    exit 1
fi

# Clean if requested
if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "[INFO] Cleaning $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

# Configure
echo "[INFO] Configuring..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL="$METAL" \
    -DCRISPEMBED_BUILD_SHARED="$SHARED" \
    "${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}"

# Build
echo "[INFO] Building CrispEmbed..."
cmake --build "$BUILD_DIR" -j"$JOBS"

echo ""
echo "[SUCCESS] Build complete!"
echo "  CLI:      $BUILD_DIR/crispembed"
echo "  Server:   $BUILD_DIR/crispembed-server"
echo "  Quantize: $BUILD_DIR/crispembed-quantize"
echo ""
echo "Usage: $BUILD_DIR/crispembed -m all-MiniLM-L6-v2 \"Hello world\""
