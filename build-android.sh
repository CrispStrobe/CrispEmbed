#!/bin/bash
# Build CrispEmbed for Android (NDK cross-compilation).
#
# Prerequisites:
#   - Android NDK (set ANDROID_NDK_HOME or NDK_HOME)
#   - CMake 3.14+
#
# Usage:
#   ./build-android.sh                      # All ABIs (arm64-v8a, armeabi-v7a, x86_64)
#   ./build-android.sh --abi arm64-v8a      # Single ABI
#   ./build-android.sh --vulkan             # With Vulkan GPU acceleration
#
# Output: build-android/<abi>/libcrispembed.so

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-android"
API_LEVEL=24  # Android 7.0 (minimum for Vulkan)
VULKAN=OFF
ABIS=("arm64-v8a" "armeabi-v7a" "x86_64")

# Find NDK
NDK="${ANDROID_NDK_HOME:-${NDK_HOME:-${ANDROID_HOME}/ndk-bundle}}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --abi)     ABIS=("$2"); shift ;;
        --vulkan)  VULKAN=ON ;;
        --api)     API_LEVEL="$2"; shift ;;
        --ndk)     NDK="$2"; shift ;;
        --clean)   rm -rf "$BUILD_DIR"; echo "Cleaned build-android/"; exit 0 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --abi <abi>     Target ABI (arm64-v8a, armeabi-v7a, x86_64)"
            echo "  --vulkan        Enable Vulkan GPU acceleration"
            echo "  --api <level>   Android API level (default: $API_LEVEL)"
            echo "  --ndk <path>    Android NDK path"
            echo "  --clean         Remove build directory"
            exit 0 ;;
    esac
    shift
done

TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN" ]; then
    echo "[ERROR] Android NDK toolchain not found at: $TOOLCHAIN"
    echo "Set ANDROID_NDK_HOME to your NDK installation directory."
    exit 1
fi
echo "NDK: $NDK"
echo "API: $API_LEVEL"
echo "Vulkan: $VULKAN"

for ABI in "${ABIS[@]}"; do
    echo ""
    echo "=== Building for $ABI ==="
    ABI_DIR="$BUILD_DIR/$ABI"

    cmake -S "$SCRIPT_DIR" -B "$ABI_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_NATIVE_API_LEVEL="$API_LEVEL" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCRISPEMBED_BUILD_SHARED=ON \
        -DGGML_VULKAN="$VULKAN" \
        -DGGML_BLAS=OFF \
        -DGGML_LLAMAFILE=OFF

    cmake --build "$ABI_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    # Show output
    if [ -f "$ABI_DIR/libcrispembed.so" ]; then
        SIZE=$(ls -lh "$ABI_DIR/libcrispembed.so" | awk '{print $5}')
        echo "  Built: $ABI_DIR/libcrispembed.so ($SIZE)"
    fi
done

echo ""
echo "=== Android build complete ==="
echo "Libraries:"
for ABI in "${ABIS[@]}"; do
    ls -lh "$BUILD_DIR/$ABI/libcrispembed.so" 2>/dev/null || true
done
echo ""
echo "To use in Flutter/Android:"
echo "  Copy build-android/<abi>/libcrispembed.so to"
echo "  android/app/src/main/jniLibs/<abi>/libcrispembed.so"
