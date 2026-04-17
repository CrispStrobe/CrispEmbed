#!/bin/bash
# Build CrispEmbed for iOS (xcframework with Metal GPU acceleration).
#
# Prerequisites:
#   - Xcode with command-line tools
#   - CMake 3.14+
#
# Usage:
#   ./build-ios.sh              # Build xcframework (arm64 device + simulator)
#   ./build-ios.sh --device     # Device only (arm64)
#   ./build-ios.sh --simulator  # Simulator only (arm64)
#
# Output: build-ios/CrispEmbed.xcframework/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-ios"
DEVICE_ONLY=0
SIM_ONLY=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device)    DEVICE_ONLY=1 ;;
        --simulator) SIM_ONLY=1 ;;
        --clean)     rm -rf "$BUILD_DIR"; echo "Cleaned build-ios/"; exit 0 ;;
        --help)      echo "Usage: $0 [--device|--simulator|--clean]"; exit 0 ;;
    esac
    shift
done

CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCRISPEMBED_BUILD_SHARED=OFF
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
    -DGGML_BLAS=OFF
    -DBUILD_SHARED_LIBS=OFF
)

# --- Device (arm64) ---
if [ $SIM_ONLY -eq 0 ]; then
    echo "=== Building for iOS device (arm64) ==="
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR/device" \
        -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        "${CMAKE_COMMON[@]}"
    cmake --build "$BUILD_DIR/device" --config Release -- -quiet
fi

# --- Simulator (arm64 for Apple Silicon Macs) ---
if [ $DEVICE_ONLY -eq 0 ]; then
    echo "=== Building for iOS simulator (arm64) ==="
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR/simulator" \
        -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT=iphonesimulator \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        "${CMAKE_COMMON[@]}"
    cmake --build "$BUILD_DIR/simulator" --config Release -- -quiet
fi

# --- Create xcframework ---
if [ $DEVICE_ONLY -eq 0 ] && [ $SIM_ONLY -eq 0 ]; then
    echo "=== Creating xcframework ==="
    rm -rf "$BUILD_DIR/CrispEmbed.xcframework"
    xcodebuild -create-xcframework \
        -library "$BUILD_DIR/device/Release-iphoneos/libcrispembed-static.a" \
        -headers "$SCRIPT_DIR/src/crispembed.h" \
        -library "$BUILD_DIR/simulator/Release-iphonesimulator/libcrispembed-static.a" \
        -headers "$SCRIPT_DIR/src/crispembed.h" \
        -output "$BUILD_DIR/CrispEmbed.xcframework"
    echo ""
    echo "=== Built: $BUILD_DIR/CrispEmbed.xcframework ==="
fi

echo "Done."
