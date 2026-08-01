#!/bin/bash
# CrispEmbed WASM Build Script — OCR for browser use.
#
# Usage:
#   ./build-wasm.sh                    # default build (single-threaded)
#   ./build-wasm.sh --threads          # multithreaded (requires COOP/COEP headers)
#   ./build-wasm.sh --webgpu           # experimental WebGPU backend (emdawnwebgpu)
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
BUILD_DIR_SET=false
CLEAN=false
SIMD=ON
THREADS=OFF
PROXY=OFF
WEBGPU=OFF
WEBGPU_COMPAT=OFF
CMAKE_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)    CLEAN=true; shift ;;
        --simd)     SIMD=ON; shift ;;
        --no-simd)  SIMD=OFF; shift ;;
        --threads)  THREADS=ON; shift ;;
        --proxy-to-pthread) THREADS=ON; PROXY=ON; shift ;;
        --webgpu)   WEBGPU=ON; shift ;;
        --webgpu-compat) WEBGPU=ON; WEBGPU_COMPAT=ON; shift ;;
        --)         shift; CMAKE_EXTRA=("$@"); break ;;
        *)          CMAKE_EXTRA+=("$1"); shift ;;
    esac
done

# Check emcc is available
if ! command -v emcc &>/dev/null; then
    echo "[ERROR] emcc not found. Activate Emscripten SDK first:"
    echo "  source <path-to-emsdk>/emsdk_env.sh"
    exit 1
fi

THREAD_LABEL="single-threaded"
THREAD_C_FLAGS=""
THREAD_LINK_FLAGS=""
WEBGPU_FLAGS=""
WEBGPU_LINK_FLAGS=""
if [ "$WEBGPU" = "ON" ]; then
    WEBGPU_FLAGS="-DGGML_WEBGPU=ON"
    if [ "$WEBGPU_COMPAT" = "ON" ]; then
        # Asyncify variant for browsers with WebGPU but no JSPI (Safari 26,
        # Firefox). Bigger + slower than JSPI; the demo picks it only when
        # WebAssembly.Suspending is missing.
        WEBGPU_FLAGS="$WEBGPU_FLAGS -DGGML_WEBGPU_JSPI=OFF"
        if [ "$BUILD_DIR_SET" = false ]; then BUILD_DIR="build-wasm-webgpu-compat"; fi
        echo "[INFO] WebGPU compat (Asyncify) variant -> $BUILD_DIR/"
    fi
    # JSPI: every export that can reach GPU work (and therefore suspend)
    # must be wrapped with WebAssembly.promising — list them explicitly.
    # JS callers must use ccall(..., {async:true}) for these (the JS wrapper
    # does; see _acall in wasm/crispembed-ocr.js).
    WEBGPU_LINK_FLAGS="-sJSPI_EXPORTS=[\
'wasm_ocr_init','wasm_ocr_recognize','wasm_ocr_recognize_gray','wasm_ocr_recognize_copy','wasm_ocr_free',\
'wasm_ocr_pipeline_init','wasm_ocr_pipeline_run','wasm_ocr_pipeline_free',\
'wasm_ocr_pipeline_full_init','wasm_ocr_pipeline_full_run','wasm_ocr_pipeline_full_free',\
'wasm_scan_cleanup_init','wasm_scan_cleanup_process','wasm_scan_cleanup_free',\
'wasm_text_det_init','wasm_text_det_run','wasm_text_det_free',\
'wasm_layout_init','wasm_layout_detect','wasm_layout_free',\
'wasm_ocr_render'] \
-sALLOW_MEMORY_GROWTH=0 -sINITIAL_MEMORY=536870912"
    # Chrome rejects GPUQueue.writeBuffer with views into a RESIZABLE
    # ArrayBuffer (what ALLOW_MEMORY_GROWTH produces) — same browser-API
    # class as the issue-31 TextDecoder crash. Fixed 512 MB heap instead.
    # ggml snapshot 8be60f8 ships WGSL templates as *.tmpl.wgsl, which the
    # embed script blindly embeds as invalid C identifiers (wgsl_cpy.tmpl).
    # Upstream master renamed them to plain *.tmpl (skipped by the script);
    # mirror that rename here (idempotent, working tree only).
    for t in "$SCRIPT_DIR"/ggml/src/ggml-webgpu/wgsl-shaders/*.tmpl.wgsl; do
        [ -e "$t" ] && mv "$t" "${t%.wgsl}" && echo "[INFO] renamed $(basename "$t") -> $(basename "${t%.wgsl}")"
    done
    # Local WGSL kernels for ggml-webgpu (upstream gaps): LayerNorm (NORM),
    # IM2COL, POOL_2D, CONV_TRANSPOSE_2D, UPSCALE (nearest+bilinear), ARANGE
    # + a warning when the encoder silently skips an unsupported op. All
    # validated in-browser via ggml test-backend-ops (see
    # tests/wasm-browser/README note + LEARNINGS). Idempotent apply.
    # NOTE (2026-07): these webgpu ops now live in the pinned ggml submodule
    # (CrispStrobe/ggml @ crispstrobe-ops = v0.10.2 + our ops), so the old
    # build-time patch is no longer applied. Kept in patches/ for provenance.
    echo "[INFO] ggml-webgpu ops (NORM/arange/pool2d/conv_transpose_2d + upstreamed im2col/upscale) are baked into the pinned CrispStrobe/ggml submodule"
    # Experimental: ggml-webgpu links the emdawnwebgpu port and adds
    # -sASYNCIFY itself (INTERFACE link options). Separate output dir so all
    # variants coexist (demo serves this build under webgpu/).
    if [ "$BUILD_DIR_SET" = false ] && [ "$WEBGPU_COMPAT" != "ON" ]; then BUILD_DIR="build-wasm-webgpu"; fi
    echo "[INFO] WebGPU backend enabled -> $BUILD_DIR/"
fi

if [ "$THREADS" = "ON" ]; then
    THREAD_LABEL="multithreaded (requires COOP/COEP headers)"
    THREAD_C_FLAGS="-pthread"
    # Pool sized for typical laptops; ggml uses n_threads <= pool size.
    THREAD_LINK_FLAGS="-pthread -sPTHREAD_POOL_SIZE=8 -sPTHREAD_POOL_SIZE_STRICT=0"
    # Separate output dir so single-threaded and threaded artifacts coexist
    # (the demo serves the threaded build under threaded/).
    if [ "$PROXY" = "ON" ]; then
        # Run main() on a dedicated "runtime" pthread so the servicer worker never
        # blocks; the async recognize (wasm_ocr_pipeline_run_async) proxies the
        # blocking OCR onto it, so ggml's compute threads run without the
        # pthread_join deadlock. This is the browser-safe multithreaded build.
        THREAD_LINK_FLAGS="$THREAD_LINK_FLAGS -sPROXY_TO_PTHREAD=1"
        THREAD_LABEL="multithreaded + PROXY_TO_PTHREAD (deadlock-free; requires COOP/COEP)"
        if [ "$BUILD_DIR_SET" = false ]; then BUILD_DIR="build-wasm-proxy"; fi
        echo "[INFO] PROXY_TO_PTHREAD build enabled -> $BUILD_DIR/"
    else
        if [ "$BUILD_DIR_SET" = false ]; then BUILD_DIR="build-wasm-threads"; fi
        echo "[INFO] Multithreaded build enabled -> $BUILD_DIR/"
    fi
fi

echo "============================================"
echo "  CrispEmbed - WASM Build (OCR)"
echo "  Threading: $THREAD_LABEL"
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
'_wasm_ocr_recognize_copy',\
'_wasm_ocr_confidences',\
'_wasm_ocr_mean_confidence',\
'_wasm_ocr_set_max_tokens',\
'_wasm_ocr_free',\
'_wasm_ocr_pipeline_init',\
'_wasm_ocr_pipeline_run',\
'_wasm_ocr_pipeline_free',\
'_wasm_ocr_pipeline_full_init',\
'_wasm_ocr_pipeline_full_run',\
'_wasm_ocr_pipeline_full_free',\
'_wasm_scan_cleanup_init',\
'_wasm_scan_cleanup_process',\
'_wasm_scan_cleanup_free_image',\
'_wasm_scan_cleanup_free',\
'_wasm_scan_cleanup_detect_page_split',\
'_wasm_scan_cleanup_content_bbox',\
'_wasm_ocr_render',\
'_wasm_text_det_init',\
'_wasm_text_det_run',\
'_wasm_text_det_free',\
'_wasm_layout_init',\
'_wasm_layout_detect',\
'_wasm_layout_free',\
'_malloc',\
'_free',\
'_main'\
]"

# The async proxied recognize (wasm_ocr_pipeline_run_async) is
# __EMSCRIPTEN_PTHREADS__-guarded — it only exists in --threads / --proxy-to-pthread
# builds. Exporting it from the single-thread build fails at link
# ("symbol exported via --export not found").
if [ "$THREADS" = "ON" ]; then
    EXPORTED_FUNCS="${EXPORTED_FUNCS%]},'_wasm_ocr_pipeline_run_async']"
fi

EXPORTED_RUNTIME="[\
'ccall','cwrap','FS','MEMFS','getValue','setValue','UTF8ToString','stringToUTF8','lengthBytesUTF8',\
'HEAPU8','HEAP8','HEAPU32','HEAP32','HEAPF32','ENV'\
]"

# SIMD flags
SIMD_FLAGS=""
if [ "$SIMD" = "ON" ]; then
    SIMD_FLAGS="-msimd128"
    echo "[INFO] WASM SIMD128 enabled"
fi

# Use ninja if available (faster parallel builds) + ccache
GENERATOR=""
if command -v ninja &>/dev/null; then
    GENERATOR="-G Ninja"
    echo "[INFO] Using Ninja generator"
fi
export CCACHE_DIR="${CCACHE_DIR:-${HOME}/.ccache}"

# Configure
echo "[INFO] Configuring with emcmake..."
cd "$SCRIPT_DIR"
emcmake cmake -S . -B "$BUILD_DIR" $GENERATOR \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_OPENMP=OFF \
    -DCRISPEMBED_BUILD_SHARED=OFF \
    -DCRISPEMBED_WASM=ON \
    -DCRISPEMBED_WASM_THREADS="$THREADS" \
    $WEBGPU_FLAGS \
    -DCMAKE_C_FLAGS="$SIMD_FLAGS $THREAD_C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$SIMD_FLAGS $THREAD_C_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="\
-sEXPORTED_FUNCTIONS=$EXPORTED_FUNCS \
-sEXPORTED_RUNTIME_METHODS=$EXPORTED_RUNTIME \
-sALLOW_MEMORY_GROWTH=1 \
-sINITIAL_MEMORY=134217728 \
-sSTACK_SIZE=2097152 \
-sMODULARIZE=1 \
-sEXPORT_NAME=CrispEmbedOCR \
-sENVIRONMENT=web,worker,node \
-sFILESYSTEM=1 \
-sWASM_BIGINT=1 \
-sNO_EXIT_RUNTIME=1 \
$THREAD_LINK_FLAGS \
$WEBGPU_LINK_FLAGS \
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
