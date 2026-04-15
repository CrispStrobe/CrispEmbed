#!/bin/bash
# CrispEmbed Benchmark — compare against HuggingFace, fastembed, and fastembed-rs.
#
# Usage:
#   ./benchmark.sh                                   # auto-detect everything
#   ./benchmark.sh -m models/all-MiniLM-L6-v2.gguf  # specific model
#   ./benchmark.sh -m all-MiniLM-L6-v2               # auto-download by name
#   ./benchmark.sh -n 200                            # 200 iterations
#   ./benchmark.sh --skip-hf --skip-fastembed        # CrispEmbed only
#   ./benchmark.sh --multi                           # benchmark multiple models

PORT=8090
N_RUNS=50
MODEL_PATH=""
HF_MODEL=""
SKIP_HF=0
SKIP_FASTEMBED=0
MULTI_MODEL=0
FE_RS_DIR="../fastembed-rs"

usage() {
    echo "Usage: $0 [options]"
    echo "  -m, --model <path>      GGUF model path or name (auto-download)"
    echo "  --hf-model <id>         HuggingFace model ID for comparison"
    echo "  -n, --n-runs <n>        Number of iterations (default: $N_RUNS)"
    echo "  -p, --port <port>       Server port (default: $PORT)"
    echo "  --skip-hf               Skip HuggingFace benchmark"
    echo "  --skip-fastembed        Skip FastEmbed benchmarks"
    echo "  --multi                 Benchmark multiple small models"
    echo "  --help                  Show this help"
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model)       MODEL_PATH="$2"; shift ;;
        --hf-model)       HF_MODEL="$2"; shift ;;
        -n|--n-runs)      N_RUNS="$2"; shift ;;
        -p|--port)        PORT="$2"; shift ;;
        --skip-hf)        SKIP_HF=1 ;;
        --skip-fastembed) SKIP_FASTEMBED=1 ;;
        --multi)          MULTI_MODEL=1 ;;
        --help)           usage ;;
        *)                MODEL_PATH="$1" ;;
    esac
    shift
done

echo "============================================"
echo "  CrispEmbed Benchmark"
echo "============================================"

# --- Detect Python & venv ---
PY_BIN=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then PY_BIN="$p"; break; fi
done
if [ -z "$PY_BIN" ]; then
    echo "[ERROR] Python not found."
    exit 1
fi

VENV_DIR=".bench-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating Python venv at $VENV_DIR..."
    "$PY_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PY_BIN="$VENV_DIR/bin/python"

# --- Find binaries ---
BIN=""
SERVER_BIN=""
SHARED_LIB=""
for p in build/crispembed build-blas/crispembed build-cuda/crispembed build-vulkan/crispembed \
         build/Release/crispembed build/bin/crispembed; do
    [ -x "$p" ] && BIN="$p" && break
done
for p in build/crispembed-server build-blas/crispembed-server build-cuda/crispembed-server \
         build-vulkan/crispembed-server build/Release/crispembed-server; do
    [ -x "$p" ] && SERVER_BIN="$p" && break
done
for p in build/libcrispembed.dylib build/libcrispembed.so build/Release/crispembed.dll \
         build/lib/libcrispembed.dylib build/lib/libcrispembed.so; do
    [ -f "$p" ] && SHARED_LIB="$p" && break
done

if [ -z "$BIN" ]; then
    echo "[ERROR] No crispembed binary found. Build first:"
    echo "  cmake -S . -B build && cmake --build build -j"
    echo "  ./build-macos.sh     (macOS with Metal)"
    exit 1
fi

# --- Map CrispEmbed model name to HuggingFace ID ---
map_hf_model() {
    local basename
    basename=$(echo "$1" | sed 's/\.gguf$//' | sed 's/-q[0-9].*//')
    case "$basename" in
        *all-MiniLM-L6-v2*)       echo "sentence-transformers/all-MiniLM-L6-v2" ;;
        *gte-small*)              echo "thenlper/gte-small" ;;
        *arctic-embed-xs*)        echo "Snowflake/snowflake-arctic-embed-xs" ;;
        *arctic-embed-l-v2*)      echo "Snowflake/snowflake-arctic-embed-l-v2" ;;
        *multilingual-e5-small*)  echo "intfloat/multilingual-e5-small" ;;
        *PIXIE-Rune*|*pixie-rune*)  echo "CrispStrobe/PIXIE-Rune-v1.0" ;;
        *octen*0.6*)              echo "Octen/Octen-Embedding-0.6B" ;;
        *F2LLM*0.6*|*f2llm*0.6*) echo "intfloat/F2LLM-v2-0.6B" ;;
        *jina*nano*|*jina-v5-nano*)  echo "jinaai/jina-embeddings-v3-nano" ;;
        *jina*small*|*jina-v5-small*) echo "jinaai/jina-embeddings-v3-small" ;;
        *harrier*0.6*)            echo "Harrier/Harrier-OSS-v1-0.6B" ;;
        *harrier*270*)            echo "Harrier/Harrier-OSS-v1-270M" ;;
        *qwen3*embed*0.6*)        echo "Qwen/Qwen3-Embedding-0.6B" ;;
        *)                        echo "" ;;
    esac
}

# --- Resolve model ---
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(ls *.gguf 2>/dev/null | head -1 || true)
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH="all-MiniLM-L6-v2"
        echo "[INFO] No .gguf found locally. Using default: $MODEL_PATH"
    fi
fi

if [ -z "$HF_MODEL" ]; then
    HF_MODEL=$(map_hf_model "$(basename "$MODEL_PATH")")
fi

TEXT="The quick brown fox jumps over the lazy dog near the river bank"

# --- Timing helper: uses date +%s%N on Linux, python on macOS ---
if date +%s%N &>/dev/null && [ "$(date +%s%N)" != "%N" ]; then
    get_ms() { echo $(( $(date +%s%N) / 1000000 )); }
else
    get_ms() { $PY_BIN -c 'import time; print(int(time.time()*1000))'; }
fi

fmt_rate() {
    local elapsed_ms=$1 count=$2
    if [ "$elapsed_ms" -gt 0 ]; then
        local avg=$(echo "scale=1; $elapsed_ms / $count" | bc)
        local tps=$(echo "scale=0; $count * 1000 / $elapsed_ms" | bc)
        echo "${avg}ms/text  ${tps} texts/s"
    else
        echo "0.0ms/text  -- texts/s"
    fi
}

# --- Prepare fastembed-rs binary ---
FE_RS_BIN=""
if [ $SKIP_FASTEMBED -eq 0 ]; then
    if [ ! -d "$FE_RS_DIR" ] && command -v git &>/dev/null; then
        echo "[INFO] Cloning fastembed-rs..."
        git clone -q -b feat/new-model-entries \
            https://github.com/CrispStrobe/fastembed-rs.git "$FE_RS_DIR" 2>/dev/null || true
    fi
    for p in "$FE_RS_DIR/target/release/examples/bench" "$FE_RS_DIR/target/release/fastembed-bench"; do
        [ -x "$p" ] && FE_RS_BIN="$p" && break
    done
    if [ -z "$FE_RS_BIN" ] && [ -f "$FE_RS_DIR/Cargo.toml" ] && command -v cargo &>/dev/null; then
        echo "[INFO] Building fastembed-rs..."
        (cd "$FE_RS_DIR" && cargo build --release --example bench 2>/dev/null) || true
        for p in "$FE_RS_DIR/target/release/examples/bench" "$FE_RS_DIR/target/release/fastembed-bench"; do
            [ -x "$p" ] && FE_RS_BIN="$p" && break
        done
    fi
fi

# =====================================================================
# Benchmark function: runs all engines for one model
# =====================================================================
run_benchmark() {
    local model_path="$1"
    local hf_model="$2"
    local label="$3"

    echo ""
    echo "============================================"
    echo "  Model: $label"
    echo "  CrispEmbed: $model_path"
    echo "  HF ID:      ${hf_model:-unknown}"
    echo "============================================"

    # --- CrispEmbed CLI ---
    echo ""
    echo "--- CrispEmbed CLI ---"
    echo "  Loading model (may download on first run)..."
    WARMUP_STDERR=$("$BIN" -m "$model_path" "warmup" 2>&1 >/dev/null || true)
    BACKEND=$(echo "$WARMUP_STDERR" | grep -o "using [^ ]* backend" | head -1)
    if [ -n "$BACKEND" ]; then
        echo "  Backend: $BACKEND"
    fi

    START=$(get_ms)
    for ((i = 0; i < N_RUNS; i++)); do
        "$BIN" -m "$model_path" "$TEXT" >/dev/null 2>&1
    done
    END=$(get_ms)
    echo "  CLI:    $(fmt_rate $((END - START)) $N_RUNS) (includes model load)"

    # --- CrispEmbed Python wrapper ---
    if [ -n "$SHARED_LIB" ]; then
        # Resolve GGUF path for Python wrapper (it needs actual file path, not model name)
        GGUF_PATH=""
        if [ -f "$model_path" ]; then
            GGUF_PATH="$model_path"
        else
            # Check cache directory for matching GGUF
            CACHE_DIR="${CRISPEMBED_CACHE_DIR:-$HOME/.cache/crispembed}"
            for f in "$CACHE_DIR/$model_path.gguf" "$CACHE_DIR/${model_path}-q8_0.gguf" \
                     "$CACHE_DIR"/*"$model_path"*.gguf; do
                [ -f "$f" ] && GGUF_PATH="$f" && break
            done
        fi

        if [ -n "$GGUF_PATH" ]; then
            echo ""
            echo "--- CrispEmbed Python (ctypes) ---"
            SHARED_LIB_ABS=$(cd "$(dirname "$SHARED_LIB")" && pwd)/$(basename "$SHARED_LIB")
            $PY_BIN -c "
import time, sys, os
sys.path.insert(0, 'python')

try:
    from crispembed import CrispEmbed
except ImportError:
    print('  [SKIP] crispembed Python module not found')
    sys.exit(0)

try:
    model = CrispEmbed('$GGUF_PATH', n_threads=4, lib_path='$SHARED_LIB_ABS')
    text = '$TEXT'
    model.encode([text])  # warmup

    N = $N_RUNS
    t0 = time.perf_counter()
    for _ in range(N):
        model.encode([text])
    elapsed = time.perf_counter() - t0
    print(f'  Single: {elapsed/N*1000:.1f}ms/text  {N/elapsed:.0f} texts/s')

    batch = [text] * 10
    runs = max(10, N // 5)
    t0 = time.perf_counter()
    for _ in range(runs):
        model.encode(batch)
    elapsed = time.perf_counter() - t0
    print(f'  Batch:  {elapsed/runs*1000:.1f}ms/10texts  {runs*10/elapsed:.0f} texts/s')
except Exception as e:
    print(f'  [ERROR] {e}')
" 2>/dev/null
        fi
    fi

    # --- CrispEmbed Server ---
    if [ -n "$SERVER_BIN" ]; then
        echo ""
        echo "--- CrispEmbed Server ---"

        "$SERVER_BIN" -m "$model_path" --port "$PORT" >/dev/null 2>&1 &
        SERVER_PID=$!
        srv_cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true; }
        trap srv_cleanup EXIT

        echo "  Waiting for server..."
        for ((w = 0; w < 30; w++)); do
            curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 && break
            sleep 1
        done

        # Warmup
        for ((j = 0; j < 3; j++)); do
            curl -s -X POST "http://localhost:$PORT/embed" \
                -H "Content-Type: application/json" \
                -d "{\"texts\":[\"warmup\"]}" >/dev/null 2>&1 || sleep 1
        done

        # Single-text
        START=$(get_ms)
        for ((i = 0; i < N_RUNS; i++)); do
            curl -s -X POST "http://localhost:$PORT/embed" \
                -H "Content-Type: application/json" \
                -d "{\"texts\":[\"$TEXT\"]}" >/dev/null
        done
        END=$(get_ms)
        echo "  Single: $(fmt_rate $((END - START)) $N_RUNS)"

        # Batch (10 texts per request)
        BATCH_JSON=$($PY_BIN -c "import json; print(json.dumps({'texts': ['$TEXT']*10}))")
        BATCH_RUNS=$(( N_RUNS > 50 ? N_RUNS / 5 : 10 ))
        START=$(get_ms)
        for ((i = 0; i < BATCH_RUNS; i++)); do
            curl -s -X POST "http://localhost:$PORT/embed" \
                -H "Content-Type: application/json" \
                -d "$BATCH_JSON" >/dev/null
        done
        END=$(get_ms)
        ELAPSED=$((END - START))
        if [ "$ELAPSED" -gt 0 ]; then
            AVG=$(echo "scale=1; $ELAPSED / $BATCH_RUNS" | bc)
            TPS=$(echo "scale=0; $BATCH_RUNS * 10 * 1000 / $ELAPSED" | bc)
            echo "  Batch:  ${AVG}ms/10texts  ${TPS} texts/s"
        fi

        srv_cleanup
        trap - EXIT
    fi

    # --- HuggingFace ---
    if [ $SKIP_HF -eq 0 ] && [ -n "$hf_model" ]; then
        echo ""
        echo "--- HuggingFace sentence-transformers ---"

        $PY_BIN -c "
import time, sys
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    print('  Installing sentence-transformers...', flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sentence-transformers'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer('$hf_model', trust_remote_code=True)
    text = '$TEXT'
    model.encode([text], normalize_embeddings=True)  # warmup
    N = $N_RUNS
    t0 = time.perf_counter()
    for _ in range(N):
        model.encode([text], normalize_embeddings=True)
    elapsed = time.perf_counter() - t0
    print(f'  Single: {elapsed/N*1000:.1f}ms/text  {N/elapsed:.0f} texts/s')
    # Batch
    batch = [text] * 10
    t0 = time.perf_counter()
    runs = max(10, N // 5)
    for _ in range(runs):
        model.encode(batch, normalize_embeddings=True, batch_size=32)
    elapsed = time.perf_counter() - t0
    print(f'  Batch:  {elapsed/runs*1000:.1f}ms/10texts  {runs*10/elapsed:.0f} texts/s')
except Exception as e:
    print(f'  [ERROR] {e}')
" 2>/dev/null
    fi

    # --- FastEmbed (Python ONNX) ---
    if [ $SKIP_FASTEMBED -eq 0 ] && [ -n "$hf_model" ]; then
        echo ""
        echo "--- FastEmbed (ONNX) ---"

        $PY_BIN -c "
import time, sys
try:
    from fastembed import TextEmbedding
except ImportError:
    import subprocess
    print('  Installing fastembed...', flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'fastembed'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from fastembed import TextEmbedding

try:
    model = TextEmbedding('$hf_model')
    text = '$TEXT'
    list(model.embed([text]))  # warmup
    N = $N_RUNS
    t0 = time.perf_counter()
    for _ in range(N):
        list(model.embed([text]))
    elapsed = time.perf_counter() - t0
    print(f'  Single: {elapsed/N*1000:.1f}ms/text  {N/elapsed:.0f} texts/s')
except Exception as e:
    print(f'  Skipped ({e})')
" 2>/dev/null
    fi

    # --- fastembed-rs (Rust ONNX) ---
    if [ $SKIP_FASTEMBED -eq 0 ] && [ -n "$FE_RS_BIN" ] && [ -n "$hf_model" ]; then
        echo ""
        echo "--- fastembed-rs (Rust ONNX) ---"
        "$FE_RS_BIN" --model "$hf_model" --text "$TEXT" --n-runs "$N_RUNS" 2>/dev/null || echo "  Skipped (model not supported)"
    fi
}

# =====================================================================
# Run benchmarks
# =====================================================================

echo "Binary:     $BIN"
echo "Server:     ${SERVER_BIN:-not found}"
echo "Shared lib: ${SHARED_LIB:-not found (skip Python wrapper)}"
echo "Runs:       $N_RUNS"

if [ $MULTI_MODEL -eq 1 ]; then
    # Small models that benchmark quickly and are broadly supported
    MODELS=(
        "all-MiniLM-L6-v2"
        "gte-small"
        "arctic-embed-xs"
    )

    for m in "${MODELS[@]}"; do
        hf=$(map_hf_model "$m")
        run_benchmark "$m" "$hf" "$m"
    done
else
    run_benchmark "$MODEL_PATH" "$HF_MODEL" "$(basename "$MODEL_PATH" .gguf)"
fi

echo ""
echo "============================================"
echo "  Benchmark Complete"
echo "============================================"
echo "NOTE: Server mode is the fair comparison (model loaded once)."
echo "      CLI includes model load overhead per call."
