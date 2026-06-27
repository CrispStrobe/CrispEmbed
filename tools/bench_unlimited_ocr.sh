#!/bin/bash
# bench_unlimited_ocr.sh — benchmark CrispEmbed vs llama.cpp on Unlimited-OCR
#
# Usage: bash tools/bench_unlimited_ocr.sh [test_image]
set -euo pipefail

IMAGE="${1:-/mnt/volume1/llama.cpp/tools/mtmd/test-1.jpeg}"
THREADS=4
MAX_TOKENS=256

CRISPEMBED="./build/crispembed"
LLAMA_CLI="/mnt/volume1/llama.cpp/build/bin/llama-mtmd-cli"

# CrispEmbed model
CE_MODEL="/mnt/storage/gguf-models/unlimited-ocr-q4_k.gguf"

# llama.cpp models
LLAMA_MODEL="/mnt/storage/gguf-models/llama-unlimited-ocr/unlimited-ocr-Q4_K_M.gguf"
LLAMA_MMPROJ="/mnt/storage/gguf-models/llama-unlimited-ocr/mmproj-unlimited-ocr-bf16.gguf"

echo "============================================="
echo "Unlimited-OCR Benchmark"
echo "============================================="
echo "Image: $IMAGE"
echo "Threads: $THREADS"
echo "Max tokens: $MAX_TOKENS"
echo ""

# --- CrispEmbed ---
echo "--- CrispEmbed (q4_k, $THREADS threads) ---"
if [ -f "$CRISPEMBED" ] && [ -f "$CE_MODEL" ]; then
    # Use explicit model path to avoid auto-download
    UOCR_MAX_NEW=$MAX_TOKENS /usr/bin/time -v $CRISPEMBED \
        --ocr-pipeline "$IMAGE" \
        --ocr-engine unlimited_ocr \
        -m "$CE_MODEL" \
        -t $THREADS 2>&1 | tee /tmp/bench_ce_out.txt
    echo ""
else
    echo "  SKIP: binary or model not found"
    echo "  binary: $CRISPEMBED ($([ -f "$CRISPEMBED" ] && echo 'OK' || echo 'MISSING'))"
    echo "  model:  $CE_MODEL ($([ -f "$CE_MODEL" ] && echo 'OK' || echo 'MISSING'))"
fi
echo ""

# --- llama.cpp ---
echo "--- llama.cpp (Q4_K_M + bf16 mmproj, $THREADS threads) ---"
if [ -f "$LLAMA_CLI" ] && [ -f "$LLAMA_MODEL" ] && [ -f "$LLAMA_MMPROJ" ]; then
    /usr/bin/time -v $LLAMA_CLI \
        -m "$LLAMA_MODEL" \
        --mmproj "$LLAMA_MMPROJ" \
        --image "$IMAGE" \
        -p "document parsing." \
        --chat-template deepseek-ocr \
        --temp 0 --flash-attn off --no-warmup \
        -n $MAX_TOKENS -c 4096 \
        -t $THREADS \
        --dry-multiplier 0.8 --dry-base 1.75 \
        --dry-allowed-length 2 --dry-penalty-last-n -1 \
        --dry-sequence-breaker none 2>&1 | tee /tmp/bench_llama_out.txt
    echo ""
else
    echo "  SKIP: binary or model not found"
    echo "  binary: $LLAMA_CLI ($([ -f "$LLAMA_CLI" ] && echo 'OK' || echo 'MISSING'))"
    echo "  model:  $LLAMA_MODEL ($([ -f "$LLAMA_MODEL" ] && echo 'OK' || echo 'MISSING'))"
    echo "  mmproj: $LLAMA_MMPROJ ($([ -f "$LLAMA_MMPROJ" ] && echo 'OK' || echo 'MISSING'))"
fi

echo ""
echo "============================================="
echo "Benchmark complete"
echo "============================================="
