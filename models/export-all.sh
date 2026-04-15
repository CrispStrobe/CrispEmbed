#!/bin/bash
# Export all CrispEmbed models in both Ollama and CrispEmbed format.
# Usage: ./models/export-all.sh [--ollama-only] [--crisp-only]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY="${PROJECT_DIR}/.bench-venv/bin/python"
OUT="/Volumes/backups/ai/crispembed-gguf"
QUANTIZE="${PROJECT_DIR}/build/crispembed-quantize"

mkdir -p "$OUT/ollama" "$OUT/crisp" "$OUT/quantized"

DO_OLLAMA=1
DO_CRISP=1
case "${1:-}" in
    --ollama-only) DO_CRISP=0 ;;
    --crisp-only)  DO_OLLAMA=0 ;;
esac

# ═══════════════════════════════════════════════════════════
# Encoder models (BERT / XLM-R)
# ═══════════════════════════════════════════════════════════
ENCODER_MODELS=(
    "sentence-transformers/all-MiniLM-L6-v2|all-MiniLM-L6-v2"
    "thenlper/gte-small|gte-small"
    "Snowflake/snowflake-arctic-embed-xs|arctic-embed-xs"
    "intfloat/multilingual-e5-small|multilingual-e5-small"
)

# Gated/large encoder models (may need HF token)
ENCODER_MODELS_GATED=(
    "CrispStrobe/PIXIE-Rune-v1.0|pixie-rune-v1"
    "Snowflake/snowflake-arctic-embed-l-v2|arctic-embed-l-v2"
)

# ═══════════════════════════════════════════════════════════
# Decoder models (Qwen3 / Gemma3)
# ═══════════════════════════════════════════════════════════
DECODER_MODELS=(
    "Qwen/Qwen3-Embedding-0.6B|qwen3-embed-0.6b"
    "jinaai/jina-embeddings-v5-text-nano|jina-v5-nano"
    "jinaai/jina-embeddings-v5-text-small|jina-v5-small"
)

# Models that may need special handling or HF token
DECODER_MODELS_GATED=(
    "Octen/Octen-Embedding-0.6B|octen-0.6b"
    "intfloat/F2LLM-v2-0.6B|f2llm-v2-0.6b"
    "Harrier/Harrier-OSS-v1-0.6B|harrier-0.6b"
    "Harrier/Harrier-OSS-v1-270M|harrier-270m"
)

convert_encoder() {
    local hf_id="$1" name="$2"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Encoder: $name ($hf_id)"
    echo "════════════════════════════════════════════"

    if [ $DO_OLLAMA -eq 1 ]; then
        echo "  → Ollama format..."
        $PY "$SCRIPT_DIR/convert-bert-to-gguf.py" \
            --model "$hf_id" --output "$OUT/ollama/${name}.gguf" --ollama 2>&1 | tail -5
    fi
    if [ $DO_CRISP -eq 1 ]; then
        echo "  → CrispEmbed format..."
        $PY "$SCRIPT_DIR/convert-bert-to-gguf.py" \
            --model "$hf_id" --output "$OUT/crisp/${name}.gguf" --crisp 2>&1 | tail -5
    fi
}

convert_decoder() {
    local hf_id="$1" name="$2"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Decoder: $name ($hf_id)"
    echo "════════════════════════════════════════════"

    if [ $DO_OLLAMA -eq 1 ]; then
        echo "  → Ollama format..."
        $PY "$SCRIPT_DIR/convert-decoder-embed-to-gguf.py" \
            --model "$hf_id" --output "$OUT/ollama/${name}.gguf" --ollama 2>&1 | tail -5
    fi
    if [ $DO_CRISP -eq 1 ]; then
        echo "  → CrispEmbed format..."
        $PY "$SCRIPT_DIR/convert-decoder-embed-to-gguf.py" \
            --model "$hf_id" --output "$OUT/crisp/${name}.gguf" --crisp 2>&1 | tail -5
    fi
}

quantize_model() {
    local src="$1" name="$2"
    if [ ! -x "$QUANTIZE" ]; then
        echo "  [SKIP] crispembed-quantize not built"
        return
    fi
    for qtype in q8_0 q4_k; do
        local dst="$OUT/quantized/${name}-${qtype}.gguf"
        if [ -f "$dst" ]; then
            echo "  [SKIP] $dst already exists"
            continue
        fi
        echo "  → Quantize $qtype..."
        $QUANTIZE "$src" "$dst" "$qtype" 2>&1 | tail -2
    done
}

echo "Output: $OUT"
echo ""

# Convert encoder models
for entry in "${ENCODER_MODELS[@]}"; do
    IFS='|' read -r hf_id name <<< "$entry"
    convert_encoder "$hf_id" "$name"
done

# Try gated encoder models (may fail if not authenticated)
for entry in "${ENCODER_MODELS_GATED[@]}"; do
    IFS='|' read -r hf_id name <<< "$entry"
    convert_encoder "$hf_id" "$name" || echo "  [SKIP] $hf_id (may need HF token)"
done

# Convert decoder models
for entry in "${DECODER_MODELS[@]}"; do
    IFS='|' read -r hf_id name <<< "$entry"
    convert_decoder "$hf_id" "$name"
done

# Try gated decoder models
for entry in "${DECODER_MODELS_GATED[@]}"; do
    IFS='|' read -r hf_id name <<< "$entry"
    convert_decoder "$hf_id" "$name" || echo "  [SKIP] $hf_id (may need HF token)"
done

# Quantize all Ollama F32 models
echo ""
echo "════════════════════════════════════════════"
echo "  Quantizing..."
echo "════════════════════════════════════════════"
for f in "$OUT/ollama"/*.gguf; do
    name=$(basename "$f" .gguf)
    quantize_model "$f" "$name"
done

echo ""
echo "════════════════════════════════════════════"
echo "  Done!"
echo "════════════════════════════════════════════"
ls -lhS "$OUT/ollama/"*.gguf 2>/dev/null
echo "---"
ls -lhS "$OUT/quantized/"*.gguf 2>/dev/null
