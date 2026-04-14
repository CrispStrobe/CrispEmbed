# CrispEmbed

Lightweight text embedding inference via ggml. No Python runtime, no ONNX.

**Status: Phase 1 in progress** — scaffold complete, model loads + tokenizes + runs ggml graph. Output needs validation (pre-LN vs post-LN fix pending).

## Quick start

```bash
# Build
cmake -S . -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build -j

# Convert a model
pip install torch transformers gguf
python models/convert-bert-to-gguf.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output all-MiniLM-L6-v2.gguf

# Encode text
./build/crispembed -m all-MiniLM-L6-v2.gguf "Hello world"
# prints 384-dim L2-normalized embedding
```

## Architecture

BERT/MiniLM transformer encoder via ggml graph:
- Token + Position + Type embeddings
- N × (LayerNorm → MHA → residual → LayerNorm → FFN → residual)
- Mean pooling + L2 normalization
- WordPiece tokenizer loaded from GGUF metadata

See [PLAN.md](PLAN.md) for the full roadmap.
