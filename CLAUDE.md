# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CrispEmbed is a C/C++ text embedding inference engine built on ggml. It loads GGUF models and produces embedding vectors with GPU acceleration (Metal/CUDA/Vulkan). No Python runtime, no ONNX dependency. 10 text architectures (BERT, XLM-R, MPNet, NomicBERT, ModernBERT, GTE v1.5, DeBERTa-v2, Qwen3, Gemma3, SPLADE) plus omnimodal BidirLM-Omni (text + audio + image, shared 2048-d space, 3D MRoPE + DeepStack visual injection). 23+ verified models, 43+ in registry. Python/Rust/Dart APIs, iOS/Android builds. 9.5x faster than FastEmbed on MiniLM-L6.

## Build commands

```bash
# macOS (Metal GPU) — the standard dev build
./build-macos.sh                    # Metal + Accelerate + embedded shaders
./build-macos.sh --shared           # Also build libcrispembed.dylib for Python
./build-macos.sh --clean            # Clean rebuild (needed after ggml changes)

# Manual cmake (Linux/cross-platform)
cmake -S . -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCRISPEMBED_BUILD_SHARED=ON
cmake --build build -j

# GPU variants
cmake -S . -B build -DGGML_CUDA=ON      # NVIDIA
cmake -S . -B build -DGGML_VULKAN=ON    # Cross-platform GPU
```

Binaries land in `build/`: `crispembed`, `crispembed-server`, `crispembed-quantize`.

## Running

```bash
# CLI (auto-downloads models on first use)
./build/crispembed -m all-MiniLM-L6-v2 "Hello world"

# Server (model loaded once, HTTP API)
./build/crispembed-server -m all-MiniLM-L6-v2 --port 8080

# Quantize
./build/crispembed-quantize model.gguf model-q8_0.gguf q8_0
```

## Testing

```bash
# Validate GGUF output matches HuggingFace (needs pip: sentence-transformers)
python tests/test_vs_hf.py --model sentence-transformers/all-MiniLM-L6-v2 \
    --gguf ~/.cache/crispembed/all-MiniLM-L6-v2.gguf --binary ./build/crispembed

# Test all cached models
python tests/test_all_models.py

# Benchmark (creates .bench-venv for Python deps automatically)
./benchmark.sh -n 20                      # single model
./benchmark.sh --multi -n 20              # 3 models, all engines
./benchmark.sh --skip-hf --skip-fastembed  # CrispEmbed only
```

## Architecture

### Inference pipeline

Text → **Tokenizer** (auto-detected from GGUF: WordPiece/SentencePiece/BPE) → Token+Position+Type embeddings → N × Transformer layers → **Pooling** (mean/CLS/last-token) → L2 normalization → `float[dim]`

### Source layout

- **`src/crispembed.h`** — Public C API: dense (`crispembed_encode`, `crispembed_encode_batch`), sparse (`crispembed_encode_sparse`), multi-vector (`crispembed_encode_multivec`), reranker (`crispembed_rerank`), audio (`crispembed_encode_audio`), image (`crispembed_encode_image`, `crispembed_encode_image_raw`), text-with-image (`crispembed_encode_text_with_image`, `crispembed_encode_with_image_ids` — the latter takes pre-tokenized int32 ids and bypasses the BPE tokenizer for clean parity with external tokenizers), capability queries (`crispembed_has_sparse`, `crispembed_has_colbert`, `crispembed_is_reranker`, `crispembed_has_audio`, `crispembed_has_vision`), config (`crispembed_set_dim`, `crispembed_set_prefix`, `crispembed_get_prefix`), lifecycle (`crispembed_init`, `crispembed_free`).
- **`src/crispembed.cpp`** — Core inference engine. Builds ggml computation graphs for encoder models (BERT/XLM-R/BGE-M3). `build_encoder_graph(mode)`: mode=0 → dense `encoder_out`, mode=1 → sparse `sparse_out [1,T]`, mode=2 → colbert `colbert_out [dim,T]`. Handles backend init, graph caching per mode, QKV weight fusion, batched encoding. Decoder branch initializes its own `ggml_backend_sched` + `compute_meta` (sized to `max(4096, n_layer*50+256)`) so BidirLM-Omni text/text+image runs through the GPU dispatcher rather than the CPU fallback. **Critical**: GGUF metadata (u32/f32 lambdas) must be read before `gguf_free(g)` — use-after-free will silently corrupt inference.
- **`src/decoder_embed.cpp`** — Separate forward pass for decoder-based embedders (Qwen3/Gemma3 + bidirectional Qwen3 like BidirLM-Omni). Uses RoPE, GQA, RMSNorm, q_norm/k_norm, SwiGLU. Pooling and attention mask are GGUF-driven: `decoder.pooling_method` (1=mean, 2=last) + `decoder.is_bidirectional` (0/1). When a `dec_image_input` is supplied: (a) image-embed rows are spliced into `inputs_embeds` at every `image_token_id` placeholder via a host-prepared keep-mask + patch tensor, (b) `deepstack_features[k]` are added at the same positions after the first `n_deepstack` transformer layers, (c) RoPE switches to `ggml_rope_multi(IMROPE)` with 4×T position ids `[t,h,w,e=t]` derived from `grid_thw` (the e=t pin makes ggml IMROPE numerically agree with HF `apply_interleaved_mrope` at sectors that fall outside the `mrope_section` slices, e.g. 61–62 for `[24,20,20]`).
- **`src/bidirlm_vision.cpp`** — BidirLM-Omni vision tower (Qwen2VL-style ViT). Loads `visual.*` tensors, runs patch embed + interpolated learned position embedding (4-corner gather) + 24-layer pre-LN ViT with rotate-half 2D RoPE + final patch merger and DeepStack hook mergers. Returns `image_embeds` (n_merged, 2048) + 3 × `deepstack[k]` (n_merged, 2048). The graph builder takes an `include_deepstack` flag — `crispembed_encode_image` (mean-pool only) sets it false to skip those subgraphs; `crispembed_encode_image_raw` and `crispembed_encode_text_with_image` set it true.
- **`src/bidirlm_audio.cpp`** — Wrapper that adapts the shared `crisp_audio` library (lives in `../CrispASR/crisp_audio`) for BidirLM-Omni audio inputs. Mean-pools the per-frame encoder output, L2-normalizes, returns a single 2048-d vector in the model's shared cross-modal space. Built only when CMake finds `CRISP_AUDIO_DIR`.
- **`src/tokenizer.cpp`** — WordPiece tokenizer (BERT-style).
- **`src/tokenizer_spm.cpp`** — SentencePiece Unigram tokenizer (Viterbi DP, NOT bigram merge — see LEARNINGS.md).
- **`src/tokenizer_bpe.cpp`** — GPT-2/Gemma BPE tokenizer.
- **`src/core/gguf_loader.cpp`** — GGUF file parser, tensor weight loading.

### Model architecture dispatch

The model type is auto-detected from GGUF metadata:
- **Encoder models** (BERT, XLM-R, MPNet, NomicBERT): `crispembed.cpp` → `encode_tokens()` / `encode_tokens_batch()`
- **Decoder models** (Qwen3, Gemma3, BidirLM-Omni text path): `decoder_embed.cpp` → `decoder_encode_tokens()` (with optional `dec_image_input` for image-conditioned text)
- **Vision models** (BidirLM-Omni image path): `bidirlm_vision.cpp` → `bidirlm_vision::encode()` triggered lazily by `crispembed_encode_image*`
- **Audio models** (BidirLM-Omni audio path): `bidirlm_audio.cpp` → `crisp_audio_*` from `../CrispASR/crisp_audio`

Detection heuristic: presence of `blk.0.ffn_gate` tensor → decoder path.
Encoder variants auto-detected: no `pos_embd` → RoPE (NomicBERT/ModernBERT/GTE v1.5), `rel_attn_bias` → relative position bias (MPNet), `pre_ln` → pre-LayerNorm (ModernBERT/GTE v1.5), `ffn_up_gate` → fused ggml_geglu.

**Critical**: BPE tokenizer re-loading after merge tensor init must preserve `suffix_id=-1` — overwriting with `unk_id` causes wrong SEP token.

### Tokenizer dispatch

Auto-detected from GGUF `tokenizer.ggml.type`: 0=WordPiece, 1=BPE, 2=SentencePiece. Fallback heuristic: vocab > 100K → SentencePiece.

### Server API endpoints

`examples/server/server.cpp` exposes four API dialects:
- `POST /embed` — native: `{"texts": ["..."]}`
- `POST /v1/embeddings` — OpenAI-compatible
- `POST /api/embed` — Ollama-compatible (batch)
- `POST /api/embeddings` — Ollama-compatible (legacy single)

### Python wrapper

`python/crispembed/_binding.py` — ctypes binding to `libcrispembed.{so,dylib,dll}`. Full API: `encode()` (dense, single/batch), `encode_sparse()` (BGE-M3 term weights), `encode_multivec()` (ColBERT per-token), `rerank()` (cross-encoder), `rerank_biencoder()` (cosine similarity ranking), `encode_audio()` (BidirLM-Omni PCM → 2048-d), `encode_image()` (image → mean-pooled 2048-d), `encode_image_raw()` (un-pooled image_embeds + deepstack slabs), `encode_text_with_image()` (image-conditioned text via DeepStack injection + 3D MRoPE), `set_dim()` (Matryoshka), `set_prefix()` (query/passage prefixes). Properties: `has_sparse`, `has_colbert`, `is_reranker`, `has_audio`, `has_vision`, `dim`, `prefix`. Image preprocessing helpers live in `python/crispembed/image.py` (Qwen2VLImageProcessorFast wrapper).

### Rust crates

`crispembed-sys/` — raw `extern "C"` bindings + cmake build.rs (links libcrispembed). `crispembed/` — safe `CrispEmbed` wrapper with all encode/rerank methods. Features: `cuda`, `metal`, `vulkan` pass cmake flags. Build: `cargo build --features cuda`.

### Audio path (omnimodal embedding)

BidirLM-Omni and similar Whisper-shape audio encoders are handled by the
shared `crisp_audio` library inside CrispASR:
`../CrispASR/crisp_audio/{include/crisp_audio.h, src/audio_tower.cpp}`.
CrispEmbed's CMake auto-discovers the directory at the sibling-repo path —
override with `-DCRISP_AUDIO_DIR=/some/where`. When found, `crispembed-core`
gains the `core_mel::*` symbols (vendored from CrispASR for shared use) and
the `bidirlm_audio.cpp` wrapper compiles, exposing `crispembed_encode_audio()`
on the C ABI and `encode_audio()` on the Python/Rust/Dart wrappers.

`crisp_audio` is dialect-driven: the first dialect (`CRISP_AUDIO_DIALECT_QWEN_OMNI`)
covers qwen3-asr and BidirLM-Omni; new dialects (Whisper-classic, etc.)
plug in via additional graph builders without API changes. Per-model scalars
(d_model, n_layers, n_window, output_dim, tensor_prefix, meta_prefix) are
all GGUF-driven so a single static lib serves both consumers.

CLI: `crispembed -m model.gguf --audio in.raw` (raw f32le 16 kHz mono PCM).
Python: `model.encode_audio(np.ndarray, sr=16000)`.
Rust: `model.encode_audio(&[f32]) -> Vec<f32>`.

### Vision path (BidirLM-Omni image embedding)

The vision tower is built into `crispembed-core` (no sibling-lib dependency). When the loaded GGUF contains `visual.*` tensors, `bidirlm_vision::context` opens lazily on the first `crispembed_encode_image*` call, sharing the parent backend.

Three entry points:
- `crispembed_encode_image(ctx, pixel_patches, n_patches, grid_thw, n_images, &dim)` — runs the tower without DeepStack, mean-pools merged tokens, L2-normalizes, returns one 2048-d cross-modal vector.
- `crispembed_encode_image_raw(ctx, ..., &n_merged, &dim, &n_deepstack)` — returns the un-pooled `image_embeds` followed by `n_deepstack` deepstack slabs, all `(n_merged × dim)` row-major. Used by the parity test.
- `crispembed_encode_text_with_image(ctx, text, pixel_patches, n_patches, grid_thw, n_images, &dim)` — full multimodal forward: vision tower → splice `image_embeds` into `inputs_embeds` at every `image_token_id` placeholder → run decoder with DeepStack injection at the first 3 layers and 3D MRoPE position ids derived from `grid_thw`.

CLI: `crispembed -m model.gguf --image-raw patches.f32 --grid-thw T,H,W` (preprocessed float32 patch rows).
Python: `model.encode_image(image)` / `model.encode_image_raw(image)` / `model.encode_text_with_image(text, image)`.
Image preprocessing (`crispembed.image.preprocess_image`) wraps HF `Qwen2VLImageProcessorFast`; a C++ port of `smart_resize + normalize + patchify` is future work.

### BGE-M3 testing

```bash
python tests/test_bgem3.py --gguf ~/.cache/crispembed/bge-m3.gguf --lib build/libcrispembed.so
```
Validates dense (cosine), sparse (IoU), colbert (per-token cosine) against FlagEmbedding.

### Model auto-download

`examples/cli/model_mgr.cpp` contains a registry of 35+ models (21 verified embeddings + 5 rerankers + multilingual/decoder/omnimodal models) with HuggingFace URLs. Models are cached to `$CRISPEMBED_CACHE_DIR` (or the per-user default `~/.cache/crispembed/`). The CLI and server accept short names (e.g., `-m all-MiniLM-L6-v2`) and resolve + download automatically. Python: `CrispEmbed.list_models()` returns all available models.

## Key technical details

- **Metal shaders must be embedded**: `build-macos.sh` passes `-DGGML_METAL_EMBED_LIBRARY=ON`. Without this, Metal init silently falls back to CPU. Verify with stderr output: "using embedded metal library" + "using MTL0 backend".
- **ggml is a git submodule**: Run `git submodule update --init --recursive` if `ggml/CMakeLists.txt` is missing.
- **Graph caching**: The ggml computation graph is built once per sequence length and reused. `encode_tokens_batch()` sorts texts by token count for cache hits.
- **QKV fusion**: For encoder models, separate Q/K/V weight matrices are fused into a single `[3H, H]` matrix at load time, reducing kernel launches.
- **Batch encoding**: `crispembed_encode_batch()` runs all texts through a single ggml graph with 4D flash attention. This is significantly faster than per-text encoding.
- **GGUF conversion scripts**: `models/convert-bert-to-gguf.py` (encoder models) and `models/convert-decoder-embed-to-gguf.py` (Qwen3/Gemma3 decoder + BidirLM-Omni text + audio + vision). Pre-converted models available at huggingface.co/cstr/.
- **3D MRoPE workaround**: ggml's `IMROPE` mode covers sectors `[0, 3*sections[0])` for T, `[1, 3*sections[1])`-style for H/W, with anything outside falling to a 4th `theta_e`. HF's `apply_interleaved_mrope` falls back to T at those sectors. CrispEmbed pins `pos_e = pos_t` per-token so `theta_e ≡ theta_t`, making the two definitions numerically identical for `mrope_section=[24,20,20]`.
- **Decoder scheduler init**: `crispembed_init` now creates the decoder's `ggml_backend_sched` and resizes `compute_meta` separately from the encoder branch — without this, decoder graphs silently used the CPU fallback inside `decoder_encode_tokens`. The capacity formula (`max(4096, n_layer*50+256)`) keeps headroom for image-splice + per-layer DeepStack adds.
- **CrispAudio dependency**: lives at `../CrispASR/crisp_audio` as a CMake static lib. Both repos must agree on a ggml SHA — kept in sync between CrispEmbed (submodule pin) and CrispASR (vendored ggml `GGML_VERSION_*`). Currently both at 0.10.0+.
