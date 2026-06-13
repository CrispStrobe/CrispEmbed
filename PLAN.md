# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any supported model from a GGUF file (auto-detect architecture)
2. Tokenizes input text (WordPiece / SentencePiece / BPE from GGUF metadata)
3. Runs the transformer encoder or decoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, Python, Rust, and Dart wrappers

## Architecture (v0.7.0)

```
Input text / image / audio
    │
    ├─► Text ──► Tokenizer (WordPiece / SentencePiece / BPE)
    │              │
    │              ├─► Encoder path (BERT, XLM-R, MPNet, NomicBERT,
    │              │     ModernBERT, GTE v1.5, DeBERTa-v2, SPLADE)
    │              │     Token + Pos [+ Type] embeddings
    │              │     N × Transformer layer (LN → MHA → FFN → residual)
    │              │     Pooling (mean / CLS) + optional heads
    │              │     → dense / sparse / ColBERT / reranker output
    │              │
    │              └─► Decoder path (Qwen3, Gemma3, BidirLM-Omni text)
    │                    Token embeddings + RoPE
    │                    N × (RMSNorm → GQA → SwiGLU/GeGLU → residual)
    │                    Last-token / mean pooling + L2 normalize
    │
    ├─► Image ──► ViT path (SigLIP/CLIP: vit_embed.cpp)
    │               Conv2D patch embed → transformer → mean pool → L2
    │
    ├─► Image ──► BidirLM-Omni vision (bidirlm_vision.cpp)
    │               Qwen2VL ViT + patch merger + DeepStack
    │               → image_embeds spliced into decoder
    │
    ├─► Image ──► CNN path (cnn_embed.cpp)
    │               SCRFD/YuNet face detection (FPN + anchor decode + NMS)
    │               ArcFace/SFace/AuraFace face recognition
    │
    ├─► Audio ──► BidirLM-Omni audio (bidirlm_audio.cpp)
    │               crisp_audio Whisper-shape encoder → mean pool → 2048-d
    │
    ├─► Math  ──► DeiT encoder + TrOCR decoder (math_ocr.cpp)
    │               Printed math → LaTeX via ggml graph compute
    │
    ├─► Math  ──► HMER: DenseNet-121 + GRU attention (hmer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2016)
    │
    ├─► Math  ──► BTTR: DenseNet + Transformer decoder (bttr_ocr.cpp)
    │                 Handwritten math → LaTeX (CROHME 2014, 53% exact match)
    │
    ├─► OCR   ──► InternVL2.5: InternViT-300M + InternLM2.5-1.8B (internvl2_ocr.cpp)
    │               2.1B VLM OCR, EN+DE, KV cache, dynamic tiling, OCRBench ~830
    │
    ├─► Scene ──► PARSeq: ViT + 1-layer two-stream decoder (parseq_ocr.cpp)
    │               Scene text recognition, 94-char ASCII, 24M (base) / 6M (tiny)
    │
    └─► Text  ──► GLiNER NER: dual-backbone span matching (gliner_ner.cpp)
                    Zero-shot NER with two backbone options:
                    • LFM2.5-bi (BPE → ShortConv+GQA → layer fusion → BiLSTM)
                    • DeBERTa-v3 (SPM → disentangled attn → 768→512 proj → BiLSTM)
```

## Supported architectures (v0.7.0)

| Architecture | Tokenizer | Key features | Example models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, SPLADE |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| MPNet encoder | WordPiece | Post-LN, T5-style rel attn bias | all-mpnet-base-v2 |
| NomicBERT encoder | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| NomicBERT MoE encoder | SentencePiece | Post-LN, MoE 8-expert top-2, GELU, RoPE | nomic-embed-text-v2-moe |
| ModernBERT encoder | BPE | Pre-LN, GeGLU, RoPE, per-layer theta | gte-modernbert-base |
| GTE v1.5 encoder | WordPiece | Post-LN, GeGLU, NTK RoPE | gte-base/large-en-v1.5 |
| DeBERTa-v2 encoder | WordPiece | Post-LN, c2p/p2c disentangled attn | mxbai-rerank-xsmall/base-v1 |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU | Harrier-270M, EmbeddingGemma-300m |
| BidirLM-Omni | GPT-2 BPE | Bidirectional Qwen3, MRoPE, DeepStack | BidirLM-Omni-2.5B |
| ViT (SigLIP/CLIP) | — | Conv2D patch embed, CLS/mean/attn pool | siglip-base, clip-vit-base |
| CLIP text | CLIP BPE | Pre-LN, causal mask, EOS pool | clip-text-base/large |
| CNN (SCRFD/YuNet) | — | FPN, anchor decode, NMS | scrfd-det-10g, yunet |
| LFM2.5 bidirectional | GPT-2 BPE | ShortConv+GQA, RoPE, SwiGLU, BiLSTM, GLiNER head | gliner-lfm (NER) |
| DeBERTa-v3 + GLiNER | SentencePiece | Disentangled c2c/c2p/p2c attn, 768→512 proj, BiLSTM, markerV0 | gliner-deberta (NER, Apache-2.0) |
| CNN (ArcFace) | — | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |
| DeiT+TrOCR | — | ggml graph encoder + decoder | pix2tex-mfr |
| HMER | — | DenseNet-121 + GRU attention | hmer (handwritten math) |
| BTTR | — | DenseNet + Transformer decoder | bttr (handwritten math) |
| InternVL2.5 | SentencePiece BPE | InternViT-300M + pixel unshuffle + InternLM2.5-1.8B (GQA 16/8, SwiGLU, KV cache) | internvl2.5-2b (VLM OCR) |
| PARSeq | — (char-level) | ViT-12L encoder + 1-layer two-stream decoder, GELU, 94-char ASCII | parseq (scene text) |

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crisp_audio | CrispASR build | shared library |

## File layout (current)

```
CrispEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md                     architecture + roadmap (this file)
├── HISTORY.md                  completed milestones
├── LEARNINGS.md                technical notes
├── PERFORMANCE.md              benchmarks
├── ggml/                       (submodule)
├── src/
│   ├── crispembed.h            C API
│   ├── crispembed.cpp          encoder graph + C API impl
│   ├── decoder_embed.{h,cpp}   decoder graph (Qwen3/Gemma3/BidirLM)
│   ├── bidirlm_vision.cpp      BidirLM-Omni vision tower
│   ├── bidirlm_audio.cpp       BidirLM-Omni audio tower
│   ├── vit_embed.{h,cpp}       SigLIP/CLIP ViT vision encoder
│   ├── clip_text_embed.{h,cpp} CLIP/SigLIP text encoder
│   ├── cnn_embed.{h,cpp}       SCRFD/YuNet/ArcFace/SFace
│   ├── image_preprocess.{h,cpp} C++ image preprocessor
│   ├── math_ocr.{h,cpp}        DeiT+TrOCR printed math OCR
│   ├── hmer_ocr.{h,cpp}        HMER handwritten math OCR
│   ├── bttr_ocr.{h,cpp}        BTTR handwritten math OCR
│   ├── internvl2_ocr.{h,cpp}   InternVL2.5-2B VLM OCR (KV cache)
│   ├── parseq_ocr.{h,cpp}     PARSeq scene text OCR (ViT + 2-stream decoder)
│   ├── tokenizer.h             WordPiece + SentencePiece + BPE
│   ├── tokenizer_bpe.cpp       GPT-2 byte-level BPE
│   ├── model_mgr.{h,cpp}       registry + auto-download
│   └── core/                   shared helpers (from CrispASR)
├── examples/
│   ├── cli/main.cpp            CLI binary
│   └── server/server.cpp       HTTP server (4 API dialects)
├── models/
│   ├── convert-bert-to-gguf.py
│   ├── convert-decoder-embed-to-gguf.py
│   ├── convert-siglip-to-gguf.py
│   ├── convert-clip-text-to-gguf.py
│   ├── convert-face-to-gguf.py
│   ├── convert-hmer-to-gguf.py
│   ├── convert-bttr-to-gguf.py
│   └── upload_to_hf.py
├── python/crispembed/          ctypes wrapper
├── crispembed-sys/             Rust FFI bindings
├── crispembed/                 Rust safe wrapper
├── flutter/crispembed/         Dart/Flutter FFI plugin
├── tools/quantize.cpp          C++ quantizer
└── tests/                      parity + benchmark scripts
```

## Pending roadmap

### Remaining work (prioritised)

#### Bugs / polish

- [~] **Layout detection decoder** — ImageNet normalization removed (model uses
  do_normalize=False). Encoder parity: ip3 cos=0.76→0.98 after fix. Remaining:
  (1) bilinear resize interpolation vs PIL (ip3 cos=0.76, enc_output cos=0.98),
  (2) decoder cross_out cos=0.01 — deformable attention sampling bug.
- [x] **Surya detector GPU** — `surya_det.cpp` now uses `ggml_backend_init_best()`
  (`SURYA_DET_FORCE_CPU=1` pins CPU for parity debugging). Metal verified on M1:
  F16/Q8_0 run on the GPU, heatmap parity vs CPU to ~3 decimals (sub-pixel box
  drift from F16 matmul accumulation). CUDA build works on Kaggle P100
  (GGML_CUDA_NO_VMM=ON): Q8_0+F16 both detect 17 regions correctly.

#### OCR models — remaining

- [ ] Keyven/german-ocr-3.1 (2B, Apache-2.0) — Qwen2.5-VL-2B fine-tune
- [ ] Qari-OCR (2B, Apache-2.0) — Arabic with diacritics
- [ ] Granite Vision 3.3-2B (3B, Apache-2.0) — OCRBench 852

#### Nice-to-have

- [ ] CrispCalc Dart catalog entries for InternVL2 (`OcrModelVariant`)
- [ ] Qwen3-VL multimodal (low priority, reuse BidirLM-Omni scaffolding)

### Completed (v0.8.0)

#### Performance
- [x] Batched decoder graph (~3x speedup)
- [x] KV prefix sharing for batched decoder
- [x] SigLIP attention pooling head
- [x] Layout decoder → ggml graph (2x speedup)
- [x] PPFormulaNet-L BLAS + Q8_0 (53s)
- [x] Streaming ColBERT SSE (`POST /colbert/score`)

#### Models
- [x] CLIP text + SigLIP-large + CLIP-large
- [x] ViT quantization, YuNet, SFace/AuraFace quantization
- [x] Face model quantized graph replay
- [x] ViT parity fix (patch permute bug)
- [x] Nomic v2 MoE encoder (cos=1.000)
- [x] LoRA hot-swap + Jina v5 live-test (4 adapters cos >= 0.998)
- [x] GLiNER DeBERTa-v3 NER (Apache-2.0)
- [x] ColBERT MaxSim scoring (C API + server endpoint)
- [x] BGE-M3 crash fix (clip_text guard)
- [x] LoRA quantizer fix (preserve A/B at F16)
- [x] Face pipeline Python wrapper
- [x] Q8_0 layout model verified
- [x] Layout-heron GGUFs uploaded to HuggingFace
- [x] CrispCalc Dart catalog (layout-heron)

#### OCR — ported
- [x] Qwen2.5-VL-3B (Keyven/german-ocr-3 base)
- [x] InternVL2.5-2B (2.1B, MIT, cos=1.000)
- [x] InternVL2-1B (0.9B, MIT) + C++ tokenizer encode
- [x] PARSeq (24M, Apache-2.0, scene text)
- [x] GLM-OCR (0.9B, MIT, OmniDocBench #1)
- [x] GOT-OCR2 (0.7B, Apache-2.0, full parity)
- [x] MixTex full E2E parity — cos=1.000000 encoder + decoder (6 bugs fixed)
- [x] Nanonets-OCR-s (3B, Apache-2.0, Qwen2.5-VL fine-tune)
- [x] Surya detector (parity verified, stb_image, Q8_0/Q4_K working)

---

## Implementation blueprints

Detailed specs for pending roadmap items. Each blueprint is self-contained
so a fresh agent can implement it independently.

### Blueprint: LoRA adapter hot-swap — DONE

Implemented June 2026. See HISTORY.md. C API: `crispembed_set_lora()`,
`crispembed_list_lora()`. Pre-compute merge `W' = W + (a/r)*B@A` on CPU.

---

### Blueprint: Nomic v2 MoE encoder — DONE

Implemented June 2026. cos=1.000000 vs HuggingFace. See HISTORY.md.
Key learning: NomicBERT v2-moe has Wqkv + out_proj biases (v1.5 does not).
Parity harness extended with `--arch nomic` for QKV split + MoE expert diff.

---

### Blueprint: Batched decoder graph — DONE

Implemented June 2026. `decoder_encode_tokens_batch()` in decoder_embed.cpp.
Block-diagonal F32 causal mask, per-text RoPE, last-token pooling. ~3x speedup.

---

### Blueprint: KV cache for prefix-shared decoder batches

**Goal**: When N texts share a prompt prefix (e.g. Jina v5 instruction
`"Retrieve semantically similar text.\nQuery: "`), compute KV for the
prefix once and reuse across the batch, saving ~40% of compute.

**Current state**: `decoder_encode_tokens_batch()` (decoder_embed.cpp:1158)
pads all sequences to T_max and builds a block-diagonal mask. Each text
recomputes the full prefix independently.

**Approach**:
1. Detect common prefix: compare token IDs across batch, find longest
   shared prefix length `P`. Typical Jina v5 prefix is ~15 tokens.
2. Compute prefix KV: Run a single forward pass for `[P]` tokens.
   Extract K/V tensors for each layer → `prefix_kv[layer] = {K, V}`.
3. Extend graph: For each text's remaining `T-P` tokens, use pre-computed
   prefix K/V. Modify `ggml_flash_attn_ext` call to pass concatenated
   K = `[prefix_K ; text_K]`, V = `[prefix_V ; text_V]`.
4. Mask: Each text attends causally to `[prefix + own_tokens]`, not to
   other texts' tokens.

**Challenge**: ggml's `flash_attn_ext` takes K/V as single tensors. Need
either: (a) pre-concatenate K/V on CPU before graph compute, or (b) use
`ggml_concat` in-graph (adds one concat op per layer, negligible cost).

**Files**: `src/decoder_embed.cpp`, `src/decoder_embed.h`

**Effort**: Medium (3-4 days). Prefix detection is trivial; KV caching
mechanics are well-understood from LLM inference.

---

### Blueprint: Batched decoder improvements (F16 mask + Gemma3 NaN fix)

**Goal**: Two targeted fixes for the batched decoder path.

**F16 attention mask**: Current mask is F32 (`ggml_new_tensor_2d(gctx,
GGML_TYPE_F32, T_total, T_total)` at decoder_embed.cpp:1297). For
T_total=512 that's 1 MB; for T_total=2048 it's 16 MB. `ggml_flash_attn_ext`
supports F16 `kq_b` natively. Change to:
```cpp
ggml_tensor * attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T_total, T_total);
```
CPU-side mask fill: cast `-INFINITY` and `0.0f` to `ggml_fp16_t` via
`ggml_fp32_to_fp16()`. Reduces mask memory 2x.

**Gemma3 NaN fix**: Some Gemma3 GGUFs produce NaN embeddings on certain
inputs. Root cause: `(1+w)*rms_norm(x)` can overflow when `w` is large
and `rms_norm(x)` has extreme values. The Ollama-format GGUFs pre-bake
`1+w` into norm weights (flag `gemma_norm=false`), but CrispEmbed-native
GGUFs use raw `w` with `gemma_norm=true`. Fix: clamp RMSNorm output to
`[-65504, 65504]` (F16 max) before the `(1+w)` multiply, or detect NaN
after the first layer and fall back to F32 compute for that batch.

**Files**: `src/decoder_embed.cpp`

**Effort**: Low (1 day). Both are mechanical changes.

---

### Blueprint: Streaming ColBERT late interaction scoring

**Goal**: Add `/colbert/score` endpoint to the HTTP server for MaxSim
scoring between a query's multi-vector embeddings and pre-stored document
token vectors, with streaming partial results.

**Current state**: `crispembed_encode_multivec()` (crispembed.cpp) produces
per-token ColBERT embeddings for BGE-M3. No server endpoint for scoring.
No MaxSim implementation.

**MaxSim**: For query tokens Q `[nq, d]` and doc tokens D `[nd, d]`:
`score = sum_i(max_j(cos(Q[i], D[j])))`. O(nq * nd * d) per document.

**Step 1 -- C API**: Add `crispembed_colbert_score(ctx, query_vecs, n_query,
doc_vecs, n_doc, dim)` returning a float score. Pure CPU computation (no
ggml graph needed — it's just a batched dot product + max reduction).

**Step 2 -- Server endpoint** (`examples/server/server.cpp`):
- `POST /colbert/score`: JSON body with `query_vecs` (or `query_text` to
  encode on the fly), `documents` (list of pre-computed doc vecs or raw
  texts). Returns ranked list with scores.
- SSE streaming: After each document's score is computed, emit a partial
  result via `text/event-stream`. Client sees progressive ranking.

**Step 3 -- Batched scoring**: For K documents, parallelize MaxSim across
documents (each document is independent). Use OpenMP.

**Files**: `src/crispembed.cpp` (C API), `src/crispembed.h`,
`examples/server/server.cpp`

**Effort**: Medium (2-3 days). MaxSim is simple math; SSE streaming is
the main engineering work.

---

### Blueprint: WASM build target

**Goal**: Compile CrispEmbed to WebAssembly for browser-based embedding.

**Current state**: ggml has Emscripten support (whisper.cpp WASM demo
exists). CrispEmbed uses standard ggml APIs. CMakeLists.txt has no WASM
target.

**Step 1 -- CMake toolchain**:
- Add `cmake/wasm.cmake` toolchain file for Emscripten.
- Build static library + JS wrapper. Disable audio/face/vision paths
  (browser doesn't need them). Target: `crispembed.js` + `crispembed.wasm`.
- Flags: `-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -msimd128` (relaxed SIMD).
- Threading: `-pthread -s SHARED_MEMORY=1` requires `SharedArrayBuffer`
  (needs COOP/COEP headers). Start single-threaded, add workers later.

**Step 2 -- JS API** (`wasm/crispembed.js`):
- `async function loadModel(url)` — fetch and load GGUF into WASM heap.
- `function encode(text)` — returns Float32Array of embedding.
- `async function encodeBatch(texts)` — batch encode.

**Step 3 -- Demo page** (`wasm/demo.html`):
- Load a small model (all-MiniLM-L6-v2, 24 MB Q4_K).
- Text input → encode → show embedding dims + cosine vs reference.

**Challenges**:
- Memory: 2 GB WASM limit. MiniLM Q4_K fits (24 MB); larger models need
  streaming GGUF or won't fit.
- First load: GGUF download + WASM compilation. Use `caches` API.
- No GPU: CPU-only, but ggml SIMD kernels should be fast enough for
  small models.

**Files**: `cmake/wasm.cmake`, `wasm/crispembed.js`, `wasm/demo.html`,
`CMakeLists.txt` (add wasm target)

**Effort**: Medium (3-4 days). Main risk is ggml WASM compatibility.

---

### Blueprint: INT4 GGUF for face models

**Goal**: Quantize Conv2D weights in face models (SCRFD, AuraFace, SFace)
to Q4_K for 4x size reduction.

**Current state**: `tools/quantize.cpp` skips tensors with ndims > 2
(line 172: "skipping N-D tensor (conv2d)"). Conv2D weights are stored as
2D `[OC, IC*KH*KW]` in the GGUF (flattened from 4D for quantization
compatibility). The graph replayer (`src/cnn_embed.cpp`) reshapes them
back to 4D and dequants Q→F32 before `ggml_conv_2d`.

**The fix is simple**: The quantizer already handles 2D tensors. The issue
is that face model conv weights are ALREADY 2D in the GGUF (flattened by
the converter). The quantizer skips them because of a name-based guard
(`patch_embed` check) or because certain tensors are small (< 256 elements).

**Step 1 -- Quantizer** (`tools/quantize.cpp`):
- Remove or relax the `patch_embed` / ndims guard for face models.
- Add `--include-conv` flag to opt-in to conv weight quantization.
- Keep bias/norm tensors at F32 (1D tensors, already excluded).

**Step 2 -- Graph replayer** (`src/cnn_embed.cpp`):
- Already handles Q→F32 dequant (commit 2fcde98). Verify it works for
  Q4_K (not just Q8_0).

**Step 3 -- Testing**: Quantize YuNet, SCRFD, AuraFace, SFace to Q4_K.
Compare detection boxes (IoU ≥ 0.95) and recognition embeddings (cos ≥ 0.99).

**Expected sizes**:
- AuraFace: 249 MB → ~65 MB (Q4_K)
- SCRFD: 17 MB → ~5 MB
- SFace: 37 MB → ~6 MB (already have Q8_0 at 10 MB)

**Files**: `tools/quantize.cpp`, `src/cnn_embed.cpp` (verify only)

**Effort**: Low (1 day). The infrastructure exists; this is mostly
testing.

---

### Blueprint: Live-test LoRA with Jina v5

**Goal**: End-to-end verification that LoRA hot-swap works correctly with
real Jina v5 adapters, not just unit tests.

**Current state**: LoRA infrastructure is implemented (decoder_embed.cpp,
C API, Python wrapper). Converter supports `--lora-mode=separate`. But no
end-to-end parity test exists against HuggingFace per-adapter outputs.

**Step 1 -- Convert**: `python models/convert-decoder-embed-to-gguf.py
--model jinaai/jina-embeddings-v5-text-small --lora-mode=separate --output
jina-v5-small-lora.gguf`

**Step 2 -- Per-adapter parity** (new test `tests/test_lora_parity.py`):
- For each adapter (retrieval, classification, clustering, text-matching):
  - HF: `model.set_adapter(name)` → encode test texts
  - CrispEmbed: `crispembed_set_lora(ctx, name)` → encode same texts
  - Compare: cos >= 0.999 per text
- Also test: switching adapters mid-session, listing available adapters,
  setting invalid adapter name (should fail gracefully).

**Step 3 -- Python wrapper test**: Verify `ctx.set_lora("retrieval")`,
`ctx.list_lora()`, `ctx.lora` property work end-to-end.

**Files**: `tests/test_lora_parity.py` (new), test data

**Effort**: Low (1 day). Infrastructure exists; this is testing only.

---

### Blueprint: 3D tensor quantization for MoE experts

**Goal**: Extend `tools/quantize.cpp` to quantize 3D tensors (MoE expert
weights) instead of copying them as F32.

**Current state**: Quantizer skips ndims > 2 (line 172). For nomic-v2-moe,
expert_fc1 `[8, 3072, 768]` and expert_fc2 `[8, 768, 3072]` stay F32,
limiting Q8_0 to 1122 MB (vs potential ~600 MB if experts were quantized).

**Approach**: Iterate over the outermost dimension (n_experts=8) and
quantize each 2D slice `[inter, hidden]` independently. The ggml quantize
functions work on flat arrays with a specified number of elements per row.

**Step 1** (`tools/quantize.cpp`):
```cpp
if (ndims == 3) {
    // Quantize each 2D slice of the 3D tensor
    int64_t slice_size = ne[0] * ne[1];
    for (int64_t k = 0; k < ne[2]; k++) {
        quantize_slice(src + k * slice_size, dst + k * q_slice_size,
                       ne[0], ne[1], target_type);
    }
}
```

**Step 2** -- Verify: `ggml_mul_mat_id` must support quantized `as` tensor.
Check ggml source: `ggml_mul_mat_id` calls `ggml_compute_forward_mul_mat`
per expert, which supports quantized weights. Should work.

**Step 3** -- Test: Quantize nomic-v2-moe with 3D quant, verify cos >= 0.999
(Q8_0) and cos >= 0.96 (Q4_K).

**Expected sizes** (nomic-v2-moe):
- F32: 1818 MB → Q8_0: ~600 MB (vs current 1122 MB)
- Expert weights: 6 layers × 2 × [8, 3072, 768] = ~845 MB F32 → ~225 MB Q8_0

**Files**: `tools/quantize.cpp`

**Effort**: Low (half day). Small code change + testing.

---

### Blueprint: Face pipeline Python wrapper — DONE

Implemented. `CrispFace` (detect/encode) and `CrispFacePipeline` (detect→align→encode,
match, from_registry with defaults) fully wired in `python/crispembed/_binding.py`.
Feature parity with CLI (`--face`, `--detect`, `--face-pipeline`) and server
(`POST /face/detect`, `POST /face/pipeline`).

---

### Blueprint: surya-ocr-2 (full-page OCR, 91 languages) — DONE (Metal GPU on M1 + CUDA on Kaggle both working)

**Goal**: Port surya-ocr-2 for multilingual full-page OCR with text
detection, recognition, and layout analysis.

**Source**: datalab-to/surya-ocr-2 (0.7B, OpenRail-M — free for <$5M)
GitHub: https://github.com/VikParuchuri/surya

**Architecture**: Two separate models:
1. **Detector** (38M params, 147 MB F32, 73 MB F16):
   EfficientViT-Large segformer. Input [3,1200,1200] → 2-channel
   heatmap [2,300,300] (text line, separator) → polygon bounding boxes.
   - Stem: Conv3x3(s2)+BN+Hardswish + ConvBlock residual → [32,600,600]
   - Stage 0: 2× FusedMBConv (expand 16x/4x) → [64,300,300]
   - Stage 1: 2× FusedMBConv (expand 16x/4x) → [128,150,150]
   - Stage 2: 7× MBConv (expand 16x/4x, ReLU6) → [256,75,75]
   - Stage 3: 1× MBConv (expand 24x) + 6× EfficientVitBlock → [512,38,38]
     (each VitBlock = LiteMLA linear attention + MBConv, with residuals)
   - DecodeHead: 4× Linear proj → bilinear upsample → cat(reversed) →
     Conv1x1+BN+ReLU → Conv1x1 classifier → sigmoid

2. **Recognizer** (0.7B): Qwen3.5-VL fine-tune. GGUF already exists at
   datalab-to/surya-ocr-2-gguf. Runs via llama-server with OpenAI API.
   Prompts: "OCR this block image to HTML." / "OCR this image to HTML.
   Each block is a div with data-label and data-bbox (x0 y0 x1 y1,
   normalized 0-1000)."

**Key op — LiteMLA** (O(N) linear attention):
  Q,K = ReLU(qkv_split), V_pad = pad(V,1)
  KTV = K^T @ V_pad  (head_dim × (head_dim+1), spatial aggregation)
  out = Q @ KTV / (Q @ K^T @ ones + eps)  (per-position normalization)
  Multi-scale: DW-Conv5x5 + grouped-Conv1x1 aggregation, cat with original

**Status** (2026-06-11):
- [x] Reference dumper: `tools/dump_surya_det_reference.py` (52 activations)
- [x] GGUF converter: `models/convert-surya-det-to-gguf.py` (148 tensors, BN folded)
- [x] C++ engine: `src/surya_det.{h,cpp}` (CPU-scalar, compiles+runs)
- [x] Test binaries: `test-surya-det`, `test-surya-det-diff`
- [x] Model registry entry: `surya-det`
- [x] Parity verification — heatmap max=0.9649 exact, mean=0.0113 exact
- [x] Heatmap → polygon post-processing (connected components + bbox)
- [x] Move encoder to ggml graph — 13min→1min (13x). Stages 0-2 + block0: 17s via graph. LiteMLA + decode scalar.
- [x] Upload GGUF to HuggingFace — https://huggingface.co/cstr/surya-det-GGUF (F32, F16, Q8_0, Q4_K)
- [x] Q8_0/Q4_K crash fix — dequant Q→F32 before reshape in g_conv (ne[0] block alignment)
- [x] GPU testing — Metal on M1 verified (`ggml_backend_init_best`). Stage 0-2
  graph ~4.4s GPU vs ~5.9s CPU, stage-3 block0 ~0.75s vs ~0.94s; LiteMLA +
  decode head remain CPU-scalar. F16 + Q8_0 both run, heatmap parity vs CPU to
  ~3 decimals (sub-pixel box drift from F16 matmul accumulation). Weight reads
  routed through `ggml_backend_tensor_get` so scalar paths work off GPU buffers.
  CUDA build works on Kaggle P100 (GGML_CUDA_NO_VMM=ON to bypass the upstream
  `CUDA::cuda_driver` link issue): Q8_0+F16 both detect 17 regions correctly.
- [x] Image format support: PNG/JPG via stb_image done

**GGUFs**: `cstr/surya-det-GGUF` — F32 (147 MB), F16 (73 MB), Q8_0 (41 MB), Q4_K (30 MB)

**Files**: `src/surya_det.{h,cpp}`, `models/convert-surya-det-to-gguf.py`,
`tools/dump_surya_det_reference.py`, `tests/test_surya_det{,_diff}.cpp`

---

### Blueprint: MixTex ZhEn-Latex-OCR (86M, Apache-2.0) — DONE

**Goal**: Chinese+English math LaTeX OCR. Smallest new model, introduces
Swin encoder as new building block.

**Source**: MixTex/ZhEn-Latex-OCR (85.9M, Apache-2.0)

**Architecture**: VisionEncoderDecoderModel
1. **Encoder**: Swin-Tiny (28M params)
   - Patch embed: Conv2d [96, 3, 4, 4] stride 4
   - 4 stages: depths=[2,2,6,2], heads=[3,6,12,24], dims=[96,192,384,768]
   - Window size=7, shifted window attention, relative position bias
   - Patch merging (2×2 concat + Linear) between stages
   - Output: [N, 768] where N=H/32×W/32 (180 for 400×500 input)

2. **Decoder**: RoBERTa 4-layer with cross-attention (57M params)
   - hidden=768, 12 heads, FFN=3072, GELU
   - Post-LN: self-attn → LN → cross-attn → LN → FFN → LN
   - BPE tokenizer, 25681 tokens (LaTeX + CJK)
   - max_position=300, greedy decode

**Status** (2026-06-13):
- [x] Reference dumper: `tools/dump_mixtex_reference.py`
- [x] GGUF converter: `models/convert-mixtex-to-gguf.py` (345 tensors)
- [x] GGUF files: F32=329MB, F16=165MB, Q8_0, Q4_K at `/mnt/storage/gguf-models/`
- [x] C++ engine: `src/mixtex_ocr.{h,cpp}` — runs end-to-end, Swin+RoBERTa
- [x] Parity test — cos=1.000000 on all encoder blocks (non-shifted + shifted)

**Key new op**: Swin shifted-window attention with relative position bias.
Pad → cyclic shift → window partition → local MHSA + RPB lookup → window reverse → reverse shift → unpad.

**Bugs found and fixed** (3 bugs in shifted-window attention):
1. Cyclic shift sign convention — `cyclic_shift(+s)` ≠ `torch.roll(-s)`, signs inverted
2. Pad-then-shift order — HF pads FIRST then shifts; C++ was reversed
3. GELU variant — tanh approximation → exact erf matching `nn.GELU()`

**Files**: `tools/dump_mixtex_reference.py`, `models/convert-mixtex-to-gguf.py`

**Effort**: Medium (3-4 days). Swin encoder is new; decoder is same as TrOCR.

---

### Blueprint: GLM-OCR (0.9B, MIT, GGUF exists)

**Goal**: Integrate GLM-OCR for general document OCR. GGUF already
converted by ggml-org — may only need inference integration, not
conversion.

**Source**: zai-org/GLM-OCR (0.9B, MIT)
GGUF: ggml-org/GLM-OCR-GGUF (Q8_0: 950 MB, F16: 1.79 GB)

**Architecture**: CogViT vision encoder + GLM-0.5B language decoder.
Lightweight cross-modal connector with token downsampling. 8 languages.

**Approach**: Since GGUF already exists and runs via llama-server, the
fastest path may be to study the GGUF tensor layout and write a native
CrispEmbed inference engine that loads it directly, rather than
re-converting.

**Files**: `src/glm_ocr.{h,cpp}`, possibly no converter needed

**Effort**: Medium (3-4 days). GGUF exists; main work is C++ inference.

---

### Blueprint: GOT-OCR2 (0.7B, Apache-2.0) — DONE

Implemented June 2026. Full parity on vision encoder (SAM ViT-B with
windowed + global attention, decomposed RPE) + neck + downsample +
projector + Qwen2-0.5B LLM decoder. All checkpoints cos ≥ 0.999.

**Architecture**: SAM ViT-B (12L, 768d, window_size=14, global at
[2,5,8,11]) → neck (Conv→LN2d→Conv→LN2d) → downsample (Conv 256→512→1024,
stride 2) → Linear(1024,1024) projector → Qwen2-0.5B (24L, 1024d, MHA
16/16, standard RoPE, SiLU SwiGLU) → autoregressive generation.

**Key learnings**: Per-layer diff comparison must happen inside the layer
loop (not after), since `hidden` is overwritten by each subsequent layer.
Vision uses LayerNorm+GELU (not RMSNorm+SiLU). Windowed layers need CPU
LN1 before partition, then pass both LN'd and original data into graph.

**Files**: `src/got_ocr.{h,cpp}`, `models/convert-got-ocr-to-gguf.py`,
`tools/dump_got_ocr_reference.py`, `tests/test_got_ocr_diff.cpp`

---

### Blueprint: Nanonets-OCR-s — DONE

Actual model: `nanonets/Nanonets-OCR-s` (Qwen2.5-VL-3B fine-tune, Apache-2.0,
71K downloads). Same architecture as existing qwen2vl engine — no new code.
Converted via Kaggle kernel (VPS too tight for 3B model).
GGUFs at `cstr/nanonets-ocr-s-crispembed-GGUF` (F16/Q8_0/Q4_K).
Registry entry: `nanonets-ocr-s`.

---

### Blueprint: Qari-OCR (Arabic, 2B, Apache-2.0)

**Goal**: Arabic OCR with diacritics support.

**Source**: NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct (Apache-2.0)
Fine-tune of Qwen2-VL-2B-Instruct for Arabic text.

**Note**: The published fine-tune was trained from a 4-bit quantized
base (unsloth/bnb). For GGUF porting, may need to source fp16 weights
or re-fine-tune from the full-precision Qwen2-VL base.

**Effort**: Medium (3-4 days). Same Qwen2-VL base as Nanonets — share
infrastructure.

---

### Blueprint: Keyven/german-ocr (German docs, Apache-2.0) — MERGED TO MAIN

**Goal**: German business document OCR with structured JSON output.

**Base model**: Qwen2.5-VL-3B-Instruct (Keyven/german-ocr-3 is a
fine-tune of this). Architecture: 32-layer ViT (1280d) + spatial merger
(2×2→2048d) + 36-layer Qwen2.5 LLM (2048d, GQA 16/2, mRoPE).

**Status (2026-06-12): Standalone CLI pipeline complete. Needs 16+ GB RAM.**

| Step | Status | Notes |
|------|--------|-------|
| C++ inference engine | DONE | `src/qwen2vl_ocr.{h,cpp}` — vision + LLM + generation |
| GGUF converter | DONE | `models/convert-qwen2vl-to-gguf.py` (lazy tensor, with tokenizer) |
| Reference dumper | DONE | `tools/dump_qwen2vl_reference.py` (safetensors, ~600MB peak) |
| Parity: vision encoder | DONE | 32/32 layers cos=1.000 |
| Parity: spatial merger | DONE | cos=1.000, max_abs=6e-4 |
| Parity: LLM decoder | DONE | 2/2 layers cos=1.000 with mRoPE |
| E2E generation (Q4_K) | DONE | "Um die Rechnung im Bild als" — coherent German |
| Quantization | DONE | Q8_0 (3.9GB), Q4_K (2.6GB, vision Q8_0 floor) |
| HuggingFace upload | DONE | `cstr/qwen2.5-vl-3b-crispembed-GGUF` (F16 + Q8_0 + Q4_K) |
| Wire into C ABI | DONE | `crispembed.cpp` dispatch, arch="qwen2vl" auto-detect |
| CLI + model registry | DONE | `model_mgr.cpp` entry, `--ocr` auto-dispatch |
| Unit + smoke tests | DONE | `test_qwen2vl.cpp` 14/14 pass (unit + Q4K smoke) |
| C++ image preprocessor | DONE | `image_preprocess.cpp` wired into recognize_raw() |
| BPE tokenizer in C++ | DONE | Vocab+merges in GGUF, loaded at init, set_prompt() tokenizes |
| GPT-2 byte decoder | DONE | Output is decoded UTF-8 text, not raw token IDs |
| KV cache | DONE | Prefill extracts K/V, decode uses cache (O(1)/token) |
| Load Keyven fine-tune | REVISED | german-ocr-3.1 is 2B (not 3B) — different LLM config, needs investigation |
| Windowed ViT attention | TODO | Currently full attention all layers (correct but slower) |
| Python bindings | TODO | Wire via `CrispMathOcr` auto-dispatch |
| CrispCalc catalog | TODO | `OcrModelVariant` entries |

**Standalone CLI now works** (on machines with 16+ GB RAM):
```
crispembed --ocr qwen2vl-3b image.png
# or with custom prompt:
qwen2vl_ocr_set_prompt(ctx, "Extrahiere die Rechnung als JSON.");
```
The Q4_K model (2.6 GB) + ggml compute graph needs ~5 GB peak RAM.
OOMs on 8 GB machines — use Kaggle (16 GB) or desktop for inference.

**Next priority:** Load Keyven fine-tune weights + test on real invoices.

**GGUFs**: `cstr/qwen2.5-vl-3b-crispembed-GGUF` on HuggingFace:
- `qwen2.5-vl-3b-f16.gguf` — 7.57 GiB (no tokenizer — needs re-convert)
- `qwen2.5-vl-3b-q8_0.gguf` — 3.93 GiB (v2: with BPE tokenizer)
- `qwen2.5-vl-3b-q4_k.gguf` — 2.61 GiB (v2: with BPE tokenizer)

**Key learnings:**
- Vision weights need Q8_0 floor in quantizer (Q4_K degrades OCR)
- `ggml_set_output()` required on intermediates to prevent memory reuse
- GQA interleave via 4D reshape+repeat (not `ggml_repeat` which tiles)
- mRoPE uses neghalf rotation with `GGML_ROPE_TYPE_MROPE`
- Token splicing: `x = embed * keep_mask + image_patches`
- `AutoConfig` varies by transformers version — read config.json directly
- Kaggle: always use CrispASR kaggle_harness.py, don't pip install torch

**Files**: `src/qwen2vl_ocr.{h,cpp}`, `models/convert-qwen2vl-to-gguf.py`,
`tools/dump_qwen2vl_reference.py`, `tools/qwen2vl_tokenize.py`,
`tests/test_qwen2vl{,_diff,_e2e}.cpp`, `tools/kaggle/qwen2vl-convert/`
