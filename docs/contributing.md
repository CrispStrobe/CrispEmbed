# Contributing: Adding a New Model to CrispEmbed

This guide covers adding a new model backend (OCR, embedding, face, etc.) to CrispEmbed. Follow the same six-step pattern used for every model in the codebase.

## Checklist Summary

1. [ ] Write the C inference engine (`src/yourmodel.{h,cpp}`)
2. [ ] Write the GGUF converter (`models/convert-yourmodel-to-gguf.py`)
3. [ ] Write the reference dumper (`tools/dump_yourmodel_reference.py`)
4. [ ] Wire into the C ABI (`src/crispembed.cpp`)
5. [ ] **Wire into orchestrator** (OCR only): enum + `map_engine` + header comment + `ocr_orchestrator.cpp` dispatch + CrispSorter
6. [ ] Wire into CMake + CLI + model registry + all bindings (Python, Rust, Dart)
7. [ ] **Regenerate SHA-256 pins** (`python tools/fetch_model_hashes.py`) — an
       unpinned registry URL refuses to download
8. [ ] Verify parity, quantize, update CrispCalc catalog
9. [ ] `tools/format.sh --fix`, and add a row to `PLAN.md`'s active-work table

---

## Step 1: C Inference Engine

Create `src/yourmodel.h` and `src/yourmodel.cpp`.

**Header pattern** — expose a C API:
```c
typedef struct yourmodel_context yourmodel_context;
typedef struct yourmodel_hparams { /* ... */ } yourmodel_hparams;

yourmodel_context * yourmodel_init(const char * model_path, int n_threads);
void yourmodel_free(yourmodel_context * ctx);
const yourmodel_hparams * yourmodel_get_hparams(const yourmodel_context * ctx);
const char * yourmodel_recognize(yourmodel_context * ctx, /* inputs */, int * out_len);
const char * yourmodel_recognize_raw(yourmodel_context * ctx, const uint8_t * px, int w, int h, int ch, int * out_len);
```

**Implementation pattern:**
- Load GGUF via `core_gguf::open_metadata()` + `core_gguf::load_weights()`
- Map GGUF tensor names to struct fields via `map_tensors()`
- Implement encoder/decoder with CPU scalar ops (`layernorm_cpu`, `linear_cpu`, `conv2d_cpu`, `mha_1q_cpu`)
- Dequantize on the fly via `to_f32()` helper (supports F32, F16, Q8_0, Q4_K, etc.)
- Greedy decode loop with KV caching for autoregressive models

**Reusable utilities** (copy from existing backends):
- `to_f32()` — dequant any ggml type to float
- `layernorm_cpu()` — standard layer normalization
- `linear_cpu()` — matrix-vector multiply
- `conv2d_cpu()` — 2D convolution with groups/padding/stride
- `mha_1q_cpu()` — single-query multi-head attention with KV cache
- `gelu()` — GELU activation (tanh approximation)

**Existing examples to follow:**
- `ppformulanet_ocr.cpp` — HGNetv2 CNN encoder + MBart decoder (simplest)
- `ppformulanet_l_ocr.cpp` — SAM-ViT encoder + MBart decoder (windowed attention)
- `posformer_ocr.cpp` — DenseNet encoder + Transformer decoder + ARM (coverage attention)
- `mixtex_ocr.cpp` — Swin-Tiny encoder + RoBERTa decoder (shifted-window attention)
- `got_ocr.cpp` — SAM ViT-B + Qwen2-0.5B VLM (windowed+global attention, KV cache)
- `internvl2_ocr.cpp` — InternViT + InternLM2.5 VLM (dynamic tiling, KV cache)
- `math_ocr.cpp` — DeiT encoder + TrOCR decoder (standard ViT)
- `hmer_ocr.cpp` — DenseNet + GRU attention decoder
- `layout_detect.cpp` — RT-DETRv2 object detection (deformable cross-attention)
- `surya_det.cpp` — EfficientViT segformer (LiteMLA linear attention, GPU backend)
- `nafnet_denoise.cpp` — NAFNet U-Net denoising CNN (SimpleGate + SCA, CPU-scalar)
- `scan_cleanup.cpp` — document preprocessing pipeline (classical + learned denoise)

## Step 2: GGUF Converter

Create `models/convert-yourmodel-to-gguf.py`.

**Pattern:**
1. Load weights from PyTorch/safetensors/Paddle checkpoint
2. Fold BatchNorm into Conv where applicable
3. Write hyperparameters as GGUF key-values (`yourmodel.encoder.*`, `yourmodel.decoder.*`)
4. Write tokenizer as `tokenizer.tokens` string array
5. Write tensors with a clean naming convention:
   - Encoder: `enc.layers.{i}.attn.qkv.weight`, `enc.layers.{i}.mlp.lin1.weight`, etc.
   - Decoder: `dec.layers.{i}.self_attn.q.weight`, `dec.layers.{i}.ffn.up.weight`, etc.

**Quantization flags:**
- `--fp16` — all tensors in FP16
- `--q8_0` — large matmuls in Q8_0, critical tensors in F16

**Critical tensors** (keep in F16 under quantization):
- Embeddings (token, position, patch)
- LayerNorm weights/biases (small, high sensitivity)
- Relative position bias tables (tiny, critical for attention geometry)
- LM head (directly determines output tokens)
- Bottleneck/projection layers (encoder→decoder bridge)

For Q4_K, use the C-side quantizer: `crispembed-quantize input-f16.gguf output-q4_k.gguf q4_k`

## Step 3: Reference Dumper

Create `tools/dump_yourmodel_reference.py`.

> **A GGUF tensor archive is not the only shape this can take, and for a small
> classifier it is the wrong one.** `tools/dump_fireredpunc_reference.py` +
> `tests/firered_punc_parity.py` are the lightweight variant: a flat text file
> of per-token ids, logits, argmax preds and decoded output, checked in at 9 KB
> under `tests/regression/fireredpunc/`, so the comparison runs with **no torch,
> no `transformers` and no checkpoint download**. Two things it is worth copying:
> the dumper derives its `BertConfig` from the checkpoint's own tensor shapes
> rather than a `config.json` (so it does not pull a 411 MB backbone just to
> overwrite it), and the comparator takes an explicit `min-cos` instead of
> hard-coding one — because the same 0.93 that means "structural bug" at f16
> means "the quantiser" at q4_k, and the way you tell them apart is by running
> the f16 arm of the same graph.
>
> That harness existed for one day before it found a bug that had shipped:
> the model prepends `[CLS]` and appends nothing, the port appended `[SEP]`, and
> being bidirectional that moved every token's distribution (f16 cos_min
> 0.931090 -> 1.000000). The tokenizer had been at 9/9 exact vs HuggingFace the
> whole time. **Tokenizer parity is not model parity.**

**Purpose:** Capture per-layer intermediate activations from the Python reference implementation (PyTorch/HF transformers), write them to a GGUF tensor archive. The C++ test binary then compares its own activations against these.

**What to capture:**
- `input_image` — preprocessed input tensor
- `enc_layer_{i}` — output after each encoder layer
- `proj_output` — encoder output after projection (decoder input)
- `dec_layer_{i}` — output after each decoder layer
- `logits_step0` — vocabulary logits at first decode step
- `generated_ids` — full greedy decode output

**Use forward hooks** (`register_forward_hook`) to capture without modifying model code.

## Step 4: Wire into C ABI

Edit `src/crispembed.cpp`:

1. `#include "yourmodel.h"`
2. Add to `enum math_ocr_type` (for OCR models) or create new dispatch enum
3. Add architecture detection in `detect_arch()`:
   ```cpp
   if (arch == "yourmodel") return MATH_OCR_YOURMODEL;
   ```
4. Add dispatch cases in: `init`, `free`, `recognize`, `recognize_gray`

**Grep for an existing model** (e.g., `MATH_OCR_PPFORMULANET_L`) and replicate every occurrence.

## Step 5: Orchestrator + map_engine (OCR models only)

If the model is an OCR / document-understanding engine, it must be wired
into the orchestrator pipeline so it can be selected as a stage engine
from C, Python, Rust, Dart, and CrispSorter.

### 5a. Add to `enum class engine` (`src/ocr_orchestrator.h`)

```cpp
enum class engine {
    dbnet_trocr,   // 0
    surya,         // 1
    // ...existing...
    yourmodel,     // N  ← append at the end, NEVER reorder existing entries
};
```

**CRITICAL**: Append only. Never insert in the middle or reorder — the
integer values are a shipped ABI used by CrispSorter and any C consumer
via `crispembed_ocr_stage.engine`.

### 5b. Add to `map_engine()` (`src/crispembed.cpp`)

This is the C-int → enum bridge used by `crispembed_ocr_pipeline_init_stages()`.
Without this, the engine is unreachable from the C API.

```cpp
static ocr_orchestrator::engine map_engine(int e) {
    // ...existing cases...
    case N:  return E::yourmodel;    // ← same int as enum position
}
```

**The int must match the enum's ordinal position.** Verify by checking
the `ocr_stage.engine` comment in `crispembed.h` — it documents the
canonical int→engine mapping. Update that comment when adding a new engine.

### 5c. Update the `ocr_stage.engine` comment (`src/crispembed.h`)

The `crispembed_ocr_stage` struct has a comment documenting the int mapping:
```c
int engine;  // 0=dbnet_trocr 1=surya 2=got ... N=yourmodel (matches map_engine)
```
**This comment is the contract.** CrispSorter and external consumers read it
to know which int selects which engine. If the comment disagrees with
`map_engine`, consumers will select the wrong engine. Always update both
together and verify they agree.

### 5d. Wire into `ocr_orchestrator.cpp`

1. Add `#include "yourmodel.h"` at the top
2. Add a context pointer to the `context` struct: `yourmodel_context * ym = nullptr;`
3. Add a `case engine::yourmodel:` block in `run_engine()` — lazy-load + recognize
4. Add `case engine::yourmodel: return "yourmodel";` in `engine_name()`
5. Add `if (ctx.ym) yourmodel_free(ctx.ym);` in `free()`
6. **If it is a whole-page VLM**: add it to `is_vlm_engine()`, apply
   `st.params.vlm_max_tokens` in the `run_engine()` block, and add a case to
   `crispembed_ocr_model_set_max_tokens()` in `src/crispembed.cpp`

> **⚠ `--ocr-max-tokens` is one flag over FOUR hand-maintained lists.**
> `is_vlm_engine()` (the canonical set), the `run_engine()` block (must read
> `st.params.vlm_max_tokens`), `crispembed_ocr_model_set_max_tokens()` (the
> single-model `--ocr` lane), and `is_vlm` in `examples/cli/main.cpp` (which
> decides whether `model_a` resolves down the VLM or the DETECTOR branch).
> Miss one and the flag is accepted, printed in `--help`, and does nothing —
> which reads as "the model rambles", not as a bug. Miss the CLI one and the
> engine is handed a detector model path and dies inside a vision graph.
> Four engines were missing from three of those lists until 2026-08-25, and
> both stage builders (CLI and server) hardcoded `st.vlm_max_tokens = 0`.
> `tests/test_ocr_max_tokens_surfaces.py` now fails on any of it — run it after
> adding an engine.

### 5e. Expose in CrispSorter (`CrispSorter/lib/engine/`)

CrispSorter (the Tauri desktop app) selects engines by the same int IDs.
When adding a new engine to CrispEmbed, also add it to:

1. `ocr_providers_init.dart` — register with `engine_id: N`
2. `ocr_model_manager.dart` — add `OcrModelVariant` entries
3. UI dropdown label (if not auto-generated from registry)

If you don't have access to the CrispSorter repo, leave a TODO in the
commit message so it gets picked up.

### Verification

After wiring, verify the full chain:
```python
from crispembed import CrispOcrOrchestrator
orch = CrispOcrOrchestrator(stages=[
    {"engine": N, "model_a": "yourmodel.gguf"}
])
text = orch.run("document.png")
```

If this fails silently (falls back to dbnet_trocr), `map_engine` is missing
the case for int N.

## Step 6: CMake, CLI, Model Registry, Bindings

### CMakeLists.txt
```cmake
list(APPEND CRISPEMBED_SOURCES src/yourmodel.cpp)
# ...
add_executable(test-yourmodel tests/test_yourmodel.cpp)
target_link_libraries(test-yourmodel PRIVATE crispembed)
```

### CLI (`examples/cli/main.cpp`)
- For OCR models: no new flags needed — `--ocr` auto-detects from GGUF metadata
- Update help text to list new architecture name

### Model Registry (`examples/cli/model_mgr.cpp`)
Add entry to `k_registry[]`:
```cpp
{"yourmodel",
 "yourmodel-q8_0.gguf",
 "https://huggingface.co/cstr/yourmodel-gguf/resolve/main/yourmodel-q8_0.gguf",
 "Description (architecture, params)", "SIZE MB", "license",
 "https://huggingface.co/cstr/yourmodel-gguf"},
```

**Then regenerate the SHA-256 pins — this is not optional:**

```bash
python tools/fetch_model_hashes.py        # rewrites examples/cli/model_hashes.h
```

An unpinned URL **refuses to download**. A GGUF is a graph this process then
executes, so "the download succeeded" is not an integrity statement; the
registry fails closed rather than installing an unverified payload. If you skip
this step your model simply will not fetch, and the error will tell you so.
`CRISPEMBED_ALLOW_UNPINNED_MODEL=1` overrides it for a one-off.

Three CI checks then hold the entry honest, all in `main-health.yml`:

| Check | Catches |
|---|---|
| `fetch_model_hashes.py --check` | pin missing, or the re-host re-uploaded different bytes |
| `fetch_model_hashes.py --check-sizes` | `"SIZE MB"` not matching the real file (4 entries were wrong when this landed) |
| `tests/check_registry_licenses.py` | `"license"` not matching the upstream card |

For the licence, check the **base model**, not just the fine-tune's tag — and
read `license_name` as well as `license`. Qwen2.5-VL-3B is `qwen-research`
(research-only) while the 7B and Qwen2-VL-2B are Apache-2.0, and HuggingFace
stores that in `license_name`, so a checker reading only `license` sees nothing.

### HTTP Server (`examples/server/server.cpp`)
If adding a new modality (not just a new OCR model variant), wire into the server:
1. Add `--yourflag MODEL` arg parsing and context init
2. Add `POST /your/endpoint` handler (parse JSON body, load image, call C API, return JSON)
3. Add to `/health` response and startup log
4. Add cleanup in shutdown block

For OCR models: already wired via `--ocr` → `POST /math/ocr`.

**⚠ Keep the HTTP surface in sync with the C ABI.** A capability can be resident in
the loaded model yet unreachable over HTTP — e.g. a reranker's classifier head loads
(`is_reranker=1`) but `/embed` on it returns backbone vectors, not scores (issue #37).
When you add a `crispembed_*` capability to `src/crispembed.h`, add the matching
route. Audit with:
```bash
# every capability entry point vs whether server.cpp reaches it
grep -oE 'crispembed_(encode_sparse|encode_audio|rerank|rerank_batch|encode_multivec)' src/crispembed.h | sort -u
grep -c crispembed_encode_sparse examples/server/server.cpp   # 0 == no route
```
Wire a retrieval route by mirroring the `/rerank` / `/sparse` / `/colbert/score`
handlers: **capability guard** (`is_reranker` / `has_sparse` / `has_colbert` → 400)
→ **escaping-aware parse** (`json_extract_strings` for text arrays,
`json_extract_number` for scalars like `top_n`) → `std::lock_guard(model_mutex)` →
call the batch C ABI (`crispembed_rerank_batch` caches classifier weights) → JSON out
via `json_escape`. Also add the capability flag to `/health` and the startup listing.

**Retrieval routes (current):**

| Route | Capability guard | C ABI | Request → Response |
|-------|------------------|-------|--------------------|
| `POST /embed`, `/v1/embeddings`, `/api/embed` | (dense — always) | `crispembed_encode[_batch]` | `{"texts":[...]}` → embeddings |
| `POST /rerank` | `is_reranker` | `crispembed_rerank_batch` | `{"query","documents","top_n"}` → `{"query","results":[{"index","score","document"}]}` |
| `POST /v1/rerank` | `is_reranker` | `crispembed_rerank_batch` | Cohere/Jina shape (#51): `{"model","query","documents":[str]\|[{"text"}],"top_n","return_documents"}` → `{"id","model","object","results":[{"index","relevance_score","document":{"text"}}],"meta"}` |
| `POST /sparse` | `has_sparse` | `crispembed_encode_sparse` | `{"texts":[...]}` → `{"results":[{"weights":{"<token_id>":w}}]}` (SPLADE/BGE-M3) |
| `POST /colbert/score` | `has_colbert` | `crispembed_encode_multivec` | `{"query","documents"}` → per-doc MaxSim scores |

`/health` reports `reranker` / `sparse` / `colbert` so a sidecar client can discover
the routes. These keys are **present-when-active**, not always-present booleans: a
capability that is off is omitted from the JSON entirely, never emitted as `false`.
Probe with `"reranker" in health` rather than `health["reranker"] == True` (#41).
Follow that convention when adding a capability — the whole block is a chain of
`if (cap) js << ", \"cap\": true"`. **Still unrouted (lower priority):** `crispembed_encode_audio`
(omnimodal audio embed) and a bi-encoder variant of `/rerank` — add them the same way
if a use case appears.

`/rerank` and `/v1/rerank` share one `handle_rerank_request()` rather than two
handlers, because the scoring path is identical and only the request/response
spelling differs; a second copy is the drift defect `core_json` was created to
undo. The two spelling differences on `/v1/rerank` are deliberate and gated:
`relevance_score` is `sigmoid(logit)` so the 0..1 range Cohere/Jina clients
threshold against holds (monotonic ⇒ same ranking; `CRISPEMBED_SERVER_RERANK_RAW_SCORES=1`
emits the raw logit instead), and `document` is echoed only under
`"return_documents": true`, matching Cohere's default. Request parsing goes
through `core_json::json_extract_documents`, which accepts both the `[str]` and
`[{"text":str}]` document forms — `json_extract_strings` flattens the object form
into twice as many garbage documents, so do not reach for it there.

### Python Bindings (`python/crispembed/_binding.py`)
Add a class following the `CrispVit` / `CrispMathOcr` pattern:
1. Add `_setup_yourmodel_signatures(lib)` function
2. Add `CrispYourModel` class with `__init__`, inference method, `__del__`
3. Export from `__init__.py`

For OCR models: already wired via `CrispMathOcr` (auto-dispatches from GGUF arch).

### Rust Bindings (`crispembed-sys/src/lib.rs` + `crispembed/src/lib.rs`)
1. Add FFI declarations in `crispembed-sys` (extern C block)
2. Add safe wrapper struct in `crispembed` with `new()`, inference method, `Drop`

### Dart / Flutter (`flutter/crispembed/lib/src/`)
1. Add Native + Dart typedefs in `crispembed_bindings.dart`
2. Add class in `crispembed.dart` with constructor, inference method, `dispose()`

### CrispCalc Dart Catalog (`lib/engine/ocr_model_manager.dart`)
Add `OcrModelVariant` entries with Q8_0, Q4_K, and F16 variants.

### CrispCalc Provider Init (`lib/engine/ocr_providers_init.dart`)
Register the new model at the appropriate priority tier.

## Step 7: Verify Parity + Quantize

### Test binary (`tests/test_yourmodel.cpp`)
```
./test-yourmodel model.gguf ref.gguf
```
Should report:
- `proj_output: cos >= 0.9999` (encoder parity)
- Same top token as reference
- Same decoded text

**Compare every stage, and make the result an exit code.** Two failures worth
knowing about, both found in shipped harnesses:

- A harness that *ran* the decoder, printed the output shape, freed the buffers
  and compared **nothing** — so "27 stages passing" was the vision tower alone
  and the 24-layer LLM had no coverage at all.
- The same harness returned `0` unconditionally, so every stage was advisory
  and a red one could not fail CI.

Count failures and `return 1`. Cover the whole stack: capping the dump at a few
decoder layers leaves the layers where a port actually diverges untested, and
you must compare the **logits** — a decoder can be wrong in a way every hidden
state hides, and generation reads logits, not hidden states.

**Check `argmax`, not only cosine.** Cosine stays high while the argmax moves,
and the argmax is what generation acts on. A model emitting fluent nonsense can
show cos 0.9999 at every stage.

### Quantization matrix
| Format | Size | Parity (cos) | Notes |
|--------|------|-------------|-------|
| F32    | ~4x  | baseline    | Development only |
| F16    | ~2x  | 0.9999+     | Full precision |
| Q8_0   | ~1.3x| 0.9999+     | Best quantized |
| Q4_K   | ~0.7x| 0.997+      | Desktop target |

### PyTorch ground-truth debugging
If parity is bad, use the reference GGUF to narrow down the bug:
1. Run reference dumper with synthetic test image
2. Run test binary with same model + reference
3. Compare layer-by-layer: the first layer where cosine drops below 0.999 is where the bug lives
4. **Never blame FP** — always find the real bug via layer-by-layer diff

---

## Adding Utility Libraries (preprocessors, renderers, detectors)

Utility libraries (not model backends) follow a lighter pattern.
**Wiring checklist** (same layers as model backends):

1. **C++ implementation** — self-contained in `src/`, own header
2. **C API wrapper** — add to `crispembed.h` + implement in `crispembed.cpp`
3. **CMakeLists.txt** — add source + test binary
4. **Rust FFI** — add to `crispembed-sys/src/lib.rs` (raw) + `crispembed/src/lib.rs` (safe)
5. **Python** — add to `python/crispembed/_binding.py` (ctypes signatures + class)
6. **Dart** — add to `flutter/crispembed/lib/src/crispembed.dart` (FFI)
7. **CLI** — add flags to `examples/cli/main.cpp`
8. **Server** — add endpoints to `examples/server/server.cpp` if appropriate
9. **Unit tests** — `tests/test_*.cpp` with synthetic images

> **ABI gotcha — a params struct is mirrored in FIVE places.** Adding a field to a
> C++ params struct (e.g. `scan_cleanup_params`) is not enough — the struct is
> passed by value across the C ABI, so every mirror must gain the SAME field in the
> SAME order or you get silent garbage / memory corruption:
> 1. `src/<lib>.h` (internal struct) and its `scan_cleanup_defaults()`.
> 2. `src/crispembed.h` — the C-API mirror struct (`crispembed_scan_cleanup_params`).
> 3. `src/crispembed.cpp` — the `to_*()` / `_defaults()` conversions must copy the
>    new field, and **must start from `<lib>_defaults()`** (not an uninitialised
>    struct) so any field the C API doesn't expose is still sane.
> 4. `crispembed-sys/src/lib.rs` (`#[repr(C)]` struct).
> 5. `python/crispembed/_binding.py` (`ctypes.Structure._fields_`).
> Dart uses `*_process_simple` (int args, no struct) so it is exempt from the struct
> mirror, but new *functions* still need Dart FFI typedefs + a method.

### Currently implemented utility libraries

| Library | C API | CLI | Rust | Python | Dart | Server | WASM |
|---------|-------|-----|------|--------|------|--------|------|
| Classical preproc (skew, bg norm, despeckle, blackfilter, page-split, content-bbox) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1-bit DWA morphology | header | — | — | — | — | — | — |
| CC text line detection | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Page dewarping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| OCR renderers (text, hOCR, ALTO, PDF) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OCR pipeline (det+rec+reading order) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OCR full pipeline (cleanup+routing) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Text detection (DBNet/Surya standalone) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Layout detection (RT-DETRv2 17-class) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Punctuation restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| TPS dewarp (learned) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| PDF DPI profiling | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| NAFNet denoising | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| TBSRN text-line SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| PAN whole-image SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| HAT SR (SOTA) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| DAT SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| SAFMN super-resolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Real-ESRGAN super-resolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| SwinIR super-resolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Restormer restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| SCUNet denoising | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| InstructIR restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| AdaIR restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Text LID (standalone) | ✓ | — | ✓ | ✓ | — | ✓ | — |
| Table structure recognition | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| BERT NER token classification | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| LiLT layout-aware KIE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Pix2Struct document understanding | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

### Implementation patterns

- **Self-contained C++** — no external deps, may use `morph_fast.h` for 1-bit ops
- **BSD-2-attributed** if cherry-picked from Leptonica
- **Unit tests** with synthetic images (gradients, speckles, curves)
- **Orchestrator integration** — new detector/cleanup methods surface as
  per-stage options via the `crispembed_ocr_stage` builder
- **Emitting an image? Use `core_imgout::emit()`** (`src/core/image_out.h`), never
  a hand-written `printf("P6\n...")`. It writes PNG with an `iTXt` provenance
  chunk naming your engine, and signs a C2PA manifest when an identity is
  configured. `POLICY.md` §5 claims every image CrispEmbed *returns to a caller*
  is marked; a new engine that prints its own header silently makes that claim
  false. Internal temporaries are the exception — use `core_tmp::make_private()`
  (`src/core/temp_file.h`) for those, never a hand-built /tmp name.
  Pass the engine name — it is what tells a reader whether detail was
  synthesised (ESRGAN, NAFNet) or merely resampled (deskew, dewarp). Returning
  the image in an HTTP body instead? `emit_to_string()` hands back the matching
  Content-Type with the bytes, so the two cannot disagree.
  See [provenance.md](provenance.md).

---

## Development Workflow

- **Always use `git worktree`** for feature branches — never checkout in-place
- **Keep debug prints** but gate behind `CRISPEMBED_DEBUG` env var
- **Debug dumps write to fixed `/tmp` paths on purpose.** `LAYOUT_DEBUG` makes
  `src/layout_detect.cpp` write `/tmp/cpp_*.bin` and read `/tmp/py_*.bin` — the
  names are a contract with the Python reference dumper, so do not randomise
  them. Do know what you are enabling: on a shared host those paths are
  predictable (a planted symlink redirects the write) and the files hold
  activations. For anything that is not a debug interchange, use
  `core_tmp::make_private()` (`src/core/temp_file.h`) — never a hand-built
  `/tmp` name.
- **Build target:** `crispembed` (static lib) + `crispembed-cli` + `crispembed-shared` + test binaries
- **Format: run `tools/format.sh --fix` before pushing.** This is enforced —
  `.github/workflows/lint.yml` pins clang-format 18.1.8 and fails the build on
  drift. (It runs on macOS because 18.1.8 wraps some files differently there
  than on Linux, so a Linux-formatted tree can still fail.)
- **Every new `tests/*.cpp` must route `main()` through `core_util::clean_exit`.**
  Copy this shape — `tools/check_test_clean_exit.sh` fails the `Static guards`
  CI job otherwise, and it has now blocked four separate pushes:

  ```cpp
  #include "core/clean_exit.h"

  static int crispembed_test_main() {
      // ... the test body that used to be in main() ...
      return failures == 0 ? 0 : 1;
  }

  int main() { core_util::clean_exit(crispembed_test_main()); }
  ```

  Why: ggml tears down its process-global GPU device in a static destructor at
  exit, which aborts on Metal and faults on CUDA — *after* the test has printed
  its result. A passing test then reports a corrupted exit code. It applies even
  to tests that touch no GPU today, because they link `crispembed-core` and are
  one dependency away from it. Long-lived hosts (the server, the bindings) are
  deliberately excluded: they free their contexts on shutdown instead.

- **Model-free tests run on every push** and need no weights or network:
  ```bash
  cmake --build build --target test-image-provenance test-provenance-marking test-msac-tiling \
    test-qwen-pretokenize test-bpe-pretokenize
  ./build/test-image-provenance && ./build/test-provenance-marking && ./build/test-msac-tiling
  ./build/test-qwen-pretokenize && ./build/test-bpe-pretokenize
  ```
  `test-qwen-pretokenize` / `test-bpe-pretokenize` guard the byte-level BPE
  pre-tokenizers in `src/core/bpe.h` against HuggingFace's own
  `pre_tokenize_str()` output. Their case tables are GENERATED — if you change a
  pre-tokenizer, rerun `python tools/gen_bpe_pretokenize_test.py
  tests/test_bpe_pretokenize.cpp && tools/format.sh --fix` (needs network and the
  `tokenizers` package) rather than hand-editing the goldens.
  Anything you add that can be checked without a checkpoint belongs here rather
  than in the artifact-gated tiers — a test that only runs on an equipped runner
  gates nothing on most pushes.
- **Coordinate through `PLAN.md`.** Several sessions run in parallel against
  this repo. Add a row to the "Active work in flight" table before starting,
  update it at each checkpoint, and push that file to `main` so others can see
  what is claimed.

---

## Release engineering: never ship a `-march=native` build

Any workflow leg that produces an artifact a *user* downloads (release archive,
Python wheel, Rust prebuilt) must configure with `-DGGML_NATIVE=OFF` and then
run the gate:

```bash
cmake -S . -B build -DGGML_NATIVE=OFF <backend flags>
python scripts/check-cpu-baseline.py build     # fails the job if not portable
cmake --build build --config Release -j4
```

`GGML_NATIVE` defaults to ON and **executes probe programs on the build
machine** — `check_c_source_runs` with an AVX-512 binary on MSVC, `-march=native`
on GCC/Clang, `-mcpu=native` plus `dotprod`/`i8mm`/`sve`/`sme` run-probes on ARM.
The ISA of the artifact therefore depends on which CI runner took the job, and
GitHub's pools are heterogeneous. That shipped as **#41**: v0.16.1's Windows cpu
zip was compiled `/arch:AVX512` and died with `Illegal instruction` on any
consumer Intel CPU.

Two things make it hard to catch by hand, which is why the gate exists:

1. **CI can never reproduce it.** No runner lacks the extension the runner had.
2. **`CMakeCache.txt` lies when NATIVE is ON** — `FindSIMD.cmake` sets
   `GGML_AVX512` as a *normal* variable that shadows the cache entry, so the
   cache can read `OFF` while the compile line says `/arch:AVX512`. The gate
   also scans `build.ninja` / `*.vcxproj` / `flags.make` for that reason.

`CRISPEMBED_NATIVE` (which drives `-march=native` on CrispEmbed's own targets for
the `cpu_ops.h` intrinsics) defaults to `GGML_NATIVE`, so the one flag covers the
whole tree; with it off, those targets mirror ggml's configured baseline rather
than falling back to scalar. Shipped baselines are listed in the README
("CPU requirements & redistributable builds"); the full write-up is in
`LEARNINGS.md` §"GGML_NATIVE probes the BUILD machine".

---

## WASM Build

CrispEmbed compiles to WebAssembly via Emscripten for client-side browser use.

### Build scripts

| Script | Target | Output |
|--------|--------|--------|
| `build-wasm.sh` | OCR (full pipeline) | `build-wasm/crispembed_ocr.{js,wasm}` |
| `build-embed-wasm.sh` | Text embeddings | `build-embed-wasm/crispembed_embed.{js,wasm}` |

### Architecture

```
browser
  ├─ crispembed_ocr.js       ← Emscripten-generated loader (modularized)
  ├─ crispembed_ocr.wasm     ← compiled WASM binary (~2.3 MB)
  ├─ crispembed-ocr.js       ← high-level JS wrapper (issue #31)
  └─ model.gguf              ← fetched at runtime, loaded into MEMFS
```

The Emscripten module exports C functions prefixed with `wasm_*`. The high-level
JS wrapper (`crispembed-ocr.js`) provides:
- **TextDecoder fix** for resizable ArrayBuffer (V8 bug with ALLOW_MEMORY_GROWTH)
- **Canvas API abstraction** — accepts HTMLImageElement, Canvas, Video, Blob, File, URL
- **One-shot API** — `create()` → `recognize()` → `dispose()`, no manual malloc/free
- **JSON serialization** — pipeline results returned as parsed JSON objects

### WASM module components

The OCR WASM module includes the complete pipeline:

| Component | C wrapper function | Model needed? |
|-----------|--------------------|---------------|
| Single-model recognition | `wasm_ocr_init/recognize/free` | Yes (TrOCR/pix2tex GGUF) |
| Full pipeline (det+rec) | `wasm_ocr_pipeline_init/run/free` | Yes (DBNet + TrOCR) |
| Advanced pipeline | `wasm_ocr_pipeline_full_init/run/free` | Yes + optional NAFNet/SR |
| Scan cleanup (classical) | `wasm_scan_cleanup_init/process/free` | No |
| Text detection | `wasm_text_det_init/run/free` | Yes (DBNet/Surya GGUF) |
| Layout detection | `wasm_layout_init/detect/free` | Yes (RT-DETRv2 GGUF) |
| OCR rendering | `wasm_ocr_render` | No (uses pipeline results) |

### Adding a function to the WASM build

1. Add `WASM_EXPORT` function to `wasm/ocr_wrapper.c`
2. Add `'_function_name'` to `EXPORTED_FUNCS` in `build-wasm.sh`
3. Add JS wrapper method in `wasm/crispembed-ocr.js`
4. Add Dart wrapper in `flutter/crispembed/lib/src/math_ocr_web.dart`
5. Add Rust wrapper in `wasm/crispembed-ocr-wasm/src/lib.rs`
6. Run tests: `node tests/test_wasm_ocr_wrapper.js && node tests/test_wasm_ocr_live.js`

### WASM build flags

```bash
ALLOW_MEMORY_GROWTH=1     # WASM memory grows as needed
INITIAL_MEMORY=134217728  # 128 MB initial (full pipeline)
STACK_SIZE=2097152        # 2 MB stack
MODULARIZE=1              # factory function pattern
EXPORT_NAME=CrispEmbedOCR # global factory name
ENVIRONMENT=web,worker    # browser targets
FILESYSTEM=1              # MEMFS for model loading
```

### Testing

```bash
# Unit tests (no WASM build needed — validates JS, exports, consistency):
node tests/test_wasm_ocr_wrapper.js

# Live tests (requires build-wasm.sh first — loads actual WASM module):
node tests/test_wasm_ocr_live.js
```

### CI/CD

- `build-wasm.yml` — builds on push, uploads artifacts (30-day retention)
- `release-wasm.yml` — attaches WASM bundles to GitHub releases
