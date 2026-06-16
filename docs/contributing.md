# Contributing: Adding a New Model to CrispEmbed

This guide covers adding a new model backend (OCR, embedding, face, etc.) to CrispEmbed. Follow the same six-step pattern used for every model in the codebase.

## Checklist Summary

1. [ ] Write the C inference engine (`src/yourmodel.{h,cpp}`)
2. [ ] Write the GGUF converter (`models/convert-yourmodel-to-gguf.py`)
3. [ ] Write the reference dumper (`tools/dump_yourmodel_reference.py`)
4. [ ] Wire into the C ABI (`src/crispembed.cpp`)
5. [ ] **Wire into orchestrator** (OCR only): enum + `map_engine` + header comment + `ocr_orchestrator.cpp` dispatch + CrispSorter
6. [ ] Wire into CMake + CLI + model registry + all bindings (Python, Rust, Dart)
7. [ ] Verify parity, quantize, update CrispCalc catalog

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

### HTTP Server (`examples/server/server.cpp`)
If adding a new modality (not just a new OCR model variant), wire into the server:
1. Add `--yourflag MODEL` arg parsing and context init
2. Add `POST /your/endpoint` handler (parse JSON body, load image, call C API, return JSON)
3. Add to `/health` response and startup log
4. Add cleanup in shutdown block

For OCR models: already wired via `--ocr` → `POST /math/ocr`.

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

### Currently implemented utility libraries

| Library | C API | CLI | Rust | Python | Dart | Server |
|---------|-------|-----|------|--------|------|--------|
| Classical preproc (skew, bg norm, despeckle) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1-bit DWA morphology | header | — | — | — | — | — |
| CC text line detection | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Page dewarping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OCR renderers (text, hOCR, ALTO, PDF) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Punctuation restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TPS dewarp (learned) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PDF DPI profiling | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NAFNet denoising | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TBSRN text-line SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PAN whole-image SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HAT SR (SOTA) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DAT SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SAFMN super-resolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Real-ESRGAN super-resolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SwinIR super-resolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Restormer restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SCUNet denoising | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| InstructIR restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AdaIR restoration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Text LID (standalone) | ✓ | — | ✓ | ✓ | — | ✓ |
| Table structure recognition | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BERT NER token classification | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LiLT layout-aware KIE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pix2Struct document understanding | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Implementation patterns

- **Self-contained C++** — no external deps, may use `morph_fast.h` for 1-bit ops
- **BSD-2-attributed** if cherry-picked from Leptonica
- **Unit tests** with synthetic images (gradients, speckles, curves)
- **Orchestrator integration** — new detector/cleanup methods surface as
  per-stage options via the `crispembed_ocr_stage` builder

---

## Development Workflow

- **Always use `git worktree`** for feature branches — never checkout in-place
- **Keep debug prints** but gate behind `CRISPEMBED_DEBUG` env var
- **Build target:** `crispembed` (static lib) + `crispembed-cli` + `crispembed-shared` + test binaries
- **Format:** No mandatory formatter, but keep consistent with surrounding code
