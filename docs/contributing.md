# Contributing a New Model to CrispEmbed

This guide documents the end-to-end methodology for adding a new model
architecture to CrispEmbed. Every model — whether an embedding encoder,
a vision backbone, or an OCR decoder — follows the same 7-step pipeline.

## Overview

```
1. Research       → understand the architecture, find weights
2. Converter      → Python script: HF/ONNX checkpoint → GGUF
3. Reference dump → Python script: capture per-layer activations
4. C++ inference  → build ggml graph, load GGUF, forward pass
5. Diff harness   → compare C++ output vs Python reference, layer by layer
6. Quantize       → produce F16/Q8_0/Q4_K variants, test each
7. Ship           → C API, CLI flag, HF upload, docs
```

## Step 1: Research the Architecture

Before writing code, understand:

- **Exact layer structure**: How many layers, what kind (attention, conv,
  RNN), what normalization (pre-LN vs post-LN, RMSNorm vs LayerNorm).
- **Weight naming**: What keys does the checkpoint use? Print the state dict.
- **Preprocessing**: Input normalization (ImageNet mean/std? CLIP mean/std?
  Grayscale?), resize strategy, padding.
- **Post-processing**: Pooling, detokenization, NMS, thresholding.
- **Existing code to reuse**: Check both CrispEmbed and CrispASR `src/core/`.
  Never rebuild what already exists (tokenizers, attention helpers, beam
  search, GGUF loader, cross-attention). See PLAN.md "CrispASR reuse map".

Read the original paper. Read the reference Python code. Run the HuggingFace
model on a test input and capture the output. This is your ground truth.

## Step 2: Write the Converter

**File**: `models/convert-<name>-to-gguf.py`

The converter reads a checkpoint (PyTorch `.pth`/`.safetensors`, ONNX `.onnx`,
or HuggingFace model dir) and writes a single `.gguf` file containing:

1. **Metadata** (key-value pairs): architecture name, hyperparameters,
   preprocessing constants, tokenizer config.
2. **Tensors** (named weight arrays): all model weights in a flat namespace.
3. **Tokenizer** (if applicable): token strings embedded in GGUF metadata.

### Key conventions

- **Tensor naming**: Use dotted paths that match what the C++ code expects.
  Example: `enc.encoder.layer.0.attention.query.weight`. Follow existing
  converters — the C++ load function does `get("enc.encoder.layer.0....")`.

- **BatchNorm folding**: For CNN models, ALWAYS fold BN into the preceding
  Conv2d at convert time. Store fused `weight` and `bias` only. No runtime
  BN. Use `fold_bn_into_conv()` (see `convert-hmer-to-gguf.py` or
  `convert-dbnet-to-gguf.py` for the helper).

- **Conv weight flattening**: Flatten 4D conv weights `(OC, IC, KH, KW)` to
  2D `(OC, IC*KH*KW)` for GGUF quantization compatibility (quantizer needs
  ncols divisible by 32). The C++ code reshapes back to 4D before `ggml_conv_2d`.

- **ConvTranspose2d**: Weight layout is `(IC, OC, KH, KW)` — transposed
  relative to Conv2d. Flatten to `(IC, OC*KH*KW)`. The C++ side uses a
  separate `prep_deconv_weight()` that reshapes to ggml's `[KW, KH, OC, IC]`.

- **Tokenizer embedding**: Use HuggingFace `AutoTokenizer` when possible
  (handles vocab offsets like XLM-R's fairseq shift). Use
  `convert_ids_to_tokens()` to preserve space markers like `▁`. Fall back to
  raw SentencePiece only if `transformers` is unavailable.

- **Struct name uniqueness**: If your C++ code defines structs like
  `enc_layer` or `dec_layer`, prefix them with the model name
  (e.g., `math_ocr_dec_layer`). ODR violations cause silent heap corruption
  when linked into the full CLI binary.

### Testing the converter

```bash
python models/convert-<name>-to-gguf.py --checkpoint model.pth --output model-f32.gguf
# Verify: print tensor names + shapes
python -c "
import gguf
r = gguf.GGUFReader('model-f32.gguf')
for t in r.tensors: print(f'{t.name}: {list(t.shape)}')
"
```

## Step 3: Write the Reference Dumper

**File**: `tools/dump_<name>_reference.py`

This script runs the model in Python (PyTorch or pure-numpy) on a test
input and captures intermediate activations at every architectural boundary.
Writes them to a GGUF "reference archive" that the C++ diff harness loads.

### What to capture

Capture activations **after** each major stage:

- **Input**: preprocessed image/audio/text (so C++ preprocessing can be
  verified independently).
- **Per-layer outputs**: after each encoder layer, each decoder step.
- **Sub-layer outputs**: attention output (pre-residual), FFN output
  (pre-residual) — useful for pinpointing which op diverges.
- **Final output**: embedding vector, probability map, logits, etc.

### Writing the reference

Use the same GGUF writer pattern as `tools/dump_reference.py`:

```python
import gguf
writer = gguf.GGUFWriter(output_path, "model-ref")
writer.add_tensor("backbone_stage_0", activation.astype(np.float32),
                   raw_dtype=gguf.GGMLQuantizationType.F32)
# ... all stages ...
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

### Running it

```bash
python tools/dump_<name>_reference.py \
    --checkpoint model.pth --image test.png --output ref.gguf
```

## Step 4: Write the C++ Inference

**Files**: `src/<name>.{h,cpp}`

The C++ inference builds a ggml computation graph, loads weights from the
GGUF, runs the forward pass, and returns results.

### Pattern

```cpp
// 1. Load: open GGUF metadata, read hparams, load weights into backend
bool load(context** ctx, const char* path, int n_threads);

// 2. Forward: build ggml graph, set inputs, compute, read outputs
std::vector<float> forward(context* ctx, const float* input, int H, int W);

// 3. Post-process: threshold, NMS, detokenize, etc.
std::vector<result> detect(context* ctx, ...);
```

### Key patterns from existing code

- **Conv2d**: `prep_conv_weight()` handles 2D-flattened quantized weights:
  dequant → reshape 4D → cast to F16 (required by `ggml_conv_2d`).
- **ConvTranspose2d**: `prep_deconv_weight()` — same but swaps IC/OC dims.
- **Bilinear upsample**: `ggml_interpolate(g, x, w, h, c, 1, GGML_SCALE_MODE_BILINEAR)`.
  Do NOT use `ggml_upscale_ext` (deprecated).
- **Sigmoid**: `ggml_sigmoid(g, x)`.
- **stb_image**: Do NOT include `STB_IMAGE_IMPLEMENTATION` — use extern
  declarations and rely on `image_preprocess.cpp`'s copy.

### Adding to the build

In `CMakeLists.txt`:
```cmake
list(APPEND CRISPEMBED_SOURCES src/<name>.cpp)
```

## Step 5: Run the Diff Harness

**File**: `src/crispembed_diff.h` (header-only, shared across all models)

The diff harness loads the Python reference archive and compares tensors:

```cpp
crispembed_diff::Ref ref;
ref.load("ref.gguf");

auto r = ref.compare("prob_map_sigmoid", cpp_output, n_elements);
printf("cos_min=%.6f max_abs=%.2e %s\n",
       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
```

### Quality gates

| Metric | F32 gate | Q8_0 gate | Q4_K gate |
|--------|----------|-----------|-----------|
| cos_min | >= 0.999 | >= 0.995 | >= 0.95 |
| max_abs | < 0.01 | < 0.1 | < 1.0 |

### Debugging divergence

When a layer fails:

1. Capture the **specific layer** output from both Python and C++.
2. The first layer where `cos_min` drops below 0.999 is where the bug lives.
3. Common causes:
   - Pre-LN vs post-LN confusion
   - Wrong activation function (GELU tanh vs erf)
   - Missing position offset (RoBERTa +2)
   - Weight transpose (row-major vs col-major)
   - RoPE applied in wrong dimension order
   - BN not folded / folded incorrectly
   - ConvTranspose2d weight layout (IC/OC swapped)

## Step 6: Quantize and Validate

```bash
# Produce all variants
crispembed-quantize model-f32.gguf model-f16.gguf f16
crispembed-quantize model-f32.gguf model-q8_0.gguf q8_0
crispembed-quantize model-f32.gguf model-q4_k.gguf q4_k
```

### What to watch for

- **Small tensors** (ncols < 32): the quantizer skips these automatically.
  Check the quantizer output for "skip" messages.
- **Narrow models** (d_model <= 256): Q4_K may degrade significantly. Test
  live accuracy, not just cosine parity. TrOCR-small (256d) fails at Q4_K.
- **Embedding tables**: large vocab tables (50K+ tokens) are quantizable
  but may lose precision for rare tokens. The quantizer keeps them at Q8_0
  by default for embedding models.
- **Conv weights**: 4D conv weights are flattened to 2D by the converter.
  The quantizer handles 2D fine. But check that ncols is div by 32 after
  flattening — if not (e.g., 7×7×3=147), the quantizer copies them as F32.

### Live testing

Always test with **actual inputs**, not just cosine parity:

```bash
# Detection: count regions, compare scores
for q in f32 f16 q8_0 q4_k; do
    echo "$q: $(./test --model model-$q.gguf --image test.png | head -1)"
done

# Recognition: compare output text
for q in f32 f16 q8_0 q4_k; do
    echo "$q: $(./test --model model-$q.gguf --image crop.png | grep Recognized)"
done
```

If Q4_K produces wrong output, **do not ship it**. Document the limitation
in the HF README and skip that quant level.

## Step 7: Ship

### C API

Add to `src/crispembed.h`:
```c
CRISPEMBED_API void * crispembed_<name>_init(const char *path, int n_threads);
CRISPEMBED_API void crispembed_<name>_free(void *ctx);
CRISPEMBED_API const <result_type> * crispembed_<name>_run(void *ctx, ...);
```

Implement the wrapper in `src/crispembed.cpp`. Use an opaque `void*` context
with an internal wrapper struct.

### CLI

Add `--<name>` flag to `examples/cli/main.cpp`. Follow the pattern of
`--ocr`, `--hmer`, `--face-pipeline`. Handle early exit before the
generic model resolution code if needed.

### HuggingFace upload

1. Write `models/hf_readme_<name>.md` with YAML frontmatter (license, tags,
   base_model, pipeline_tag).
2. Create repo: `huggingface_hub.create_repo("cstr/<name>-GGUF")`
3. Upload README + all passing GGUF variants.
4. Verify file listing matches expectations.

### Documentation

Update these files:
- `README.md` — add model to feature list + quick-start example
- `HISTORY.md` — work log entry with architecture details + test results
- `LEARNINGS.md` — any non-obvious technical lessons (ODR, weight layouts, etc.)
- `PLAN.md` — mark items as done, update architecture diagram

### Git workflow

Always work in a git worktree:
```bash
git worktree add /mnt/volume1/CrispEmbed-<feature> -b feat/<feature> main
cd /mnt/volume1/CrispEmbed-<feature>
git submodule update --init --recursive
```

Commit early, commit often. Push the feature branch. Merge to main with
`--no-ff` for a merge commit. Never force-push main.
