# GOT-OCR2

GGUF port of [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0):
SAM ViT-B vision encoder + Qwen2-0.5B decoder, ~0.7B params, for document /
LaTeX / table OCR.

- Source: `src/got_ocr.cpp`
- Models: [`cstr/got-ocr2-crispembed-GGUF`](https://huggingface.co/cstr/got-ocr2-crispembed-GGUF)
- Usage: `crispembed --ocr got-ocr2 image.png`

## Architecture

- **Vision**: SAM ViT-B (12 layers, 768d, 12 heads, 16×16 patches, 1024×1024 input)
  - Windowed attention (ws=14), global attention at layers [2, 5, 8, 11]
  - Decomposed relative position encoding
  - Neck: Conv(768→256) → LN2d → Conv(256→256) → LN2d
  - Downsample: Conv(256→512→1024, stride 2) → 256 vision tokens
  - Projector: Linear(1024, 1024)
- **LLM**: Qwen2-0.5B (24 layers, 1024d, MHA 16/16, SiLU SwiGLU, RoPE θ=1M)
- **Tokenizer**: tiktoken (151860 vocab)
- **Prompt** (ChatML): `<|im_start|>system\n…<|im_end|><|im_start|>user\n<img>{256×<imgpad>}</img>\nOCR: <|im_end|><|im_start|>assistant\n`
  (im_start=151644, im_end=151645, img=151857, /img=151858, imgpad=151859)

## Quantization — ship Q4_K (default)

| Build | Precision | Size | Notes |
|-------|-----------|------|-------|
| `got-ocr2-q4_k.gguf` | Q4_K | 445 MB | **Default.** Correct OCR, fastest decode on M1 |
| `got-ocr2-q8_0.gguf` | Q8_0 | 599 MB | Correct; slower per-token than Q4_K on M1 (see below) |
| `got-ocr2-f16.gguf`  | F16 | 1.44 GB | Full-precision baseline |

Produce them with:

```bash
crispembed-quantize got-ocr2-f16.gguf got-ocr2-q4_k.gguf q4_k
crispembed-quantize got-ocr2-f16.gguf got-ocr2-q8_0.gguf q8_0
```

The quantizer's `--decoder-f16` flag (keep `l.*` decoder weights at F16) is
**optional and not required for correctness** — it exists only for diagnostic /
A-B comparison. See the history note below for why it was once thought mandatory.

## Precision & parity

Verified against the real HF model (transformers `GotOcr2`) + a Python f32
reference (`tools/dump_got_ocr_reference.py`, harness `tests/test_got_ocr_diff.cpp`):

- **Vision** (ViT layers, neck, downsample, projector): cos ≥ 0.998 vs HF.
- **LLM decoder**, per-layer, **plain Q8_0 weights vs f32 reference**:

  ```
  llm_layer_0: cos_min=0.999960   PASS
  llm_layer_1: cos_min=0.999962   PASS
  llm_layer_2: cos_min=0.999971   PASS
  llm_layer_3: cos_min=0.999972   PASS
  llm_layer_4: cos_min=0.999985   PASS
  llm_layer_5: cos_min=0.999994   PASS
  ```

- **End-to-end OCR**: Q4_K, Q8_0 and F16 all produce byte-identical output on the
  test page ("The quick brown fox jumps over the lazy dog. 12345").

## Per-token decode speed (Apple M1)

Measured with `CRISPEMBED_GOT_OCR_BENCH=1 GOT_OCR_STEP_PROFILE=1`, KV cache
active (T=1 incremental, 256 vision tokens in the prefix):

| Build | Decode / token |
|-------|----------------|
| **Q4_K** | **~20 ms** |
| F16  | ~38 ms |
| Q8_0 | ~42 ms |

Per-step breakdown (`GOT_OCR_STEP_PROFILE`) shows the cost is essentially all in
`ggml_backend_sched_graph_compute` (build ≈ 0.3 ms, alloc ≈ 2 ms, readback ≈
0.1 ms). So decode is compute/bandwidth-bound, not graph-rebuild-bound — every
Qwen decoder in this repo (`internvl2_ocr`, `qwen2vl_ocr`, `deepseek_ocr2`) uses
the same per-token `build → sched_reset → alloc → compute → free` pattern; there
is no persistent-graph optimization being missed.

**Q4_K is both correct and the fastest option on M1**, and 3× smaller than F16,
so it is the default. Q8_0's per-token slowness is a Metal `mul_mv` kernel issue,
documented separately in [`metal-q8_0-mul_mv-slow-m1.md`](metal-q8_0-mul_mv-slow-m1.md).

## Diagnostic env vars

| Var | Effect |
|-----|--------|
| `CRISPEMBED_GOT_OCR_BENCH=1` | print vision / neck / prefill / per-decode-step ms |
| `GOT_OCR_STEP_PROFILE=1` | per-decode-step breakdown (build/alloc/setinput/compute/readback) |
| `GOT_OCR_NO_KV_CACHE=1` | decode with full O(n²) recompute each step (parity check) |
| `GOT_OCR_FORCE_CPU=1` | force CPU backend (A/B vs Metal) |
| `CRISPEMBED_GOT_OCR_DEBUG=1` | verbose (verbosity=2) |

## History: the "decoder must be F16" false alarm (#25)

An earlier revision shipped an F16-decoder build and documented the Qwen2-0.5B
decoder as "catastrophically quant-sensitive — `llm_layer_0` cos ≈ 0.936 at Q8_0,
compounding to repeated-`color` garbage." **That was wrong.** The 0.936 came from
a per-row bug in `tests/test_got_ocr_diff.cpp`: `compare()` used the token count
(5) as the row length instead of the feature dimension (1024), so the cosine was
computed over the wrong stride. The same harness bug had previously produced a
bogus "bf16 compute" theory.

With the corrected harness (`row_dim=0`) the Q8_0/Q4_K decoder matches f32 at
cos ≥ 0.99996 and OCR is identical to F16. Corroborating evidence: got_ocr's
decoder graph is functionally identical to `internvl2_ocr`'s Qwen2-0.5B path
(same NEOX RoPE θ=1M, rmsnorm, flash_attn scale, KV-cache layout, SwiGLU), and
`internvl2-1b` already ships that same decoder at Q4_K.

**Lesson**: a parity harness is only as trustworthy as its reduction axis. A
cosine computed over the wrong dimension can look like catastrophic model
sensitivity and drive a real (wasteful) precision workaround. Always sanity-check
a "this small model can't be quantized" claim against a sibling model that ships
the same architecture at that quant.
