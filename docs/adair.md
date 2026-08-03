# AdaIR — All-in-One Image Restoration

AdaIR (Adaptive Image Restoration) is a unified image restoration model from
ICLR 2025.  It handles five degradation types with a single set of weights:

1. **Denoise** — Gaussian noise removal
2. **Derain** — rain streak removal
3. **Dehaze** — haze / fog removal
4. **Deblur** — motion / defocus blur removal
5. **Low-light** — low-light enhancement

## Architecture

- **Backbone**: Restormer (multi-Dconv head transposed attention)
- **AFLB**: Adaptive Frequency Learning Blocks — FFT-based spectral
  decomposition separates low/high-frequency components for task-adaptive
  processing
- **Cross-attention guidance**: learned degradation prompts guide the
  restoration through cross-attention layers
- **Parameters**: 28.8M
- **License**: MIT
- **Source**: [c-yn/AdaIR](https://github.com/c-yn/AdaIR)

## Parity

Cosine similarity between reference PyTorch output and CrispEmbed GGUF
engine output: **cos = 0.999924**.

Verified against a *genuine* reference — the real PyTorch AdaIR model (upstream
[c-yn/AdaIR](https://github.com/c-yn/AdaIR)) run on weights reconstructed from
`adair-5d-f32.gguf` (all 587 params load). The `test-adair-diff` harness reports
**cos = 0.999379** on a 64×64 seeded-random input (a harsher test than a natural
image). Reproduce with `tools/dump_adair_reference_from_gguf.py`; ref is on HF at
`cstr/text-super-resolution-gguf/adair-ref.gguf`.

### F16 (fixed 2026-08-02)

F16 was previously unusable — it aborted in `adair_kernel` with
`tensor buffer not set`, and once that was guarded it still produced
cos 0.729509. Neither was a backend or a quantization problem. `tools/quantize.cpp`
flattens every 4-D conv weight to 2-D `[IC*KH*KW, OC]` in the output header, and
this engine inferred three hidden widths from `->ne[3]`, which is `1` on a
flattened tensor: the GDFN hidden width, the FreModule `rate_conv` width and the
`ChannelGate` MLP width. A GDFN hidden width of 1 makes `half = hidden/2` zero,
so the following 1×1 conv was built with `ic == 0` — the zero-element kernel
descriptor the backend could not allocate.

`conv1x1_out_channels()` now derives OC from the element count, which is correct
under both layouts. F16 measures **cos 0.999383, max_abs 0.027871** on the 64×64
`adair-ref.gguf` fixture, against **0.999382 / 0.027892** for f32 through the
same binary — i.e. F16 costs nothing measurable here. Set
`ADAIR_LEGACY_NE3_DIMS=1` to restore the old `ne[3]` read for bisection.

Confirmed through the real CLI as well: restoring a 96×96 crop with the f16 and
the f32 model returns rc=0 for both and the two output images agree at
cosine `1.0`, max_abs `1/255`.

The registry still ships f32: repointing it needs the f16 uploaded to
`cstr/adair-GGUF` and a SHA-256 pin in `examples/cli/model_hashes.h`.

## Performance

All convolution sites — the U-Net down/up/reduce/output convs plus the
1×1/3×3/depthwise convs inside MDTA attention, the GDFN feed-forward, the
cross-attentions, and the FreModule — run via `ggml_conv_2d` /
`ggml_conv_2d_dw` on a dedicated CPU-backend scheduler. Persistent F32 kernels
are cached and keyed by the dequantized weight pointer, and F16-cast in-graph
(conv expands to im2col(F16)+mul_mat, which the CPU scheduler can't place
against an F32 kernel). This is **~5.2× faster per tile** (≈15.4 s → ≈3.0 s on a
64×64 tile, Apple M1) at cos 0.999385 vs 0.999379 scalar — no accuracy
regression. The 2D FFT in the AFLB (`fft1d`/`fft2d`) and the attention softmax
stay SIMD-scalar. Default ON; set `ADAIR_SCALAR=1` to force the scalar path.

## Usage

### CLI

> **Output format:** PNG with an `iTXt` provenance chunk by default (`CRISPEMBED_IMAGE_FORMAT=ppm` for raw Netpbm). See [provenance.md](provenance.md).

```bash
crispembed --adair-model adair-5d-f16.gguf --adair noisy.png > restored.png
```

### Server

```bash
crispembed-server --adair-model adair-5d-f16.gguf

curl -X POST http://localhost:8080/adair/restore \
     -d '{"image": "noisy.png"}'
```

### Python

```python
from crispembed import CrispAdaIR

ir = CrispAdaIR("adair-5d-f16.gguf")
out = ir.process(pixels, width, height)  # returns ndarray (H, W, 3)
```

### Rust

```rust
use crispembed::CrispAdaIR;

let mut ir = CrispAdaIR::new("adair-5d-f16.gguf", 0).unwrap();
let (pixels, w, h) = ir.process(&input_rgb, width, height).unwrap();
```

### Flutter / Dart

```dart
final ir = CrispAdaIR('adair-5d-f16.gguf');
final out = ir.process(pixels, width, height);
ir.dispose();
```

## Model registry

The model is available via the built-in model manager:

```bash
crispembed --list-models | grep adair
# adair-5d  adair-5d-f16.gguf  57 MB  MIT
```
