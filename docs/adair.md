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
`cstr/text-super-resolution-gguf/adair-ref.gguf`. The convolution path is still
SIMD-scalar (conv→ggml port pending; the 2D FFT in the AFLB stays scalar).

## Usage

### CLI

```bash
crispembed --adair-model adair-5d-f16.gguf --adair noisy.png > restored.ppm
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
