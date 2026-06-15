# TPS Spatial Transformer (Learned Dewarping)

Thin-Plate Spline dewarping for camera-captured and book-scanned documents. Handles irregular distortion (perspective, finger occlusion, book spine warping) better than the classical polynomial dewarp in `dewarp.cpp`.

## Architecture

Two-layer design:

1. **TPS math** (`src/tps_warp.{h,cpp}`) -- model-free linear algebra:
   - `tps_solve()` -- solve the (N+3)x(N+3) TPS interpolation system
   - `tps_warp()` -- inverse warp with bilinear sampling
   - `tps_map_point()` -- map a single point through the transform
   - Radial basis: U(r) = r^2 * ln(r), Gaussian elimination with partial pivoting

2. **Localization network** (`src/tps_locnet.cpp`) -- learned control point prediction:
   - 4x Conv3x3+BN(folded)+ReLU + pool (MaxPool2x2 / AdaptiveAvgPool)
   - FC1: 128->64+ReLU, FC2: 64->N*2
   - Channels: 3->16->32->64->128 ("small" variant, ~108K params)
   - CPU-scalar forward pass, no ggml graph

## Model

- **Source**: PaddleOCR RARE recognition model (`rec_mv3_tps_bilstm_att_v2.0`)
- **License**: Apache-2.0
- **GGUF**: `/mnt/storage/gguf-models/tps-loc-f32.gguf` (424 KB)
- **Converter**: `models/convert-tps-loc-to-gguf.py` (pickle-based, no PaddlePaddle needed)
- **Fiducial points**: 20 (10 top + 10 bottom edge)
- **FC dim**: 64

## Parity

Verified with `tools/dump_tps_reference.py` + `tests/test_tps_parity.cpp`:

| Stage | cos_min | max_abs | Status |
|-------|---------|---------|--------|
| fc2_out (raw coords) | 1.000000 | 0.000000 | PASS |
| points_pixel | 1.000000 | 0.000000 | PASS |

Exact bit-for-bit match between Python reference and C++ (F32 vs F32).

## API

### C ABI (`crispembed.h`)

```c
// Model-free: manual control points
int crispembed_tps_dewarp(
    const uint8_t * gray, int w, int h,
    const float * src_x, const float * src_y,
    const float * dst_x, const float * dst_y, int n,
    uint8_t * out);

// Model-based: CNN predicts control points
int crispembed_tps_auto_dewarp(
    const uint8_t * gray, int w, int h,
    const char * model_path, uint8_t * out);
```

### CLI

```bash
# Learned TPS dewarp (needs model GGUF)
crispembed --tps-dewarp tps-loc-f32.gguf curved_page.png > straight.pgm
```

### HTTP Server

```bash
curl -X POST http://localhost:8080/preprocess/tps-dewarp \
  -H 'Content-Type: application/json' \
  -d '{"image": "curved_page.png", "model": "tps-loc-f32.gguf"}'
```

### Python

```python
from crispembed import CrispPreprocess
pp = CrispPreprocess()
result = pp.tps_dewarp(gray_array, w, h, "tps-loc-f32.gguf")
```

## Integration Matrix

| Layer | tps_dewarp (manual) | tps_auto_dewarp (model) |
|-------|-------|-------|
| C API (crispembed.h) | Y | Y |
| CLI | - | Y (--tps-dewarp) |
| Server | - | Y (/preprocess/tps-dewarp) |
| Python | - | Y |
| Rust | Y | Y |
| Dart/Flutter | - | Y |

## Files

| File | Purpose |
|------|---------|
| `src/tps_warp.h` | Public header (C API) |
| `src/tps_warp.cpp` | TPS math (solve, warp, map_point) |
| `src/tps_locnet.cpp` | Localization CNN inference + auto_dewarp |
| `src/classical_preproc.{h,cpp}` | `tps_dewarp()` wrapper |
| `models/convert-tps-loc-to-gguf.py` | PaddleOCR/PyTorch -> GGUF converter |
| `tools/dump_tps_reference.py` | Reference dumper for parity testing |
| `tests/test_tps_warp.cpp` | 19 unit tests (TPS math) |
| `tests/test_tps_locnet.cpp` | 10 unit tests (CNN loading + inference) |
| `tests/test_tps_parity.cpp` | 7 parity tests (C++ vs Python) |

## Performance

On the dev VPS (x86_64, no GPU, single-threaded):
- Localization CNN: ~70ms for 200x64 image
- TPS solve + warp: ~2ms for 200x64 image
- Full pipeline: ~230ms (includes model load from disk)

With model pre-loaded, expect ~75ms per image.
