# Handover: RT-DETRv2 Layout Detection — Score Parity

## Task

Fix the RT-DETRv2 document layout detection model (`layout_detect.cpp`)
so that detection scores match the ONNX reference. Python ONNX produces
7 detections with max_score=0.65. C++ currently produces 0 detections
(max_score=0.047). The architecture is fully implemented but scores are
~14× too low.

## Files

- **C++ implementation**: `/mnt/volume1/CrispEmbed-layout/src/layout_detect.cpp` (1441 lines)
- **C++ header**: `/mnt/volume1/CrispEmbed-layout/src/layout_detect.h`
- **Converter**: `/mnt/volume1/CrispEmbed-layout/models/convert-layout-to-gguf.py`
- **Test binary**: `/mnt/volume1/CrispEmbed-layout/tests/test_layout_detect.cpp`
- **Python reference dumper**: `/mnt/volume1/CrispEmbed-layout/tools/dump_layout_reference.py`
- **ONNX model**: `/mnt/storage/models/docling-layout-heron/model.onnx` (164 MB, Apache 2.0)
- **GGUF model**: `/mnt/storage/models/layout-heron-f32.gguf` (161 MB)
- **Test image**: `/tmp/layout_test_640.png` (640×640 synthetic document)

## Worktree and build

```bash
cd /mnt/volume1/CrispEmbed-layout          # worktree on branch feat/layout-detect-fix
cd build
CCACHE_DIR=/mnt/volume1/.ccache ninja test-layout-detect   # ~3s incremental
timeout 180 ./test-layout-detect /mnt/storage/models/layout-heron-f32.gguf /tmp/layout_test_640.png 0.3
```

Set `LAYOUT_DEBUG=1` for verbose per-tensor output. All debug fprintf is
gated behind this env var via the `LDBG()` macro.

## Diff harness methodology

### 1. Generate Python reference

The ONNX model can emit any intermediate tensor by adding it to
`model.graph.output`:

```python
import onnx, onnxruntime as ort, numpy as np
model = onnx.load('/mnt/storage/models/docling-layout-heron/model.onnx')
# Add any node output as a graph output:
model.graph.output.append(onnx.helper.make_empty_tensor_value_info('layer_norm_1'))
sess = ort.InferenceSession(model.SerializeToString())
# Run:
img_u = ...  # [1, 3, 640, 640] uint8 CHW
sizes = np.array([[640, 640]], dtype=np.int64)
outs = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})
```

The test image is created by:
```python
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (640, 640), (255, 255, 255))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
draw.text((50, 50), 'Section Header', fill=(0,0,0), font=font)
draw.text((50, 100), 'Body text here.', fill=(40,40,40), font=font)
draw.rectangle([50, 200, 300, 400], outline=(0,0,0), width=2)
```

The ONNX takes uint8 input, internally casts to float and divides by 255.
The C++ `detect_file` does `pixel / 255.0f` (no ImageNet normalization —
the BN folding absorbed normalization into the conv weights).

### 2. Save reference to GGUF

```python
import gguf
writer = gguf.GGUFWriter('/tmp/ref.gguf', 'layout-ref')
writer.add_tensor('tensor_name', data.astype(np.float32),
                   raw_dtype=gguf.GGMLQuantizationType.F32)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

### 3. Compare in C++

The `crispembed_diff.h` header loads a reference GGUF and compares:
```cpp
#include "crispembed_diff.h"
crispembed_diff::Ref ref;
ref.load("/tmp/ref.gguf");
auto r = ref.compare("tensor_name", cpp_data, n_elements);
// r.cos_min, r.max_abs, r.is_pass()
```

### 4. Mark ggml tensors for output

In the ggml graph, mark tensors with `ggml_set_name` + `ggml_set_output`,
then read after `ggml_backend_graph_compute`:
```cpp
ggml_set_name(tensor, "my_name");
ggml_set_output(tensor);
// ... after compute:
ggml_tensor* t = ggml_graph_get_tensor(gf, "my_name");
float vals[5];
ggml_backend_tensor_get(t, vals, 0, 5 * sizeof(float));
```

The current code already has debug output tensors: `aifi_qkv`, `aifi_attn_raw`,
`aifi_attn_out`, `aifi_resid`, `aifi_norm1`, `aifi_out`, `ip3/ip4/ip5`,
`bb_stem`, `bb_s{0-3}_b{0-N}`, `csp0_conv1`, `csp0_bn{0-2}`, `csp0_conv2`.

Pre-generated Python references exist at:
- `/tmp/layout-ref-backbone.gguf` — backbone relu outputs
- `/tmp/layout-ref-perlayer.gguf` — all 52 relu outputs
- `/tmp/layout-ref-stages.gguf` — backbone C3/C4/C5 + encoder S3/S4/S5
- `/tmp/layout-ref-csp.gguf` — CSP block 0 internals
- `/tmp/aifi-attn-ref.gguf` — AIFI Q/K/V/scores/attended/out_proj

## What is PROVEN correct

These have been verified to 4-decimal-place parity against the Python ONNX:

| Component | C++ value | Python value | Verified how |
|-----------|-----------|-------------|-------------|
| Backbone stem (maxpool) | max=1.8347 | max=1.8353 | Range match |
| Backbone s0.b0 | max=3.1542 | max=3.1542 | **Exact** |
| Backbone s1.b0 | max=4.4234 | max=4.4234 | **Exact** |
| Backbone s2.b0 | max=3.8976 | max=3.8976 | **Exact** |
| Backbone s3.b0 | max=1.4070 | max=1.4070 | **Exact** |
| Input proj ip3 | [-8.3031, 8.9570] | [-8.3031, 8.9570] | **Exact** |
| Input proj ip4 | [-13.8728, 17.8263] | [-13.8728, 17.8263] | **Exact** |
| Input proj ip5 | [-12.9403, 13.4626] | [-12.9403, 13.4626] | **Exact** |
| AIFI QKV combined | [-84.7360, 67.4371] | [-84.7360, 67.4371] | **Exact** |
| AIFI attn_raw | [-4.2, 3.9] | [-3.3, 4.1] | Close |
| Converter BN folding | 55 Conv→Mul→Add patterns | — | Converter output verified |

## What is KNOWN broken

| Component | C++ | Python | Gap |
|-----------|-----|--------|-----|
| AIFI attn_out (out_proj) | [-6.5, 6.9] | [-11.7, 7.7] | ~55% |
| AIFI final (norm2) | [-4.8, 4.3] | [-5.6, 5.3] | ~82% |
| Encoder s3 | 62.1 | 91.0 | 68% |
| Encoder s4 | 50.0 | 98.5 | 51% |
| Encoder s5 | 79.7 | 77.7 | 103% (matches) |
| Decoder cross-attn (norm2) | [-5.5, 4.6] | [-11.0, 9.7] | ~48% |
| **Max class score** | **0.047** | **0.65** | **7%** |

## Architecture overview

```
Input [3, 640, 640] uint8
  → div(255) → [0,1] float
  → ResNet-50-D backbone (BN-folded Conv→Mul→Add, stride on branch2b, avgpool shortcut)
    → C3 [256, 80, 80], C4 [256, 40, 40], C5 [256, 20, 20]  (after input_proj)
  → AIFI encoder on S5 (self-attention + FFN on 400 spatial tokens)
  → FPN top-down (lateral SiLU → upsample → concat → CSP block) × 2
  → PAN bottom-up (downsample SiLU → concat → CSP block) × 2
  → Decoder input_proj (1×1 conv per scale, CPU-side via cpu_linear)
  → 6× Transformer decoder (self-attn + deformable cross-attn + FFN)
  → Detection heads (bbox MLP + class sigmoid)
```

## Specific bugs / areas to investigate

### 1. AIFI attention head interleaving (HIGH PRIORITY)

The AIFI self-attention output reshape is WRONG and this is the primary
source of the score gap.

**The problem**: After attention, the output `[hd=32, N=400, heads=8]`
must be reshaped to `[256, 400]` with correct head interleaving.
ONNX does: `[8, 400, 32]` → `[400, 256]` (concatenate heads).
C++ does: `[32, 400, 8]` → `[256, 400]` without permute (wrong order).

When the correct permute `[hd, N, heads] → [hd, heads, N] → reshape [D, N]`
is applied, the `attn_out` max MATCHES Python (7.60 vs 7.70), but the
encoder s5 drops from 79.7 to 11.3. This is UNEXPLAINED.

**Investigate**: Why does the correct head interleaving cause s5 to drop?
The out_proj is a full 256×256 linear that should absorb any permutation.
Check if the `linear` function (ggml_mul_mat auto-transpose) handles the
out_proj weight correctly for the permuted input.

### 2. `linear` auto-transpose (INVESTIGATE)

The `linear` helper at line ~390 does:
```cpp
if (w->ne[0] != x->ne[0] && w->ne[1] == x->ne[0])
    w = ggml_cont(g, ggml_transpose(g, w));
return ggml_mul_mat(g, w, x);
```

For 256×256 square weights, `ne[0]==ne[1]` so no transpose is applied.
But `ggml_mul_mat(W, x)` computes `W^T @ x`. For ONNX Gemm(transB=1),
this happens to be correct (verified algebraically). BUT: if the input
features are in wrong order (wrong head interleaving), the weight
produces wrong results even though the math is correct.

### 3. `cpu_linear` weight indexing (VERIFIED CORRECT)

The CPU-side linear uses `W[i + o * in_d]` (line ~880). This was changed
from `W[o + i * out_d]`. For GGUF tensors stored as `[ne0=in_d, ne1=out_d]`,
the element `weight[o, i]` is at `data[o * in_d + i]`. The computation
`y[o] = sum_i W[i + o*in_d] * x[i]` = `sum_i data[o*in_d + i] * x[i]`
= `sum_i weight[o, i] * x[i]` = `weight @ x`. This is correct for ONNX
MatMul(input, weight) and Conv2d(input, weight) with `(OC, IC)` layout.

**BUT**: For decoder weights that were renamed from `val_XXXX` (the converter's
anonymous weight renaming), the data layout may be different. The converter
renames weights by tracing MatMul→Add(bias) pairs in the ONNX graph. Verify
that the renamed weights have the correct `[ne0, ne1]` orientation.

### 4. Decoder memory (decoder.input_proj)

The decoder uses its own `input_proj` convolutions (lines ~900-935) to
project encoder features before building the memory for cross-attention.
These are applied via `cpu_linear`. The decoder.input_proj output ranges
are `[-7.6, 8.4]` for scale 0 vs Python `[-22.2, 22.2]` — 38%.

**Check**: Are the decoder.input_proj weights loaded correctly? The weights
have `.weight_bias` (BN-folded bias). The `load_conv` function looks for
`.bias` then `.weight_bias`. Print the weight data and compare with ONNX.

### 5. CSP block structure

Each CSP block does:
```
cat = concat(branch_input_a, branch_input_b)  [512ch]
a = conv1(cat) → SiLU → bottleneck0 → SiLU → bottleneck1 → SiLU → bottleneck2 → SiLU
b = conv2(cat) → SiLU
out = Add(a, b)
```

This was verified against ONNX node trace (nodes 302-318). The CSP
internals match at 82-96%. No known bugs, but the compounding error
from AIFI attenuates features through the 12 bottleneck convolutions.

### 6. Deformable cross-attention offsets

The offset processing (line ~1130) uses:
```cpp
float px = ref_cx + dx * 0.25f * ref_w * 0.5f;
float py = ref_cy + dy * 0.25f * ref_h * 0.5f;
float sx = px * fW;  // feature map pixel coords
```

This was derived from the ONNX nodes 474-486. The ONNX applies:
- offsets * 0.25 (per level×point scale)
- offsets * ref_wh (reference box size)
- offsets * 0.5
- positions = ref_xy + scaled_offsets
- grid = positions * 2.0 - 1.0 (for GridSample)

The C++ skips the grid normalization (positions are directly converted to
pixel coords). This may be wrong — the positions should be in [0, 1]
normalized coords, then multiplied by feature map size.

### 7. Decoder self-attention QKV (IN cpu_linear)

The decoder self-attention uses `cpu_linear` with in_proj_weight `[256, 768]`
(ne0=256, ne1=768). The `cpu_linear` does `y[o] = sum_i W[i + o*in_d] * x[i]`
with in_d=256, out_d=768. This gives Q/K/V combined output `[768, 300]`.

The dec0_norm1 (self-attn output) matches Python at 96%, so this is
approximately correct. But the decoder weights from the converter's
val_XXXX renaming should be verified.

## ONNX graph reference

Key ONNX node ranges for the test image:

```
Backbone:
  relu (stem conv1): [0, 2.12] shape [32, 320, 320]
  relu_23 (c3):      [0, 4.82] shape [512, 80, 80]
  relu_41 (c4):      [0, 3.79] shape [1024, 40, 40]
  relu_50 (c5):      [0, 56.3] shape [2048, 20, 20]

Encoder:
  getitem (ip3):     [-8.30, 8.96] shape [256, 80, 80]
  getitem_3 (ip4):   [-13.87, 17.83] shape [256, 40, 40]
  getitem_6 (ip5):   [-12.94, 13.46] shape [256, 20, 20]

AIFI:
  add_1599 (input+pos):    [-13.94, 14.46] shape [1, 400, 256]
  linear (Q):              [-28.53, 53.99] shape [400, 1, 256]
  linear_1 (K):            [-84.74, 67.44] shape [400, 1, 256]
  linear_2 (V):            [-77.38, 46.13] shape [400, 1, 256]
  bmm (scores):            [-1967.9, 126.4] shape [8, 400, 400]
  softmax:                 [0.0, 1.0] shape [8, 400, 400]
  bmm_1 (attended):        [-3.32, 4.12] shape [8, 400, 32]
  linear_3 (out_proj):     [-11.66, 7.70] shape [400, 256]
  add_1703 (residual):     [-19.08, 15.02] shape [1, 400, 256]
  layer_norm (norm1):      [-7.85, 6.03] shape [1, 400, 256]
  layer_norm_1 (norm2/out):[-5.56, 5.31] shape [1, 400, 256]

CSP block 0 (FPN S5→S4):
  getitem_18 (conv1):      [-11.01, 17.06]
  silu_1 (conv1 SiLU):     [-0.28, 17.06]
  conv2d_60 (bn0):         [-24.26, 29.85]
  silu_4 (bn2 SiLU):       [-0.28, 51.64]
  silu_5 (conv2 SiLU):     [-0.28, 12.07]
  add_1876 (CSP output):   [-0.56, 53.88]

Encoder final:
  encoder_s3:              [-0.56, 91.01] shape [256, 80, 80]
  encoder_s4:              [-0.56, 98.52] shape [256, 40, 40]
  encoder_s5:              [-0.56, 77.68] shape [256, 20, 20]

Decoder memory:
  concat_4:                [-22.19, 22.23] shape [8400, 256]

Decoder layer 0:
  layer_norm_3 (norm1):    [-6.76, 4.38] shape [300, 256]
  layer_norm_4 (norm2):    [-10.97, 9.67] shape [300, 256]
  layer_norm_5 (norm3):    [-8.38, 7.55] shape [300, 256]

Final:
  max score:               0.6496
  detections (>0.3):       7
```

## Recommended approach

1. Start by fixing the AIFI head interleaving issue. The `attn_out` matches
   Python when the correct permute is applied (7.60 vs 7.70) but s5 drops.
   This is the critical path — trace WHY s5 drops with correct permute.

2. If the head interleaving is unfixable due to the linear/out_proj
   interaction, investigate whether the `linear` function needs explicit
   transpose for the AIFI out_proj weight specifically.

3. After AIFI is fixed, the encoder features should match, making the
   decoder memory correct, which should bring detection scores up.

4. For each fix, use the diff harness: dump Python reference → mark C++
   tensor → compare after compute. The first stage where cos_min drops
   below 0.999 is where the bug lives.
