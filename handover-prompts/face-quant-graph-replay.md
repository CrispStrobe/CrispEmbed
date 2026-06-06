# Handover: Quantized Face Model Inference via Graph Replayer

**Status: RESOLVED (2026-06-06)**

## What this was

Face models (YuNet, SCRFD) use the `replay_graph()` path in `src/cnn_embed.cpp`
for inference. Conv weights are stored as 2D `[OC, IC*KH*KW]` in the GGUF
(flattened from 4D for quantization). The graph replayer reshapes these
back to 4D `[KW, KH, IC, OC]` before calling `ggml_conv_2d`.

Three bugs in the reshape logic caused crashes:
1. Group attrs parsed AFTER reshape — depthwise convs used wrong IC
2. `ggml_n_dims()` reports 2 for 4D weights with trailing 1s like `[3,3,1,1]`
3. ggml only supports Q→F32 dequant, not Q→F16 directly

## Current state (all fixed)

### What works
- SFace F32/Q8_0/Q4_K via sequential conv blocks — tested, cos=0.9999
- AuraFace F16 via graph replayer — tested, produces correct embeddings
- YuNet F32/F16/Q8_0 via graph replayer — detects faces correctly
- SCRFD F32 runs without crash (no ONNX source to reconvert for quant test)

### What crashes
- YuNet F16/Q8_0 via graph replayer — `ggml_reshape_4d` asserts `nelements != ne0*ne1*ne2*ne3`
- SCRFD Q8_0 likely same issue

### Root cause
In `replay_graph()` (line ~1031), the Conv op handler reshapes 2D→4D:

```cpp
if (ggml_n_dims(w) == 2) {
    int64_t OC = w->ne[1], flat = w->ne[0];
    int64_t IC = (int64_t)x->ne[2];  // input channels from previous layer
    int64_t ka = flat / IC;
    ...
    w = ggml_reshape_4d(g, w, KH, KH, IC, OC);
}
```

For **depthwise convs** (group = OC), the ONNX graph stores weights as
`[OC, 1, KH, KW]` → flattened to `[OC, KH*KW]` = `[OC, 9]` for 3x3.
But `x->ne[2]` gives the input channel count (e.g., 16), not IC=1.
So `flat/IC = 9/16` which is wrong, and the reshape fails.

### The fix needed
The graph replayer's Conv handler needs to detect depthwise convs from the
graph node's `group` attribute (already parsed as `group_n` from `g%d` in
the attrs string). When `group_n > 1`, use IC=1:

```cpp
if (ggml_n_dims(w) == 2) {
    int64_t OC = w->ne[1], flat = w->ne[0];
    int64_t IC = (group_n > 1) ? 1 : (int64_t)x->ne[2];
    ...
}
```

But the current code computes `group_n` AFTER the reshape attempt (the attrs
are parsed at line ~1047). The reshape code at line ~1033 runs before attrs
parsing. **Move the attrs parsing before the reshape**, or pre-parse the
group from the node descriptor string.

## How to debug with crispembed-diff

1. **Python reference**: Run YuNet ONNX through onnxruntime, dump all
   intermediate tensors to a reference GGUF:
   ```python
   python tools/dump_vit_reference.py --model yunet --output yunet-ref.gguf
   ```
   (Needs a YuNet-specific variant of the dumper using onnxruntime hooks)

2. **C++ comparison**: Add `#include "crispembed_diff.h"` to cnn_embed.cpp,
   load the reference, and compare each `replay_graph()` node output:
   ```cpp
   crispembed_diff::Ref ref;
   ref.load("yunet-ref.gguf");
   // After each node in replay_graph:
   auto r = ref.compare(node_name, tensor_data, n_elem);
   ```

3. The first node where cos drops below 0.999 identifies the broken op.
   For this bug, it will be the first depthwise conv.

## Files to modify

| File | Change |
|------|--------|
| `src/cnn_embed.cpp` | Fix `replay_graph()` Conv handler: parse attrs before reshape, use group_n for IC |

## Build & test

```bash
cd /mnt/volume1/CrispEmbed-fresh
ninja -C build -j$(nproc)

# Test F32 (should work, no reshape needed):
build/crispembed -m /mnt/volume1/yunet/yunet.gguf --detect test_face.jpg

# Test F16 (will crash until fixed):
build/crispembed -m /mnt/storage/yunet-f16.gguf --detect test_face.jpg

# Test Q8_0:
build/crispembed -m /mnt/storage/yunet-q8_0.gguf --detect test_face.jpg
```

## Also pending

- **ViT parity**: cos≈0.8 for CLIP/SigLIP vision. QKV fusion didn't help.
  The per-layer delta is in the attention score matmul (head_dim=64 inner
  product). Possible avenues: different softmax implementation, GPU backend
  testing, matching HF's exact attention kernel.

- **Quantizer F16 guard**: The relaxed quantize guard (`!is_tiny_embd` only)
  may be too aggressive for text models — could quantize norm weights or
  small tensors that shouldn't be touched. Should add back the name filter
  for non-F16 types.
