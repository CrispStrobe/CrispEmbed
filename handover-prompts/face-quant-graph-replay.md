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

## Fix applied (commits 83446e5, 2fcde98)

In `src/cnn_embed.cpp` replay_graph() Conv handler:

1. **Parse attrs first**: `stride`, `pad`, `group_n` parsed from `n.attrs`
   before the reshape block, so `is_dw = (group_n > 1)` is available for IC.

2. **Element count validation**: Use `ggml_n_dims(w) <= 2` (catches OC=1
   weights that report ndims=1), then validate `KW*KH*IC*OC == nelements`
   before applying reshape. Skips genuine 4D weights like `[3,3,1,1]`.

3. **Q→F32→F16 dequant chain**: `ggml_is_quantized(w->type)` triggers
   cast to F32 first, then F32→F16 for ggml_conv_2d.

## Test results

| Model | Type | Status | Detection |
|-------|------|--------|-----------|
| YuNet (old GGUF, 4D weights) | F32 | PASS | 1 face, conf=0.749 |
| YuNet (flat GGUF, 2D weights) | F32 | PASS | 1 face, conf=0.749 |
| YuNet (flat) | F16 | PASS | 1 face, conf=0.749 |
| YuNet (flat) | Q8_0 | PASS | 1 face, conf=0.731 |
| SCRFD | F32 | PASS | Runs, 0 faces (input size issue) |

## Reference data created

- `/mnt/storage/yunet-ref.gguf`: 106 ONNX intermediate tensors for diff
- `/mnt/storage/yunet-flat.gguf`: Reconverted YuNet with 2D-flattened weights
- `/mnt/storage/yunet-f16.gguf`: F16 quantized
- `/mnt/storage/yunet-q8_0.gguf`: Q8_0 quantized

## Remaining (not in scope)

- **Quantizer F16 guard**: Investigated — the guard is fine. All 2D tensors
  in text models are weight matrices; norm/bias tensors are 1D (excluded).

- **ViT parity** (cos≈0.8): Separate investigation needed. No ViT models
  cached locally. Per-layer attention divergence, likely matmul precision.
