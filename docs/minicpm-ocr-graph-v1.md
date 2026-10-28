# MiniCPM-OCR stored-type decoder experiment v1

This is a default-off **source candidate**, with no build, numerical or speed
result yet. `CRISPEMBED_MINICPM_OCR_GRAPH=1` selects it only in MiniCPM-OCR.
Unset, empty and literal `0` retain `decoder_cpu`. Existing model, source,
preprocessing, prompts, image layout, five-case gold, stop IDs and token cap
remain unchanged. Quantized tensors additionally require the existing
`CRISPEMBED_QWEN35_QUANT=1` gate. Calibration mode
`CRISPEMBED_QWEN35_IMATRIX=1` is explicitly unsupported and rejected rather
than silently skipping its observer. No default or publication follows from this file.

The original remains Transformers 5.21.0 at the checkpoint and exact pins in
[minicpm-ocr-source.json](minicpm-ocr-source.json). The original inference
blueprint and file/line decisions are in
[minicpm-ocr-inference-spec.md](minicpm-ocr-inference-spec.md). In particular,
the source defines zero-centered RMSNorm, gated full attention, partial NeoX
RoPE on equal sequential axes, causal chronology independent of coordinates,
and a hybrid linear/full-attention schedule. None is inferred from speed.

The source inventory before implementation identified these reusable pieces:

| Existing piece | Reuse in this candidate |
| --- | --- |
| `src/qwen35_decoder.cpp` | Layer schedule, residual/FFN order, zero-centered RMS arithmetic, final last-token projection |
| `src/qwen35_attention.cpp` | Gated-Q split, head norms, equal-axis RoPE, causal softmax and F32 KV layout/arithmetic |
| `src/qwen35_linear.cpp` | Matrix validation, small F32-vector loading and linear projection layout |
| `src/qwen35_delta.cpp` | **Called directly:** convolution, recurrent state update, ggml gated-delta operator and gated normalization |
| `src/minicpm_ocr_vision.cpp` | The existing backend graph allocator/borrowed weight lifetime pattern; vision arithmetic unchanged |
| `src/core/gguf_loader.h` | Existing loaded tensor ownership, original stored tensor types and backend buffers |

The new `src/qwen35_graph_decoder.cpp` is composed from those repository
routines. It sends each original stored F32/F16/Q8_0/Q4_K/Q6_K matrix directly
to `ggml_mul_mat`, explicitly selecting F32 accumulation precision. Its
per-context CPU backend is reused; graph/input/output allocations end after
synchronous readback. Loaded weight tensors are borrowed without mutation or
conversion and must have host backend buffers. Small norms, convolution
weights and vectors still use the existing F32 conversion. The norm, SiLU,
full-attention and residual arithmetic is retained from the old path. This is
a **projection-graph decoder and prefill**, not a claim that the whole network
or full-attention softmax is fused. It does not implement a new attention
kernel or copy an upstream engine implementation.

Request state remains explicitly owned by `decoder_outputs::cache`: six
full-attention F32 KV caches and eighteen linear conv/recurrent states. Cached
single-token calls receive the previous state; independent requests start with
null previous state. This preserves the qualified cache layout and chronology.
The candidate still copies growing full-attention KV vectors; eliminating those
copies and fusing more operators are separate steps after the first measured
A/B. No persistent process-global cache, weight expansion or hidden fallback is
introduced. A selected candidate failure clears output and rejects the request.

The initial profile is intentionally strict: batch1, width1024, vocabulary
248100, tied embedding/head,24 layers, alternating three linear/one full,
4096 total tokens, exact original head/RoPE/epsilon configuration, and position
axes equal to `past + arange(tokens)`. Other Qwen/Ovis/Josev engines continue
to use their existing route.

`tests/minicpm-graph` is a standalone hosted build. Its actual C++ controls
cover stored F32/F16/Q8_0/Q4_K projection against the retained dequantized
control, original bytes/types, same-context repeat, independent contexts,
unset/empty/0 gates, invalid inputs, quant gate, failure clearing, clear and
recovery. Release checks use explicit requirements, not disabled assertions.
These small fixtures do not qualify model arithmetic. The root integrator must
add `src/qwen35_graph_decoder.cpp` to MiniCPM's `CRISPEMBED_SOURCES` before
building the real CLI/static/shared library; no root CMake change is included
in this owned source patch.

Before adoption, hosted execution must retain:

1. The existing benchmark's exact uploaded source, original Transformers and
   llama.cpp external-speed baselines. No original gold recapture.
2. F32/F16 graph off/on for **all five** immutable cases (`line`, `para-en`,
   `table`, `tall`, `tiny`), all input/generated IDs, text, EOS/cap and confidence.
   Compare prefill layers/norm/logits through actual `crispembed_diff::Ref`
   lossless F32 adapters plus independent metrics at fixed0.002/0.99999;
   preserve first divergence, mean/RMS, L2 and magnitudes, even on failure.
3. A test-only, production-excluded same-input projected-stage comparison,
   including actual stored tensor types. A projection component result cannot
   substitute for the composed decoded panel. Ordinary/observed/repeat outputs
   must be neutral before comparing performance.
4. Paired per-stage and whole-case timings on the same host/thread count,
   with actual source/build and model hashes. A speed target is not evidence.

Q8_0 currently awaits the running benchmark's all-five verdict. The publisher
recipe now agrees with that benchmark: preserve the4-D
`v.emb.patch_embedding.weight` at F16. This fixes format/load compatibility;
it does not prove Q8 numerical or decoded quality.

A distinct protected Q4_K recipe must be declared before outputs. Preserve
vision, merger, patch embedding, embeddings/tied vocabulary head, norms,
convolution and all small vectors. Only decoder2-D projection matrices may
be selected. Validate every actual stored type and unchanged protected bytes,
then run the identical five-case decoded and metric gates before publication.
No Q4 model has been produced or qualified by this preparation.
