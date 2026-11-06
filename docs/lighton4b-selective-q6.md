# Default-off LightOn4B selective Q6 candidate

The generic full-matrix projection engine already dequantizes Q6_K. Its streamed
row projection and grouped-GDN projection validators still excluded Q6, as did
the LightOn4B owner. LightOn4B previously
rejected it at its narrower, exact owner/recipe boundary because only Q8, plain
Q4 and attention-Q8/FFN-Q4 layouts had been declared. This candidate adds a
separate native opt-in, `CRISPEMBED_LIGHTON4B_Q6_K=1`; old recipe admission and
stored tensors retain their previous behavior with the gate absent or0.

The custom marker is`lighton4b-q4km-q6-q8head-v1`, with UINT32
`general.file_type=15`. The loader requires the complete exact layout, rather
than trusting the file label:32 Q6_K matrices,216 Q4_K matrices and the original
Q8_0 embedding shared by the tied output head. All474 protected tensors keep
the existing F32 policy. Missing/wrong metadata, any changed matrix type,
untied/retyped output or simultaneous attention-Q8/FFN-Q4 selection is rejected
before payload loading. WASM does not enter this new candidate.

## Source-backed selective names

The policy is pinned to llama.cpp
[18ab62a8 quantization](https://github.com/ggml-org/llama.cpp/blob/18ab62a8ec5144ae5d2f65e78374ba69652d75dd/src/llama-quant.cpp),
[Qwen tensor names](https://github.com/ggml-org/llama.cpp/blob/18ab62a8ec5144ae5d2f65e78374ba69652d75dd/gguf-py/gguf/tensor_mapping.py)
and [natural layer ordering](https://github.com/ggml-org/llama.cpp/blob/18ab62a8ec5144ae5d2f65e78374ba69652d75dd/src/llama-model-loader.h).
Its Q4_K_M value-family includes fused QKV as well as separate V. The
higher-bit predicate selects layers0,1,2,3,6,9,12,15,18,21,24,27,28,29,30,31
in each32-member value/down family. For each selected LightOn layer, Q6 replaces:

- `l.layers.N.mlp.down_proj.weight`;
- `l.layers.N.linear_attn.in_proj_qkv.weight` for recurrent layers, or
  `l.layers.N.self_attn.v_proj.weight` for full-attention layers.

All exact32 names and the complete723-tensor plan are in
[the generated immutable recipe](lighton4b-selective-q6-v1.json). This matches the
upstream selective type rule, with a deliberate exception: the tied output stays
Q8_0, instead of the upstream default Q6_K. It is therefore called a Q4_K_M-style
custom recipe, not an identical whole llama.cpp export. Native grouped-value
row/column order remains the frozen LightOn layout; upstream conversion reorders
some recurrent projections, so an upstream GGUF cannot be substituted wholesale.

The exact storage increase over the current plain Q4 payload is164,904,960 bytes
(about157.27MiB), plus metadata differences. This is a storage calculation, not
an inference memory or decoded-quality measurement.

## Generation and verification boundaries

A future candidate must be quantized directly from the original BF16 checkpoint
or an authenticated unquantized F16 baseline. Q8→Q6 or Q4→Q6 requantization does
not recover original information and is prohibited by the plan's input inventory
check. No F32 model is needed: source BF16/F16 values can be converted in bounded
row chunks for quantization. The existing converter exports F16/F32; the generic
quantizer already accepts explicit `--tensor-type NAME=q6_k` overrides. Its plain
`q4_k` selector does not itself implement this mixed policy and writes the S
header14 when standard labels are selected. A candidate writer must explicitly
write the verified M header15 and custom marker after checking all723 actual
shapes/types and protected payload hashes. The current tool emits the exact plan
and validates unquantized source inventories; it has not generated a model file.

With both weight-rounding and Q6 gates enabled, dequantized named text Q6 values
also receive the existing BF16 oracle-weight conversion. Q6 retains the old
serial dequant fallback; no parallel speed claim is added. Quantized storage stays
unchanged. This option makes no new arithmetic or decoded parity claim.

The bounded model-free controls exercise header-only exact ownership, corruption
of each249 quantized matrix types, strict metadata/gate negatives, independent
scalar Q6 decoding of65,792 synthetic values, native projection/row tiling,
BF16 conversion and stored-byte preservation. Hosted builds also cover the real
native runner and preflight, plus the old native/mixed/BF16-weight controls.
No actual Q6 checkpoint, calibration, held-out case or EOS result has run yet.
A model job remains blocked on the current calibration decision; no default,
release, model alias or upload is authorized by this structural test.


The first hosted model-free build42206305308 compiled every target, then failed
control703 at the actual streamed Q6 projection. All preceding independent raw
Q6 decoder and full projection controls passed. That exposed the narrower stream
whitelist; inspection found the equivalent grouped-GDN whitelist. Both routes
now require the same explicit LightOn native Q6 gate, preserving legacy rejection
when it is off. The retry adds actual synthetic grouped-GDN prefill and cached
decode equivalence against independent F32 dequantized weights, plus gate-off
rejection and immutable cache controls. The initial failed receipt is retained.
