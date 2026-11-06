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


[Hosted42207472045](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42207472045)
at`20199554a92e1aba04fc3c28593d717d53892693` now passes all713 controls,
including actual Q6 streamed projection and grouped-GDN prefill/cached decode.
The old native, mixed-policy and BF16-weight controls also pass; native/preflight
binaries compile and retain argument rejection. The
[complete receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q6-controls/q6-controls.json)
records binary hashes, exact32-name digest and zero model payload/inference.
Artifact13498105831 is retained90 days, SHA
`6486186a9ca981051add8f20312d4931e687634aed82411116c58bb5b64028f8`.
This closes structural/operator support only. Actual Q6 source conversion,
checkpoint payload validation and calibration/held-out decoded quality remain open.

## Prepared direct-source conversion lane

The proposed [conversion/first-score protocol](lighton4b-q6-conversion-protocol-v1.json)
preserves691 payloads byte-for-byte from the authenticated plain Q4 file,
including all216 remaining Q4 matrices, the Q8 tied head and474 protected tensors.
Only the32 selected matrices are generated anew, directly from the authenticated
original BF16 checkpoint. No existing Q4/Q8 value is decoded into a Q6 input.
The bounded C++ helper processes whole rows in at most65,536 values; source BF16,
new Q6 payload and decoded-BF16 hashes are independently recorded. Python verifies
every copied/replaced tensor, preserves raw tokenizer/metadata bytes, and compares
all32 native Q6 decoded-BF16 hashes against the independent gguf decoder.

The source checkpoint, base and new candidate total at most17,306,578,488 bytes.
Require27GiB free shared filesystem space and8GiB effective RAM before transfer;
row-quantization buffers stay below2MiB; raw copies use4MiB and header cloning
reads at most64MiB, with a conservative512MiB transient Python allowance.
Builds and native temporary outputs need extra
space within that guard. Before each first-score run, advise clean checkpoint/base
pages away and recheck8GiB effective RAM plus2GiB cgroup hard margin. One worker,
serial quantization/build parallelism2, native threads4 (matching baseline controls)
and a90-minute worker deadline are proposed. The shared workspace does not
have the required disk headroom and must never receive these large files.

Only the fixed calibration table/Arabic captures and their first score are read.
The original batch JSON is intentionally excluded: older transport helpers also
inspect a held-out cap flag, so they must not be reused for this selection lane.
Baseline and Q6 use the same binary/worker/gates, with baseline first-score hashes
required to match retained calibration controls before measuring the new candidate.
One-token scores are diagnostic; no completion or decoded-quality claim follows.
Held-out captures/labels remain in the separate lane. The model job is not yet
dispatched: resource/source review remains required first. The closed calibration-only worker
and model-free converter controls are now prepared separately.


The [model-free converter proof42210897150](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42210897150)
passes:65,792 framed BF16 values cross the65,536-value chunk boundary; output
matches the existing quantizer supplied an independent F32 fixture, and native
versus independent gguf decoded-BF16 hashes match. Neighbor bytes are unchanged
and a wrong original BF16 source hash is rejected. No checkpoint/model runs.
The [receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q6-conversion-controls/q6-conversion-controls.json)
and artifact13497839615 are retained90 days.

A complete closed worker is prepared at`tools/ci-heavy/lighton4b_q6_candidate.py`.
It reads only the two individual calibration receipts, five prepared inputs per
case and their first-score vectors; it never downloads the original batch JSON.
The exact future dispatch (not executed) is:

```sh
gh workflow run heavy-cpu.yml --ref audit/lighton4b-q6-conversion-20261106   -f script=tools/ci-heavy/lighton4b_q6_candidate.py   -f runner=ubuntu-latest -f build_mode=portable -f pip= -f args=
```

Only the dedicated Q6 job can run; generic model jobs are excluded. Its fixed
worker deadline is90 minutes, with110 minutes total CI allowance for setup and
artifact upload. Native uses four threads to match the retained baseline hashes;
build parallelism is2 and conversion is serial. The frozen calibration images
have86/82 input tokens and288/272 patches. Candidate tensors/scores/audits remain
remote for90 days, with no model-hub upload, alias, default or release change.
Dispatch still awaits resource/source review and queue coordination.
