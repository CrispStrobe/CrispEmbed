# WASM layer3 same-input attribution v1

This is test-only source preparation after the reviewed negative
[rounding A/B40391544739](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40391544739).
Its original sentence passes through `layer_2` within `9.54e-7`, then
`layer_3` reaches maximum absolute error `0.00291892886`. The unchanged ten-case
panel rejects eight mode1 final vectors. Keep the default-off rounding candidate,
legacy alternatives, original artifacts and all sixteen negative labels intact.
No model or compiler execution was performed while preparing this lane.

The zero-input reusable [workflow](../.github/workflows/wasm-layer3-attribution-v1.yml)
uses the frozen `580e3ae3be3671ddce5e945b0725d0583ec34110` runtime with ggml
`a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. Its executable helpers come from the
new claimed diagnostic branch, whose main merge also preserves unrelated work.
Actual builds use a separate disposable checkout of frozen580; later production
source/build changes are not silently accepted. Exact source and overlay hashes,
diffs, compile commands and service identities are retained. Exact prior compiler
flags are preserved: encoder/wrapper use pthreads without SIMD, while the pinned ggml CPU kernel and WASM quantizer use SIMD128 and pthreads.
Generated Emscripten commands must match that matrix before execution. The browser
server sends COOP `same-origin` and COEP `require-corp`; the page requires
`crossOriginIsolated` and SharedArrayBuffer before module initialization. The native
diagnostic replay export must appear unmangled in `nm -D --defined-only`;
`<cmath>` is present in the frozen encoder source.

Before model access, require the completed negative source580 service run, exact
small review ZIP and report SHA, raw ZIP ID/size/SHA and every declared raw member.
The original model comes only from initial fixture12671621333, with SHA256
`77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2`.
Actual GGUF names, types, shapes and full tensor-payload hashes select the supported
activation converter. The filename does not establish a tensor's quantization type.
The [machine-readable contract](wasm-layer3-attribution-v1.json) freezes these pins.

A disposable test overlay retains23 layer3 subnodes: block input; raw and biased
Q/K/V; actual flash-attention output; raw/biased attention output projection;
attention residual/normalization/affine results; raw/biased FC1; exact GELU; raw/
biased FC2; FFN residual/normalization/affine result. Independent view names preserve
ordinary node names, and OUTPUT flags retain each view's allocation owners before
allocation. No new numerical operation replaces production arithmetic.

A second disposable overlay instruments the actual ggml `mul_mat` conversion
barrier. It records the selected-plan workspace bytes, input floats, actual weight
payload, stored and activation types, plan flag and selected-trait first-channel
dots for all six layer3 matrices. The dot evaluation is labeled separately from
the graph output. This observes the workspace used by the operator; it never
substitutes the shared global trait converter as proof of the per-backend gate.
Q8_0 FP16 scales and signed quantized bytes, or Q8_K F32 scales and preserved sum
fields, are recorded. Required captures must exist; a specialized/llamafile path
that bypasses this hook rejects the diagnostic rather than inventing observations.

Native1/2 and browser gate0/1 ordinary outputs must exactly match the frozen A/B
final bytes. Retained captures must preserve those same outputs and all eight
original stage hashes. Token IDs must remain the original twelve IDs. Any changed
anchor, absent subnode, unsupported actual type or missing workspace blocks the
attribution. Compiler/container differences are not excused by a numerical budget.

After those observations, replay each actual projection on two distinct axes:
**native captured input** through native and browser operators, then **candidate
captured input** through both operators. A test-only export builds actual ggml
matmul graphs using the loaded original weight and existing backend context.
Each axis requires identical input bytes across runtimes, exact original weight
bytes and own-input reproduction of its full-graph raw projection. Compare full
projection errors/magnitudes/cosine, activation byte and scale mismatches and
selected first-channel dots. Recheck the browser's ordinary output after replay.

Full named subnodes are compared by `crispembed-diff` and an independent raw F32
reader. Maximum absolute `0.002` and minimum row cosine `0.99999` remain unchanged;
no intermediate or full-panel failure is waived. Successful attribution means
that observations are source-bound and neutral, not that the candidate passes.
If the same-input operators agree while native/candidate-input conversions differ,
the evidence supports quantization-boundary sensitivity at that operator. If the
same-input operators differ, inspect conversion/scales before dot accumulation.
If the first difference is in attention, residual, normalization or GELU, those
operators need a separately frozen local replay. Do not infer either conclusion
from final-vector improvement alone.

Hosted execution is limited to60 minutes and two build workers, with no framework
installation. Large model/array/module artifacts stay remote for90 days. Small
reports, compiler logs and per-phase rejection records are uploaded even on failure.
There is no production API/default/build edit, asset publication, representative
retrieval-quality claim or qualification change in this lane.
