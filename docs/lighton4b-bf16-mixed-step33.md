# Original mixed BF16 shared-prefix step33: passive activation replay

One bounded source-only CPU replay examines the first genuine mixed BF16 versus
native decision difference at zero-based position33. The genuine178-token and
native156-token runs share all33 preceding IDs. Replay original greedy generation
only through34 score rows, including the divergent decision; this uses the
genuine mixed prefix, not the different official-unquantized prefix after25.
Native trace/full head-rounding A/B is a separate already authorized job; this
worker adds no native inference or artifact credential.

The [closed protocol](lighton4b-bf16-mixed-step33-v1.json) retains all original
checkpoint/package/processor/input pins, fixed153Q8+96Q4/474protected tensor map,
effective parameter hashes, tied head and both baseline/first-score anchors.
Full capture SHA `ed4ef1b60c41d4d3b2a131d107c2563ee971c931873202f5f470683ebb9a1d58`
is authenticated through immutable public evidence source. All34 generated IDs
and complete score hashes must exactly match the retained178-row run. This is
the observer-neutrality gate; no new gold or tolerance is introduced.

Passive forward hooks copy actual values only at the34th mixed root call:

- Decoder input embedding and actual rotary position IDs, preserving full source
  position shape plus the final three rotary axes used by the model.
- All32 decoder layer outputs, final norm and actual head input/output.
- Layer0 input norm, QKV/z/a/b, attention, post-attention residual, post norm,
  FFN gate/up, actual activation entering down projection, and down output.

There are50 unique arrays, F32 storage preserving actual BF16 values/dtype metadata,
at most8MiB total. Names align with the native trace, with three extra source
boundaries (`positions.full`, `head.input`, `head.output`). Full source head
output is compared with the retained generated score hash rather than assuming
the absence of logits processing. Finalnorm/head-input bytes must agree.
No cache state, parameters, methods or returned tensors are modified by hooks.
All34 score hashes/IDs must remain exact; shape/count/finite/cap checks apply.

The hook blueprint is checked against the actual authenticated Transformers5.5.4
model SHA `aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048`,
matching [the official upstream source](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The original source removes a possible text-position plane before using the
three rotary axes, so the full shape is retained and the native comparison uses
those three axes. Source operators stay unchanged. CPU capability, Torch git
revision, thread count and oneDNN enablement are recorded without inferring the
internal GEMM route from configuration alone.

Compare actual corresponding native arrays only after checking shared input IDs,
positions, frozen native score identity and native observer/cache neutrality.
Native dequantized matrices remain F32; genuine source parameters are BF16.
An embedding/hidden difference therefore cannot automatically be called a kernel
bug. Same-input and effective-weight precision controls remain necessary when
attributing RMS/QKV or head differences. The native final-head BF16 tie correction
and full decoded A/B remain independent evidence.

This is a34-token partial diagnostic, deliberately no EOS/full-completion claim.
Require35 audited root calls including the one-token official anchor and exactly
one mixed vision call. Use one CPU worker with45-minute model bound and the
existing22GiB initial RAM/27GiB scratch/2GiB output, small-chunk/cache and final
8GiB effective/2GiB hard-margin guards. Three added model-free observer controls
plus17 inherited controls run before model work. Large score/activation arrays
stay remote; no local model/build, F32 model, upload, new PR or default/release
change follows from preparation.
