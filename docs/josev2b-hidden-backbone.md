# Explicit Josev hidden-only decoder profile

The separate experimental `josev2b_backbone` component accepts exactly320 F32
text tensors,2048 hidden channels,6144 FFN channels,24 layers with18 linear and
six full-attention blocks, and the original norm/RoPE profile. Both
`CRISPEMBED_QWEN35_OCR=1` and `CRISPEMBED_JOSEV_BACKBONE=1` are required.
Unknown/missing/extra shapes, F16/quantized tensors, an output vocabulary matrix,
invalid token IDs, empty input or more than512 tokens reject. This bound follows
the frozen original four-case state-clipping contract; it does not qualify the
configuration's262,144-position capacity.

`src/josev2b_backbone.cpp` maps only this explicit profile and retains all24
layer outputs and every final norm token row. It supplies sequential text
coordinates on each of the three rotary axes. The internal hidden-only decoder
returns before the vocabulary projection and does not generate tokens or consume
previous cache state. Failed runs clear caller-visible output. Existing
`map_small_decoder` remains byte-identical and rejects the2048 profile; ordinary
`decoder_cpu` retains its symbol and generation behavior through the original
path. The new component is absent from the root library/C API/registry build.

The bounded `josev-backbone-build` metadata phase uses the existing runtime's
exact ggml dependency and a standalone portable two-worker CPU build. Actual
C++ controls construct metadata without model payload allocation, exercise the
valid source profile, disabled/literal-zero gates, malformed dimensions,
unqualified precision, unexpected vocabulary head, recovery, old-profile
rejection and stale-output clearing. These are structural controls, not model
parity or a new release.

Remaining before composition:

- Authenticate the original checkpoint and all320 text tensor names/shapes,
  promote original BF16/F32 exactly and read back every converted byte remotely.
  Preserve Apache notices and retain converted/raw artifacts remotely.
- On hosted CPU resources, compare embedding, all24 layer outputs and final norm
  against all four frozen original inputs with actual `crispembed-diff::Ref`,
  independent errors/magnitudes and fixed budgets. Do not regenerate gold.
  Trace rotary/attention/delta boundaries at the first differing layer.
- Retain the actual head component audit and compare the complete composed typed
  responses with the original. Component cosine alone does not qualify it.
- Run the existing Ovis/LightOn decoder numerical and decoded controls on the
  exact modified source before integration. Public tokenization/schema/API and
  representative task quality follow original-first composed parity.

See [source and output semantics](josev2b.md) and
[the independent original receipt](../tests/results/josev2b/original-review.json).
