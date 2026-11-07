# Captured-input attention primitive investigation

Initial source `8f747ee2e` passed nine hosted setup tests, authenticated the frozen
physical input and eleven layer3 source tensors, and compiled its native helper.
It failed after 100.46 seconds at the first native-helper invocation: borrowed
weights lacked the CPU backend buffer required by `tensor_floats`. No native
hidden byte closure or numerical attention comparison occurred. This is a
probe setup failure, not an attention/cache quality result.

[The review](../tests/results/qwen35-ocr/lightonocr3-4b-attention-primitives/v1-setup-negative/review.json)
and retained-byte index preserve the original receipt hashes and initial failure.
Raw large arrays remain remote. The proposed tools-only fix `5f82c4d0a` attaches
owned CPU buffers to eleven tensors (about67.24 MB extra transient storage),
leaving production access checks unchanged. It is not yet measured.

The probe must reproduce actual captured native first16 layer3 hidden and scalar
gated context exactly before interpreting duplicate Q/K/V. Pinned upstream
F16/F32-cache graphs then consume identical native post-RoPE Q/K/V; actual
upstream hidden2 is not captured. Batched4/16 primitives cannot establish
full-model parity, tokenwise cache correctness, decoded quality or shipping
qualification. The experimental physical path and all arithmetic alternatives
remain default-off; the prior six-arm RMS regression remains unchanged.

## Corrected probe: actual native closure and conditional source graph

Tools-only source `5f82c4d0abd7785d72f24e9b381930b1fe0ec683` completed in
218.22 seconds. All nine setup tests, actual native layer3 first16 hidden, scalar
gated-context byte closure and eight composed/observed graph pairs pass. The
[exact report](../tests/results/qwen35-ocr/lightonocr3-4b-attention-primitives/v2/attention-primitives.json)
has SHA256 `fa9ae86949ee15e475b41b35ae1ccc54b31f93957acf055e5a250d09302bdfe7`.
Independent review rehashed37 needed stage files (9,699,328 bytes), recomputed
all26 N16 metrics exactly and separately confirmed actual scalar pooled bytes.

On identical native post-RoPE Q/K/V, F32-cache graph stages are byte-identical
between the pinned backends. F16-cache graphs first differ at raw QK: N16
RMS0.0018778863, max0.03656006 against reference magnitude330.2810. Context
RMS is0.0000956899 and gated-context RMS0.00000818464. These are primitive
results, not errors against an actual upstream layer3 hidden-state capture.

The difference has a concrete source-level explanation. The native fork's
[F16 CPU trait](https://github.com/CrispStrobe/ggml/blob/3a27b6c665b191cb230be367db5c52d81dd3e2ac/src/ggml-cpu/ggml-cpu.c)
uses an F16-by-F32 dot to preserve F32 inputs, explicitly protecting other
models from F16 overflow. The pinned upstream
[trait](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-cpu/ggml-cpu.c)
uses F16-by-F16, so Q and softmax P are converted before KQ/PV dots. This
intentional global fork behavior must remain intact.

Scalar F16 K/V alone differs from the source graph QK by RMS0.0018779544;
additionally rounding Q reduces that to0.00000740503. Additionally rounding P
reduces context RMS to0.0000126761 and gated RMS to0.000000739914; remaining
dot/softmax/gating reduction differences are measured, not waived. Actual native
scalar pooled versus conditional upstream F16 graph has RMS0.0000122786 and
max0.000610024. A narrowly gated physical-only paired-F16 operator may now be
tested without changing vendor traits or unrelated models. No such path is
promoted, and decoded quality remains unproven.

## Explicit paired-F16 dot experiment

Source `0b76ce1365b57360616e5fce44784db1665bbf10` calls the existing original
paired-F16 dot explicitly, without changing vendor dispatch. Both backends pass
symbol/overflow controls, and all ten corresponding N4/N16 helper stage hashes
are identical. The [report](../tests/results/qwen35-ocr/lightonocr3-4b-attention-primitives/paired-v1/paired-attention.json)
SHA256 is `04b194e16e7243dbc22d3db74c9d86a0d42e8d4918b528064c0afdf50b369904`;
elapsed118.65 seconds. Ten selected vectors were independently authenticated;
ten N16 metrics reproduce exactly for max/byte equality and within one binary64
ULP for RMS.

The helper is still not byte-identical to the original source graph. N16 QK
RMS is0.00000206412/max0.0000610352; context RMS0.0000118884/max0.000530243;
gated context RMS0.000000691884/max0.0000839084. N4 gated RMS is
0.000000000608240/max0.0000000149012. Exactly one N16 probability crosses an
F16 rounding boundary (flatindex23300, delta0.000244140625), so the context
residual cannot yet be attributed independently to PV. Next controlled checks
feed exact retained source probabilities into PV and isolate actual QK graph
dispatch. This is a private-backend-ABI operator experiment, not a production
implementation or decoded-quality proof; no fallback or vendor mutation occurred.

## Exact source-P PV and QK dispatch isolation

The bounded follow-up at source `b46bd2b99` completed in164.22 seconds.
Regenerated upstream ON reproduces all ten frozen source graph stages bytewise
before interpreting other arms; all eight composed/observed pairs pass. The
[report](../tests/results/qwen35-ocr/lightonocr3-4b-attention-primitives/dispatch-v1/attention-dispatch.json)
SHA256 is `d5b7ad89bc6f175e3ef5162bb6ab5b036a3e8acec261c83b191928af05638677`.
Independent readback authenticates ten selected vectors (2,621,440 bytes),
recomputes eight metrics (exact maximum errors, RMS within one binary64 ULP),
and checks the eight observer hash pairs.

Feeding the exact retained source probabilities into paired-F16 PV reduces N16
context RMS from0.0000118884 to0.0000000234302 (max0.000000476837), and gated
RMS from0.000000691884 to0.00000000216636 (max0.000000119209). Most previous
context error therefore propagated from QK/softmax through a probability rounding
boundary; an independent large PV defect is not supported.

Original paired-F16 raw QK is byte-identical to the upstream graph with LLAMAFILE
disabled at both N4 and N16. Actual source LLAMAFILE ON retains QK RMS
0.00000206412/max0.0000610352. The remaining small QK residual is attributable
to that dispatch/reduction choice on these exact inputs. OFF is a diagnostic
profile and must never replace the actual ON reference or its qualification
settings. Native ON/OFF both retain the intentional F16-by-F32 operand behavior.

This closes primitive attribution only. A physical-only implementation can now
be investigated without changing global vendor traits, but must preserve
measured residuals, fail closed on unsupported backends and pass decoded gates
before any promotion. No model forward, production math or shipping policy
changed in this experiment.
