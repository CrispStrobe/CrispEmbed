# Frozen receipt prefill attribution

This hosted diagnostic authenticates the completed physical RoPE fixed panel
and its original receipt reference, then observes the pinned quantized source
for prefill plus one sampled token in OFF/ON observer arms. Both source arms
must reproduce the retained source first score, sample probabilities, vision
features and positions exactly. No native model inference, full source decode,
original BF16 model or full F32 text model is permitted.

The analysis reads the already retained native RoPEON vectors. It requires all
256 source/native embedding and position rows to match byte for byte, compares
hidden boundaries in actual graph execution order, and records RMS/max errors
alongside source/native magnitudes. The source's final layer is cropped to the
last token and is compared only there. Detailed source stages for GDN layers
0–2 and full-attention layer3 are retained remotely; available native first-text
chunk layer0 stages are compared directly. This does not assume attention,
RoPE or batching is the remaining cause. A differing boundary determines the
next bounded same-input component experiment.

All original fixed-panel floors and observed BF16 regressions remain unchanged.
The tools carry no runtime gate/default, release or application change. Raw
vectors and decoded source responses stay remote and private.

Small controls: `python3 tools/receipt-prefill-trace/test_controls.py`.
