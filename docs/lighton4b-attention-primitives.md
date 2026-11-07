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
