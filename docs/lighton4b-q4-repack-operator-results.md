# Physical Q4 CPU_REPACK operator diagnostic

Source `3b41475c88f570cf899bac12ae6c4f80e4eb3400`, pinned native GGML
`3a27b6c665b191cb230be367db5c52d81dd3e2ac` and upstream
`6c01da6278ab6c69ed67d9124806b50d2b299ba7` completed the bounded CPU
operator worker in 96.740 seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-source-repack-operator/v1/repack-operator.json)
SHA256 is `4dc28c17717082a36e9f2cad3729a31a83a97c3234b07d3c00002232744a66f5`.

All five N16 borrowed projections reproduce actual retained native captures
bytewise before interpretation. Mapped-buffer attachment alone leaves every
output unchanged. All twelve Q4 preferred-buffer comparisons (four matrices,
N1/N4/N16) match across both pinned backends and whole-row transports. Repacking
changes Q4 outputs slightly versus raw quantized buffers: maximum absolute
difference 6.12065196e-6, RMS at most 1.01446614e-6. The Q6 negative control is
exact across backends for N1/N4; N16 retains RMS 2.12041319e-7 and maximum
4.76837158e-6. That separate residual is not attributed to repacking.

The actual source loader selects CPU_REPACK/extra for the measured Q4 matrices
and CPU_Mapped/no-extra for Q6. Whole selected matrices (36,464,640 bytes) are
authenticated against immutable physical-source offsets and transport hashes.
This authenticates selected ranges, not a newly downloaded full model. Inputs
are actual captured native norm/postnorm states, not actual upstream hidden
states. N4 matches text-prefix geometry; N1/N16 are component replays. The out
projection and complete graph are not closed by this panel.

No full-model inference, decoded comparison, release or quality promotion
follows from this operator result. A separate default-off physical Q4 buffer
cache needs lifecycle/budget/cache-mode controls, actual runtime source-fed
equality and same-feature decoded OFF/ON evidence before integration. Global
vendor traits and the public release fence stay unchanged.
