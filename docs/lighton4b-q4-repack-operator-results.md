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

## Initial runtime-cache controls

Source `26763167f` compiled the gated runtime and passed the supported
lifecycle/budget/cache-mode unit (57 checks, 2.034 seconds) and the main-library
closed-stub unit (2 checks). The worker then failed at reference-only adapter
compilation: the adapter omitted `qwen35_math.h`, which declares `project_cpu`.
[Initial receipt and compiler log](../tests/results/qwen35-ocr/lightonocr3-4b-q4-repack-cache/v1-adapter-compile-negative/index.json)
preserve the 443.392-second partial result. No tensor ranges were downloaded,
no source-fed projections ran, and remaining compatibility controls were not
executed. This is a harness compile negative, not an arithmetic failure.

Independently reviewed source `5f6dfb9f0` adds the missing direct include and
compiles the adapter before the expensive main-library build; all runtime
bytes remain unchanged. A bounded retry is pending. No quality or release
claim follows from the initial unit passes.

## Complete runtime-cache controls

Retry source `5f6dfb9f09b49002201c7702ff71a5e345294af2` completed in491.644s.
[Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q4-repack-cache/v2/runtime-repack-controls.json)
SHA256 `c7130c6d7b54e6859439bbf87c8e54880c30bd5d2db33c3571ded179e6711cf5`
records all30 actual-input runtime OFF/ON outputs byte-identical to the sealed
raw/preferred operator outputs. Supported lifecycle57, F16-attention255297,
native-ops142, HF-layout39, Q6-head84 (65,792 independent scalar values), main
SDK5 and main closed-stub2 controls pass. All eight downloaded unit/profile
files match the child receipt hashes; runtime and sealed native operator
builds have equal GGML Boolean options. This is supported execution, not an unsupported-path skip.

The child artifact inventory includes one stale outer-bootstrap JSON entry
because the outer process finalizes after the child. That row is excluded from
immutable artifact claims. The independently completed final bootstrap and
its actual hash are preserved separately in the evidence index. This metadata
limitation does not change the closed unit outputs or30 measured comparisons.

No full model, real Emscripten rebuild or decoded comparison ran in this worker.
The main unsupported stub is tested; actual new-code WASM linkage is not newly
qualified. Next is one bounded source-feature table512 OFF/ON model diagnostic,
requiring every old282 raw row and38 stages before evaluating the alternative.
All existing observed/composed/held-out release gates remain in force.
