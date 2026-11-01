# LaMa first-layer attribution

**The retained full-mask numerical failure remains unresolved.** A bounded
follow-up reproduced the first native/original difference and separated the
first convolution from batch normalization. It did not change production
arithmetic or relax any acceptance limit.

The [complete attribution record](../tests/results/inpainting/lama-first-layer-attribution-2026-11-01.json)
contains all 14 boundary comparisons, source/model/library/executable hashes,
original and native payload identities, command receipts and memory observations.
It supplements the [earlier full-model record](lama-cpu-validation.md).

## Why this boundary

The original full-mask run first exceeds the absolute stage-error limit at
`generator.model.5.conv2.ffc.convg2g.fu.rfft`. Earlier source-fed tests showed
that all three actual production FFT implementations pass their component
limit when given the identical original real input. The first recorded native
feature, `generator.model.1.local`, already differs slightly from the original.
The new experiment therefore isolates the first feature-producing operations.

The original side loads the pinned original source and checkpoint, verifies
the frozen 128 × 128 full-mask input, and executes only the original reflection
padding and first `FFC_BN_ACT` module. Hooks copy the raw convolution and batch
normalization outputs before the original in-place ReLU can change them. The
ordinary and hooked outputs are byte-identical, and both match the earlier
frozen original first-layer output. No full generator is executed.

The native side feeds that exact original padded tensor to the actual eager
`Engine` and persistent `Builder` helpers. The graph experiment runs with the
CPU convolution layout optimization off and on. A separate batch-normalization
input receives the original raw convolution, removing upstream convolution
error from that comparison. The experiment retains the tensors, not only
aggregate metrics; every original payload is checked again after native replay.

## Observed differences

All entries below are maximum absolute errors against the corresponding
original boundary.

| Boundary | Eager native | Portable graph | Optimized CPU graph |
| --- | ---: | ---: | ---: |
| Raw first convolution | 3.81469727e-6 | 3.81469727e-6 | 3.81469727e-6 |
| Batch normalization after native convolution | 1.19209290e-6 | 1.43051147e-6 | 1.43051147e-6 |
| ReLU after native convolution and normalization | 5.66244125e-7 | 5.36441803e-7 | 5.36441803e-7 |
| Batch normalization fed the original raw convolution | 2.38418579e-7 | 2.98023224e-7 | 2.98023224e-7 |

The three raw native convolution outputs have the same SHA-256 digest. Each
variant's ReLU output also has exactly the digest of its retained full-model
`generator.model.1.local.f32` tensor. This establishes that the bounded run
reproduces the earlier first-layer boundary, including its existing drift.

The source-fed normalization test establishes a smaller additional arithmetic
difference after the convolution. Two diagnostic F32 affine evaluation orders,
using reciprocal-derived scaling with either separate multiply/add or explicit
FMA, still differed from the original normalization. Their maximum errors were
5.36441803e-7 and 4.76837158e-7 respectively. These diagnostic formulas were not
installed in production; they do not establish a useful replacement.

All 14 comparisons satisfy the unchanged intermediate budget: maximum absolute
error ≤ 0.002, relative L2 ≤ 0.0003 and cosine ≥ 0.99999. These are isolated
boundary results. They do not cancel the later full-mask failures, establish
the relative contribution of each early difference to those failures, or prove
full-model parity.

## Scope and next decision

This result supports keeping the existing FFT implementation and the numerical
gates. The earliest measured mismatch is already present in convolution, with
an additional smaller normalization difference. A production normalization
rewrite is not justified by this probe. A later investigation would need an
explicit same-input test of the convolution reduction order or a measured
performance change with the full retained quality gates.

The original partial capture, native probe build and native helper execution
took approximately 23.29, 14.26 and 0.36 seconds respectively. These are command
diagnostics, not inference benchmarks or speedup claims. Their observed child
peak RSS values were 636,376, 421,808 and 46,984 KiB. The native probe loads
mapped weights and executes only the first layer, so its memory observation
cannot describe a complete LaMa process. No GPU was exercised.
