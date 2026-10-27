# Prepared MoEdern1.5M dual-reference native experiment

This is a separately versioned CPU/a06 experiment, based on claimed main
`cd6aad7f6dd3eef487031f745bd4c5a978346d50`. The model implementation, header,
standalone probe and controls are byte-identical to authored preparation
[`79e350a0`](https://github.com/CrispStrobe/CrispEmbed/tree/79e350a096db831543dca711fd6393e61d63fa79).
Model math is unchanged. The standalone v2 CMake and actual Ref schema use
separate names and explicitly select ggml
`a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. The earlier unexecuted v1/890
contract remains blocked and is not widened, executed, rewritten or qualified.
The new source guard closes the complete linked runtime/build/binding paths
against this declared main and permits only the two new model source files.
No root library/CMake, C ABI, public tokenizer, registry or alias is changed.

The [v2 protocol](moedern-colbert-native-contract-v2.json) preserves two distinct
reference roles. Successful original
[40412164041](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40412164041)
at `790f7b1d0372c12d1807dfb2fc7fc3af20a3c445`, receipt
`b50ca1369f454d61b8c95583d8eb77a847c47ede29e52ed66b575784f4e4660f`,
is the primary frozen target: all 250 retained arrays, all ten ordinary complete
and selected vector outputs, and all three MaxSim scores remain unchanged.
The original F32 GGUF `9f9792f0...` is authenticated from its exact retained
conversion receipt; no new model conversion or original recapture occurs.

Corrected observer
[40423259251](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40423259251)
at `8b153739503d6d817c7271a5ce319358eadaaab3` remains a failed producer with
`completed:false` and exact rejection `Prior original continuity differs: scores`.
Its 21 newly captured pipeline leaves are accepted only as separately reviewed
supplemental measurements. They replace none of the old 250 targets. These
leaves contain six F32 Dense/Normalize observations and 15 integer/boolean
ID, mask and query-expansion snapshots. All three pipeline mask semantics stay
distinct: supplied mask before the masking module, scoring mask afterward,
and independent effective backbone attention. Repeated labels keep their
literal names and occurrences; original batch axes are never removed.

This boundary follows the reviewed first-attention diagnosis
[40434080299](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40434080299)
at `6fccd8635f8489d2eff2d47814a75cf77df6d584`. Exact original-source input,
projection, Q/K/V and mask values reproduce the corrected observer's first SDPA
and attention-module outputs under current CPU flash. Old SDPA differs by at
most `1.1920928955078125e-7`; forced math differs from corrected-observer SDPA
by at most `4.76837158203125e-7`. Actual Ref and independent metrics pass the
unchanged operator gates. Current strides, hardware and actual profiler kernels
are recorded, while historical hardware/strides/kernel identity remains unknown.
This does not establish an exact cause for every downstream difference or make
old byte continuity pass. No backend change is adopted by this native experiment.

Preparation authenticates exact successful/failed service outcomes, both old
and corrected raw/review archives, and the immutable reviewed SDPA report.
It rehashes every original array, enforces exact checkpoint/loaded-weight/source,
ID/mask/case/shape equality and requires exactly 250 shared plus 21 added keys.
Only the old stdlib schema, source-contract JSON and owned-panel JSON are
materialized from authenticated `790` Git blobs into a task-owned source tree;
historical inference workers and their completion guards are never executed.
The installed packages are only NumPy2.2.6 and GGUF0.17.1, with no Torch,
Transformers, tokenizer, new corpus or weights download. All source weights,
the 32-tensor F32 GGUF and raw reference arrays come from retained artifacts.

Before native forward, every one of the shared 250 rows is paired as additional
original reproducibility evidence: 198 F32 rows use actual Ref plus independent
NumPy errors/magnitudes; 52 integer/boolean rows require exact values. All
three score drifts remain reported against the old scores. These comparisons
must stay within the existing numeric envelope; they do not accept byte
continuity or produce new completed gold.

The native coverage declaration is explicit and frozen before outputs. The
unchanged probe cannot emit all 250 original module-hook observations. Exactly
112 primary arrays map to native checks: 108 F32 stages/vectors and four exact
router-index arrays. The remaining 138 historical inputs, aliases and activation
or bookkeeping snapshots are individually listed with a reason in the protocol.
They remain retained and checked original-to-original, without a native-parity
claim. All 55 native stages must exist for every case, including four observations
per full-capture case for which no independent original target exists: attention
probabilities and routed-only sums. Native-only observations stay finite and
are retained without a comparison claim. Eight non-full-capture cases have
primary final-vector targets but no original intermediate targets.

All ten supplied IDs/masks, effective attention and scoring masks are exact.
Both fully captured cases retain the existing mapped primary backbone stages;
their three Dense/mask/Normalize F32 leaves and all 15 bookkeeping leaves are
checked separately against supplemental observations. Every complete ordinary
and selected vector and all three native/independent MaxSim scores compare to
the OLD successful primary producer. Normalized observer bytes must equal
ordinary complete-vector bytes, without reshaping original arrays. The unchanged
probe also requires ordinary/observer/repeat identity, gate/failure recovery and
two-context ownership.

Every mapped F32 comparison uses actual `crispembed_diff::Ref` and independent
NumPy metrics under the fixed absolute `0.002` and row-cosine `0.99999` budgets;
MaxSim remains `0.002`. Exact integer checks are never cast through a float
comparator. Lossless GGUF adapters preserve full shape and float bits, use
contiguous row axis zero, and keep primary/supplemental namespaces separate.
Metrics include RMS/mean/signed error, original/native L2 and min/max magnitudes.
The first differing and first failing records preserve their reference kind.
Reporting agreement at relative `2e-7` cannot waive either numerical gate.
Compiled Ref equal/changed-row controls precede the panel. Numerical rejection
retains all ten cases and three scores before final nonzero exit; a structural
case failure is recorded while remaining cases continue.

The [hosted CPU workflow](../.github/workflows/moedern-native-v2.yml) is registered
as metadata phase `moedern-native-v2`. It requires 2GiB available RAM/4GiB free
disk, uses two build workers and a 60-minute bound, and retains all outcomes,
large vectors/adapters and exact executables remotely for 90 days. Source
preparation runs only lightweight stdlib controls locally and dispatches nothing.
Parent source review is required before execution. Public Unigram/role APIs,
27M/115M, quantization, representative retrieval, runtime defaults, release and
application adoption remain separate after actual 1.5M parity.
