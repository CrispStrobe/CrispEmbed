# Live-input WASM native-order norm candidate v1

This default-off candidate follows the independently reviewed local
[norm-risk run40834745535](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834745535)
at`e7f3ab64dda2fdb42c8305637a258ce4b10bdef4`. Its38 small files rehash exactly:
30 old byte anchors,12 native-order norm recoveries and56 actual Ref comparisons
pass. The computed matching-input attention norm/scaled/affine are exact across
native1/native2/WASM. Matching-input FC1 error is1.430511474609375e-6; the naturally
different residual still produces FC1 error0.0005726814270019531. One Q8 scale
changes; signed activation bytes remain identical. That is local support, with
no whole-model/default/release qualification. Gamma/beta already remain F32;
this is a reduction-order candidate, not a critical-weight storage repair.

`CRISPEMBED_WASM_NATIVE_ORDER_NORM` uses `core_env::on` and snapshots once per
context. It is active only in Emscripten for BERT384/six layers/twelve heads/
1536 FFN/post-LN/epsilon1e-12. Unsupported profiles, native builds, unset, empty
and literal zero keep the existing `ggml_norm` graph nodes. Existing alternatives
and all numerical targets remain. Embedding and the two per-layer post-LN sites
use the candidate, for13 norm nodes in the admitted model. Pre-LN/final-norm
paths are unchanged. No public API, model bytes, quantizer, ggml source, Q8
default or runtime release version changes.

The connected `ggml_map_custom1` F32 node reads live source rows and writes its
own destination. It accepts no saved norm/output/workspace. Its immutable epsilon
userdata belongs to the model context through the graph lifetime. Rows partition
by actual `ith/nth`, with outer dimensions/strides preserved. The portable code
uses the proven a06 AVX2 order: sequential F64 sum rounded to F32 mean, F32
center/square, eight-value F32 reduction tree, F64 chunk accumulation, then F32
variance/epsilon/sqrt/reciprocal/scale. Contraction is explicitly off in the new
translation unit. Existing gamma/beta graph operations and stored weight types
remain unchanged. Invalid one-row input is checked before writing; invalid graph
arithmetic remains visibly nonfinite rather than manufacturing a finite result.

The source contract binds the exact assembled base plus only four declared
runtime files. All other source/build/binding Git objects and every executed
helper/reference are protected. Historical source guards are never retargeted
or monkeypatched. Reused low-level archive/parsing/overlay functions keep their
old contracts. A new standalone [workflow](../.github/workflows/wasm-native-order-norm-candidate-v1.yml)
builds with two workers on hosted CPU only. No framework or original model run
is needed; model payloads and traces are never downloaded locally. The retained
operator archive includes old coefficient bytes, authenticated without using
them as inputs to the norm primitive; whole-model fixture access follows the
component barrier.

The ordered acceptance barriers are:

1. Authenticate the exact source and actual GHA service, then retain the old
   norm-risk report/raw input hashes. Compile the new connected graph on native
   and the historical script-profile pthread WASM. Geometry, nonfinite, zero, threads and recovery
   controls run. All 12 freshly computed norm outputs must recover the same native
   bytes before whole-model fixture access. Actual Ref positive/numeric/wrong-shape controls
   must exit0/1/2.
2. Authenticate the completed negative580 A/B report and all retained array
   hashes, its old-native ten-case freeze, both native first-sentence stage files,
   and the exact immutable MiniLM IQ4 fixture. No frozen target is replaced.
3. Build the actual candidate native library and a disposable test-overlay WASM
   module. Keep the old580 full-model `--no-simd` script+pthreads profile and
   compare actual old/current flags per translation unit. Despite that script
   option, pinned a06 CMake always adds `-msimd128` to its CPU backend target.
   The frontend/new helper have SIMD off; GGML has SIMD on. Record both rather
   than claiming the whole module is scalar. The local risk helper's SIMD128
   profile remains separate; it does not authorize a silent frontend change.
4. Require current native1/2 to retain all20 old final-vector byte anchors and
   ordinary/observer/repeat identity. Native always ignores the new gate.
5. Hold existing `CRISPEMBED_WASM_Q8_0_ROUND=1` for norm unset/empty/zero/one.
   Before norm one executes, all ten norm-off outputs and first-sentence stages
   must reproduce the old580 Q8-gate-one bytes. Additionally run Q8-rounding-zero
   plus norm-zero against all ten old580 legacy-gate-zero bytes. All previous
   whole-model/operator/F32-cast negatives remain recorded. Norm one is a
   composed rounding-one plus norm candidate, not a norm-only change from
   legacy rounding zero. Neither gate's default changes.
6. Only after the off proof, run norm one for all ten cases. Compare twenty old
   native final targets plus16 first-sentence stage targets (eight stages at both
   native thread counts) through actual `crispembed_diff::Ref` and independent
   row metrics. Gates stay absolute0.002/minimum row cosine0.99999. Retain RMS,
   mean/absolute errors, magnitudes, L2, hashes, every failed row/stage, actual
   IDs, pooling-buffer bytes and ordinary/observer/repeat results. Interleave
   contexts0/1/0 after changing the environment to one; old contexts must retain
   their own behavior. Numeric failures are collected before nonzero exit.

Full-model positivity would qualify only this owned ten-case numerical/length
panel under its exact source/build/model. It would not establish representative
task accuracy, SIMD128 parity, other-model coverage, default promotion, release
assets or app adoption. The old eight-stage coverage remains first-sentence
only. No saved output, precomputed activation workspace or reference buffer is
injected into inference. Production includes no test control exports; environment
and pooling observers exist only in the disposable hosted overlay. All small
reports/failures and raw arrays/modules are retained remotely for90 days.

The browser retains all 1,536 returned bytes before freeing every ordinary,
observed, repeat and interleaved result. Files plus base64/SHA receipts carry
byte identity; JSON numeric vectors are display data and preserve only logical
values (JSON erases negative zero). Off anchors use authenticated old actual
pooling-result buffers, while old JSON-derived hashes remain separately visible.
