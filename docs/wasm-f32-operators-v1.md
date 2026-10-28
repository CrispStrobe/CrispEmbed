# Retained-input WASM F32 operator attribution v1

Prepared source only: there has been no execution of this new protocol. It makes
no runtime/default/release qualification and changes no production source. The
remaining browser failure and the original0.002/0.99999 gates are preserved.

The reviewed [layer-3 producer40424064413](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40424064413)
at `dfd7bb4f35d3460f1ea67f6f5ce35cc408e2e6f6` retained23 subnodes for one
12-token sentence. Its runtime was
[`580e3ae3be3671ddce5e945b0725d0583ec34110`](https://github.com/CrispStrobe/CrispEmbed/tree/580e3ae3be3671ddce5e945b0725d0583ec34110)
and ggml was
[`a06fc0bb46b73cc10011fc90ae4054aee8438aeb`](https://github.com/CrispStrobe/ggml/tree/a06fc0bb46b73cc10011fc90ae4054aee8438aeb).
The report SHA256 is
`da8767490c953a6dbf544fc3b1a22b8aabb058be3f46dd7527c5cd3ddde1782d`.
Its six quantized projections already replayed both input axes with exact
activation bytes/scales. This lane does not repeat those projection experiments.

The original layer-3 raw artifact12702610607 is24,900,090 bytes, SHA256
`1444edaf1e3751ee7f0c6d5a05faef0af16f476b2c8fce4bd7785ec950e52721`.
The small review artifact12702046203 is1,031,136 bytes, SHA256
`4847ab1b9f2f10680e9647bc19fc462bbbea212379756fffd896bba9cd122e05`.
Hosted preparation authenticates service source/attempt, ZIP SHA and every old
raw member plus binaries. Only two explicitly declared SONAME aliases may be
additional members, and each must have the exact authenticated library bytes.
Old reports, protocols and negative judgments remain unchanged.

The retained native dynamic-symbol listing exposes no GGML operator API. This
lane therefore builds standalone wrappers against the exact frozen ggml source,
using Emscripten6.0.2 and the original native/WASM backend settings. It compares
actual `vec.cpp`, `ops.cpp`, `ggml-cpu.c` and `ggml-cpu.cpp` compiler definitions,
ISA and arithmetic options with the retained producer records. The native AVX2
profile uses LLAMAFILE; the browser backend uses SIMD128 and pthreads, with
LLAMAFILE off. Compiler/profile disagreement blocks interpretation. The new
standalone C ABI is test-only and is never linked into a production archive.

Sixteen boundaries cover Q/K/V bias adds, actual flash attention, attention-output
bias and residual, attention normalization/scale/bias, FC1 bias, GELU-erf, FC2
bias, and FFN residual/normalization/scale/bias. Flash preserves the original
`[32,12 heads,12 tokens,1]` reshape, `permute(0,2,1,3)`, F32 inputs, scale
`1/sqrt(32)`, and absent mask/RoPE. Norm uses the original ordered metadata keys
or the source default`1e-12f`, recorded explicitly. A different epsilon rejects.

All inputs come from native, WASM gate0 and WASM gate1 retained subnodes. Bias and
scale vectors come literally from the SHA-authenticated original MiniLM GGUF,
`77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2`,
through the exact source580 ordered name aliases. Only F32 coefficients with the
expected dimensions are accepted. No model forward, tokenizer replay, reference
recapture or new inference weights are involved. The model is accessed only on
hosted CI to read these frozen coefficients; no payload is needed on a developer
host.

For each of the three retained input axes, native threads1/2 and the real
cross-origin-isolated browser module execute every operation and repeat it.
Inputs must remain unchanged. Positive norm, invalid-shape, nonfinite and
failure-recovery controls exercise the actual C ABI. Browser outputs are saved
incrementally so a later error retains earlier results. The module must use
SharedArrayBuffer with COOP/COEP isolation; a Node numerical substitute cannot
satisfy this proof.

Before any cross-backend metric is interpreted,64 own-input/output byte anchors
must match the old full graph: native1/native2 on native inputs, and WASM on both
gate0/gate1 inputs. If this fails, the experiment rejects attribution and retains
its outputs as a source/toolchain/layout mismatch. This prevents a compiler or
standalone graph difference from being mislabeled an original operator cause.

After the barrier,48 same-input comparisons use actual
`crispembed_diff::Ref`, lossless F32 adapters and independent NumPy metrics.
Real Ref positive/numerical-rejection/wrong-shape controls run first. Reports
retain maximum/mean/RMS absolute errors, signed mean error, signed/absolute
ranges, both L2 norms and minimum row cosine. Numerical failures remain visible
under the original0.002/0.99999 budgets; attribution completion does not qualify
the failed whole model or waive its vector gates.

AVX2 grouped-F32 variance reduction versus scalar-F64 accumulation, native FMA
versus WASM multiply/add, and GELU/libm differences are hypotheses. No emulation
or production candidate is activated here. A distinct test-only norm-tree
experiment may be proposed only after actual same-input norm anchors and
measurements justify it. Even an exact component replay would not establish
composed model causality or final-quality parity.

The reusable [workflow](../.github/workflows/wasm-f32-operators-v1.yml) is prepared
for the existing metadata dispatcher; registration and dispatch are coordinated
separately. It builds GGML only with two workers on hosted Linux, retains review
and raw artifacts for90 days, and downloads no model framework. All source,
operator inputs, compiler commands, commands/log hashes and new raw outputs are
retained remotely. Download only needed small reports for review.

## Genuine hosted CPU fallback

`tools/kaggle/wasm-f32-operators/kernel.py` can replay the same retained
operators when Actions is queued. Render one immutable source and attach the
authorized artifact-read credential privately. The bootstrap checks CPU-only
resources, reports its actual provider and script SHA, and never sets GitHub
service variables. Frozen original service/artifact/member checks, compiler
profiles, 64 own byte anchors and 48 cross-backend comparisons are unchanged.
No original model forward is repeated.

The fallback pins SDK 6.0.2 at
`ca38f487f28b7c3c16f8f70cd0e012099ac4b7e2`, its dependency manifest, official
Node 22.16.0 archive SHA, Playwright 1.56.0 and NumPy 2.2.6. Native libraries,
raw packets, browser module and scalar reports remain in hosted outputs.
Prepare and review source first; check actual session capacity and cancel a
superseded queued task before dispatch, so the same panel is not run twice.
A changed host that cannot reproduce an own-byte anchor is a rejection.

## Initial standalone build rejection

The first [actual standalone run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40749050064)
at source `52739ca5838148adf8f27f8df816ad61b7097eb2` authenticated all retained
inputs and coefficients and completed the native build. WASM compilation
completed, then linking rejected an undefined `quantize_row_q8_0_wasm_nearest`.
The Emscripten configuration reported processor `x86`, so frozen GGML selected
generic CPU sources and omitted `ggml-cpu/arch/wasm/quants.c`. No native operator,
browser, own-byte anchor or cross-backend metric ran. This is build evidence,
not numerical evidence.

[The initial receipt](../tests/results/packaging/wasm-f32-operators-initial-build.json),
[authenticated preparation](../tests/results/packaging/wasm-f32-operators-initial-prepared.json),
[configure log](../tests/results/packaging/wasm-f32-operators-initial-wasm-configure.log)
and [link log](../tests/results/packaging/wasm-f32-operators-initial-wasm-build.log)
preserve the initial rejection and immutable small/raw artifact identities.
The raw artifact remains remote.

The bounded correction restores exactly the frozen `build-embed-wasm.sh`
`-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm` toolchain selector. It verifies actual
SIMD128/pthread compile records for the four CPU/operator sources and the
architecture quantizer, rejecting omission, duplication, generic fallback or
arithmetic-flag drift from the original compiler records. Native configuration,
frozen GGML, all 16 operators, 64 own-byte anchors, 48 same-input comparisons
and 0.002/0.99999 gates remain unchanged. A new hosted build and all mandatory
anchors are still required before any numerical conclusion.
