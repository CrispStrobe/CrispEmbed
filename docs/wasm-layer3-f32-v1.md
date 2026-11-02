# Retained layer3 F32 weightcast risk protocol

This separately versioned, test-only protocol follows the
[workspace causal result](wasm-q8k-cross-result-v1.md). It tests a possible
correction before introducing any production gate. No complete encoder, original
rerun, new target, default flip or model qualification is included.

The [closed declaration](wasm-layer3-f32-v1.json) admits exactly the saved
IQ4_XS `blk.3.ffn_down.weight` coefficients and two natural twelve-token GELU
inputs from the frozen layer3 producer. The reviewed causal service and complete
review/raw ZIP identities are checked before reading members; every listed raw
member is authenticated. The older trace, model/source and compiler profiles
are independently authenticated through their unchanged provider. Only this
hosted job downloads those retained payloads.

Six ordinary quantized outputs (native1/2 and browser, each on both saved inputs)
must reproduce the old output SHA256 values before any F32 candidate arm runs.
An actual `ggml_cast` then derives F32 coefficients from the unchanged IQ4_XS
weight. The native and browser derived2359296-byte coefficient arrays must be
identical to an independent literal IQ4_XS decoder before comparisons. The cast
operand and matmul input are F32; pinned GGML's F32 CPU trait selects an F32 dot
operand, removing Q8_K activation conversion for this one test projection. The
model payload and all original profile choices remain unchanged. Actual native
LLAMAFILE-on and WASM LLAMAFILE-off/SIMD128/pthread/PIC profiles must match their
frozen profiles; generated commands and mapped provider hashes are retained.

Ordinary, coefficient-observed and repeated outputs must be byte-identical.
Native1/2 and browser execute invalid mode/extent, nonfinite input/weight scale,
unchanged output and coefficient buffers on failure, and zero-input recovery
controls. Both actual derived coefficient buffers and all immutable inputs remain
separate test artifacts. No reference probe or callback is added to production.

All25 predeclared comparisons use actual `crispembed-diff::Ref` and independent
NumPy maximum/mean/RMS/L2/magnitudes/minimum-row-cosine metrics:

- Two crossbackend F32 comparisons and two native thread-count comparisons.
- Six same-input comparisons against that runtime's unchanged quantized output.
- Twelve comparisons against the unchanged old native1/2 original-input outputs:
  both natural inputs, for each of native1/2 and browser.
- Three within-runtime comparisons between the two F32 natural-input outputs.

The conjunction of maximum absolute error0.002 and minimum row cosine0.99999
remains mandatory. All numeric rejections are collected before the job rejects;
structural failures are distinct. An independent F64 dot on the literal decoded
coefficients is a labelled diagnostic, never a replacement golden reference.
A native/browser pair agreeing with each other cannot override changed old
native targets. The cast changes approximate native arithmetic, so such a
failure is a real risk rather than a reason to loosen gates.

Even a supported projection result does not qualify a production correction.
Any later context-scoped default-off experiment must still preserve unset,
empty and zero legacy bytes, and pass all ten old native1/2 final-vector gates
and the original sentence's eight stage gates, with ordinary/observer/repeat and
interleaved context controls. Other projections can still amplify natural drift.

The standalone workflow is bounded to60 minutes and two build workers, with
4GiB available RAM/3GiB free disk/two CPUs admitted before transfers. Small
reports/partial failures and separate raw buffers/modules are retained for90 days.
Source preparation runs only standard-library controls; no execution or
qualification is claimed before the actual hosted result is reviewed.
