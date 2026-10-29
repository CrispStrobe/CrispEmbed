# Retained natural-input Q8_K workspace cross-admission v1

This separately versioned test protocol follows the [64-anchor/48-F32-operator result](wasm-f32-operators-result-v1.md). It preserves the old whole-model rejection and the12 successful identical-input quantized-projection replays. It does not run an encoder, recapture an original, change production arithmetic, add an F32-weight branch or promote rounding. Source preparation is not an executed causal result.

The [machine-readable contract](wasm-q8k-cross-v1.json) binds original trace40424064413, its exact model/source580/a06 and the reviewed operator result40771950208. Both retained natural GELU input arrays, their actual selected-plan Q8_K workspace bytes/scales/sums, the exact IQ4_XS `blk.3.ffn_down.weight` payload and both runtimes' saved outputs are authenticated before a new build. Weights remain type23 with physical `[1536,384]` and activations use type15 `[1536,12]`; output is `[384,12]`, logically twelve rows of384. The original twelve token IDs and full-graph results are retained dependencies, not inputs to a new model forward.

The standalone CPU graph first runs the ordinary F32-input projection. The identical earlier test-only kernel capture is inserted after the real conversion barrier in a disposable frozen GGML checkout. Its generated workspace and output must reproduce the retained bytes exactly in native1/2 and browser. Production source and archives are untouched. Actual old/new arithmetic compiler flags, including WASM SIMD128/pthreads and PIC, must agree; all emitted native provider/module hashes are retained and rechecked.

Pinned GGML explicitly supports matmul's second input as either F32 or the weight's vec-dot type. Its conversion branch executes only when those types differ: [CPU support](https://github.com/CrispStrobe/ggml/blob/a06fc0bb46b73cc10011fc90ae4054aee8438aeb/src/ggml-cpu/ggml-cpu.cpp#L462), [actual conversion/dot path](https://github.com/CrispStrobe/ggml/blob/a06fc0bb46b73cc10011fc90ae4054aee8438aeb/src/ggml-cpu/ggml-cpu.c#L1273). A second graph admits the saved Q8_K tensor directly. This can change dispatch, so direct-admission outputs must also reproduce every retained same-input projection output **byte for byte**. All six normal/direct arms execute in native1/2/browser first. Eighteen original byte anchors are required before either hybrid intervention executes, and are rechecked before comparison. Failure of one anchor rejects the diagnostic, with no tolerance exception.

The protocol freezes two ordinary F32 arms and six workspace arms before outputs:

| Arm | Retained F32 metadata input | Arithmetic Q8_K operand |
|---|---|---|
| `f32_native` | Native GELU | Actual regenerated native workspace |
| `f32_candidate` | Browser-candidate GELU | Actual regenerated candidate workspace |
| `native_native` | Native | Saved native |
| `native_candidate` | Native | Saved candidate |
| `candidate_native` | Candidate | Saved native |
| `candidate_candidate` | Candidate | Saved candidate |
| `native_scale_candidate_integers` | Native | Native F32 scales, candidate signed bytes and matching sums |
| `candidate_scale_native_integers` | Native | Candidate F32 scales, native signed bytes and matching sums |

In direct-admission arms the retained F32 input is validated attribution metadata; it is **not** an additional arithmetic operand. The input/workspace crossings measure whether the admitted workspace determines the output. Hybrid blocks keep each256 signed bytes and their16 sum fields together, retain scale bytes losslessly, validate finite scales and reject inconsistent sums. They are interventions, not valid original activations or candidate model outputs.

All44 predeclared comparisons retain actual `crispembed-diff::Ref` and independent F32 raw metrics, absolute/RMS/norm/magnitudes and minimum row cosine. Eight compare native/browser on the identical arm;36 compare six workspace arms to each of the two natural baselines in native1/2/browser. Numeric rejection code1 is collected across all interventions; structural code2 fails. Unchanged0.002/0.99999 flags are reported, including expected cross-natural failures; their existence is not waived or converted into model qualification. Both backends execute wrong extents, nonfinite inputs/scales, inconsistent sums, unchanged output on failure and recovery controls. Repeats and native1/2 must be byte identical.

The standalone hosted job is bounded to60 minutes/two build workers, with4GiB available RAM/3GiB disk/two CPUs required before transfer. Only the fixed old fixture supplies coefficients; no tokenizer, framework or model forward runs. Raw inputs/workspaces/outputs/modules stay in separate90-day remote artifacts, with small reports and partial errors uploaded even on failure. No job is dispatched by source preparation. Whole-model ten-case replay and any proposed production correction remain separately pending after actual causal evidence and review.
