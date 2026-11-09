# Actual-input SwiGLU attribution

The native scalar FFN activation differs from the pinned upstream fused SwiGLU, but this experiment finds no changed quantized integer codes or large output amplification from that activation difference. Production remains unchanged. The [hybrid GDN decoded regression](lighton4b-gdn-full-ab-results.md) remains binding; no quality or release claim follows from this component test.

Source `38171f0d4a60d1eacb98b75461fb7f0e96815d22` completed the CPU-only operator worker in137.355 seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-swiglu-source-probe/v1/swiglu-source-probe.json) SHA256 is `d6372a51965f260a09437d9142ba80059a4c157f29f6a273220327908ed8a24d`. The exact captured inputs come from neutral upstream8df/reader5a8 and completed native OFF/ON40b2 receipts. Both independent source/layout/API reviews passed before submission; actual hosted compilation, synthetic backend controls and all9 whole actual projections subsequently passed.

The original published GGUF payloads are Q4_K gate/up (13,271,040 bytes each) and Q6_K down (19,353,600 bytes), totaling45,895,680 bytes. Every range and full tensor hash matches physical import920b; coordinates and packed bytes remain unchanged. Preferred-Q4 source4 gate/up and original-Q6 source4 down reproduce the actual captured upstream outputs exactly. RAW native OFF/ON16 projections reproduce the actual native outputs exactly. There is no Q4-down substitution, re-quantization, F32 model, model inference or held-out input.

[The pinned Qwen3.5 FFN](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/qwen35.cpp) selects parallel SiLU; [the pinned graph builder](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/llama-graph.cpp) uses `ggml_swiglu_split(gate, up)`. Native linear and attention blocks use scalar `(gate/(1+exp(-gate)))*up`. Scalar replay exactly matches actual native activations. Native GGML fused SwiGLU exactly matches upstream GGML on all8 predeclared same-input sets, and both exactly match the live source4 activation.

| Identical gate/up context | Activation reference RMS magnitude | Scalar activation RMS / max error | Same-Q6 native down RMS / max error | Changed Q8_K scales / blocks | Changed integer qs / bsums |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual source4 | .0381853 | 1.39921e-9 / 5.96046e-8 | 6.37168e-9 / 3.72529e-8 | 21 /144 | 0 /0 |
| Native GDN-OFF16 | .0380511 | 1.83717e-9 / 5.96046e-8 | 7.01247e-9 / 5.96046e-8 | 97 /576 | 0 /0 |
| Native GDN-ON16 | .0380511 | 1.88675e-9 / 1.19209e-7 | 7.38826e-9 / 6.70552e-8 | 107 /576 | 0 /0 |

For native16 contexts, the target is upstream fused arithmetic on those exact native gate/up arrays; it is not an actual upstream16 model trace. Prefix1/4 cases are also measured separately. Down-output RMS reference magnitudes for the three table rows are .0465778, .0375984 and .0375984.

Public Q8_K converter vectors are interventions, not directly captured internal buffers. All48 forced-Q8-input versus F32-input down-output comparisons close bit for bit, qualifying those output interventions within the tested cases/backends. No integer qs or block sums change between scalar and fused activation on any input set; only floating block scales change. This rejects the proposed integer-threshold amplification mechanism for SwiGLU in these cases. It does not rule out threshold effects from earlier post-normalization input drift.

The independent audit downloaded878,720 bytes of activation/code arrays, verified every file against the worker manifest and rechecked source4 fused/source identity, scalar errors, both forced-Q8 output closures and the21-scale/zero-integer result. [Audit and hashes](../tests/results/qwen35-ocr/lightonocr3-4b-swiglu-source-probe/v1/independent-audit.json) are retained; full raw arrays remain in completed Kaggle `lighton4b-swiglu-source-operators-v1`, version1.

A distinct backend negative remains: for identical16-token activations and Q6 weights, native versus upstream down projection differs by approximately1.23e-8 RMS/max2.68e-7 (reference magnitude .0375984); corresponding1/4-token comparisons are exact. This is kept separate from activation arithmetic. No runtime change or new decoded run was performed; these component results do not establish a quality improvement. The next bounded attribution examines actual source-versus-native early FFN postnorm inputs and their Q4 gate/up converter/output effects with identical-input controls.
