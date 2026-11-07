# Actual-input GDN replay

The bounded operator-only replay at source `2f1261053` measured the first four text tokens of the observed table. The [raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-live-gdn-replay/report.json) has SHA256 `7ff684cf80e3f1d11368615c0aa54fbc6219404ce1a31f238bd152a6957981c7`. It completed in 134.237 seconds and transferred five whole parameter ranges totaling 6,030,080 bytes. It ran no model generation, used no held-out cases and establishes no decoded-quality improvement. Large operator arrays remain in the completed Kaggle output `lighton4b-live-gdn-stage-replay-v1`.

The upstream graph is pinned to `6c01da6278ab6c69ed67d9124806b50d2b299ba7`; native GGML is `3a27b6c665b191cb230be367db5c52d81dd3e2ac`. Actual live source convolution, SiLU, gated post-normalization and preferred Q4 output projection are all byte-exact to replay. Their reference RMS magnitudes are respectively 0.424982, 0.259703, 0.070063 and 0.059582. Native sixteen-token replay closes the actual native attention output; native four-token post-normalization and both output routes close the native sixteen-token prefix exactly. Observer/composed controls pass.

The independent state audit also closes all 524,288 values of the actual next graph pre-delta state after the four-token prefix. Its reference RMS magnitude is 0.0314137. The recorded head permutation and state transpose change coordinates; the audit compares the resulting F32 values bit for bit. This tests a real nonzero state rather than only an initialized zero state.

| Counterfactual post-normalization versus live-source replay | RMS error | Maximum absolute error |
| --- | ---: | ---: |
| Native operators and native projected inputs | 2.80247e-8 | 3.15905e-6 |
| Native operators, source Z only | 9.42353e-9 | 4.76837e-7 |
| Native operators, source Z/alpha/beta | 9.85190e-9 | 4.76837e-7 |
| Source graph on native GGML, source Z/alpha/beta | 1.49973e-9 | 1.19209e-7 |

These are same-input operator interventions, not complete model alternatives. RAW output-route metrics compare to the reference RAW route; they must not be described as a comparison to the actual source preferred output. Prior full-repack decoded negatives still apply.

A concrete source difference remains in softplus. [Native GGML at its pinned commit](https://github.com/ggml-org/ggml/blob/3a27b6c665b191cb230be367db5c52d81dd3e2ac/src/ggml-cpu/unary-ops.cpp) computes `logf(1 + expf(x))`; [the pinned upstream](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-cpu/unary-ops.cpp) computes `log1pf(expf(x))`. Existing decoder scalar softplus already matches the upstream alpha-biased, softplus and decay arrays exactly. Replacing that scalar path with native GGML softplus introduces a difference. A subsequent default-off hybrid experiment therefore retains scalar softplus and separately tests source-shaped GDN operators. This receipt does not verify that unexecuted candidate or authorize its promotion.


## Hybrid setup negative

The first default-off hybrid operator worker at source `a6e5b3d3d2ec24ab7fc9efe240dbbf59acc01546` stopped before compilation and parameter transfer. The inherited compile-command guard expected one decoder helper entry; adding the separate cache-controls target correctly generated a second entry. The [raw failure receipt](../tests/results/qwen35-ocr/lightonocr3-4b-live-gdn-replay/hybrid-v1-setup-negative/gdn-hybrid-probe.json), SHA256 `006db7e404942044de8ab45f0bb2b31ecb588920028c6bb909cc76a16fde6f61`, records 22.772 seconds; the bootstrap took43.673 seconds. Operator computation remained false and no numerical result was produced. Fix `fcb9d6c72` requires exactly both named targets for both helper files, each with contraction disabled; duplicate, missing, unknown-target and missing-flag mutations are rejected. This setup failure is retained independently of any corrected retry.
