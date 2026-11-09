# Same-input GDN localization: small differences, no runtime correction

The corrected [raw result](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-source-primitives/v2/gdn-primitives.json)
passes actual native helper values/state and retained first-sixteen attention
byte controls. On identical native projected inputs, the earliest observed
operator difference is convolution SiLU, with RMS about `1.04e-8` and maximum
`9.54e-7`. This experiment identifies no large first-block GDN semantic mismatch
and qualifies no production correction or decoded-quality improvement.

| Stage, sixteen tokens | RMS error vs pinned source graph | Maximum absolute error |
| --- | ---: | ---: |
| Raw convolution | 0 (byte exact) | 0 |
| Convolution SiLU | 1.037740107e-8 | 9.536743164e-7 |
| Q L2 normalization | 7.993060790e-9 | 1.192092896e-7 |
| K L2 normalization | 7.812400753e-9 | 1.192092896e-7 |
| Fused GDN output | 6.534327760e-10 | 4.470348358e-8 |
| Recurrent state | 4.667023859e-9 | 7.152557373e-7 |
| Post RMS normalization | 5.358527509e-8 | 1.907348633e-6 |
| Gated post normalization | 8.234390780e-9 | 4.768371582e-7 |
| Dequantized F32 output projection | 5.226904237e-9 | 4.768371582e-7 |
| Retained Q4_K output projection | 2.632940414e-8 | 4.768371582e-7 |

Alpha bias, source softplus and decay are byte exact between the reconstructed
legacy native path and the primary source graph on these inputs. Beta sigmoid
and Z SiLU have small differences. The four-token results are retained separately
in the raw report. These composed comparisons include upstream arithmetic within
the component; they do not isolate every nonlinear primitive on identical input.

The source graph also runs on the native ggml revision. Those backend controls
are **not universally byte exact**: softplus/decay differ, and sixteen-token
state/output differences reach `7.15e-7` maximum. With identical native
post-normalized inputs, dequantized F32 output projection is byte exact across
both revisions; retained Q4_K projection is exact at four tokens but differs at
sixteen tokens by RMS `2.59e-8`, maximum `4.77e-7`.

The [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-source-primitives/independent-review.json)
checks all six composed/observed controls, actual main attention closure,
parameter/input identity and every reported stage hash across the initial and
corrected runs. The initial run's source raw RMS buffers were unmaterialized
zeros: CPU RMS-plus-multiply fusion skipped that intermediate. Its large raw RMS
comparison is **invalid**, preserved for audit and excluded from attribution.
The correction requests intermediate output flags. Only those four source RMS
artifact hashes change between runs; every other stage hash remains identical,
and composed output/state controls remain byte exact.

The reference blueprint is pinned to llama.cpp
[`6c01da6278ab6c69ed67d9124806b50d2b299ba7`](https://github.com/ggml-org/llama.cpp/tree/6c01da6278ab6c69ed67d9124806b50d2b299ba7):
[`qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/qwen35.cpp),
[`build_gdn_l2_norm`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/models.h),
and [`build_delta_net_fused`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/delta-net-base.cpp).
The zero-state K=1 fused branch follows no-draft context defaults.
[`ggml_cpu_try_fuse_ops`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-cpu/ggml-cpu.c)
and [`ggml_node_has_n_uses`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-impl.h)
explain the initial intermediate observation limitation.

Hosted CPU operator workers `143170539/v1` and `/v2` used exact task sources
`7427940f6567f66b4dc9fcfc288ffdd71ee72faf` and
`0bbe9d8769f13cacd8788194298bf8cb3c26f84e`. The corrected result SHA256 is
`0805d52b3c1e02c81251f2dff668c88076354c706a8decc2e95be035ba930593`.
[Protocol](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-source-primitives/protocol-v1.json),
submitted bootstraps, exact build caches, logs, source checks, timing and small
artifact hashes are retained with both runs. Five complete publisher tensor
ranges total 6,030,080 bytes; no model was loaded or decoded, and no full F32
model was created. Compilation used two threads, operator kernels four, with
2 GiB effective RAM / 4 GiB disk admission and a 30-minute controller cap.
All stage arrays remain remotely retained.

The inputs are actual native first-block projected activations from the pinned
physical baseline, not upstream full-model internal activations. Four tokens
match the source's initial text-prefix batch length; sixteen include image tokens
and form a component counterfactual. Both start with zero recurrent state and
convolution history. Later state, other layers, upstream F16 attention caches,
full multimodal features and decoded output remain outside this measurement.
The separate negative same-feature quality/ablation results remain in force;
all experimental production gates stay off by default.
