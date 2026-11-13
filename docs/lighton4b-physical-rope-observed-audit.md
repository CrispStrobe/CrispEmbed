# Physical RoPE observed A/B: independent small-artifact audit

The [independent audit](../tests/results/qwen35-ocr/lighton4b-physical-rope-observed-small-audit.json) binds producer receipt `aaf5298b944cea3d53d7e6e4ce88da78c58c3cec6a651733c5aff993fa4c6ffc`, runtime candidate `a8a22749ab0b757ae1f317b6fa5b84ffd148b939` and tested controls `855471a734c3188fcf6e83b93eeaac7ba6f101a51728ff838b5b04e6babf8d2e`. The model controller is `eb2d04219f7a8567803e77b3ff5bbaa398b16629`; the [observed protocol](../tools/physical-rope-observed-ab/protocol.json) preserves the physical checkpoint, inputs, cap512 and threads4. Only physical RoPE changes OFF→ON; fused SwiGLU remains ON. This audit performed no inference, source capture, rebuild or held-out evaluation.

All three ON token sequences and decoded strings equal the frozen quantized source responses, and all six arms end with EOS. This source fidelity result retains the poorer comparison against the original BF16 outputs:

| Observed case | OFF → ON tokens | Character edits against original BF16 | Word edits against original BF16 | ON source token/text equality |
|---|---:|---:|---:|---|
| Table | 218 → 282 | 20 → 348 | 6 → 97 | Exact |
| Form | 167 → 170 | 175 → 189 | 45 → 46 | Exact |
| Arabic line | 6 → 6 | 0 → 0 | 0 → 0 | Exact |

Character and word edit counts, source equality, matching token prefixes, reference lengths and stop states were recomputed independently from the immutable small JSON responses. The edit-distance implementation uses bit-parallel operations, separate from the producer's dynamic-programming scorer. Each ON output meets its declared quantized-source floor, which is distinct from improving the original BF16 comparison or establishing task accuracy. Table and form remain negative relative to their OFF results; Arabic stays exact. No expert label adjudication or quality promotion follows.

The six native first-score vectors and three source first-score vectors were checked against their recorded complete sizes and SHA-256 hashes, full-vocabulary finiteness and actual greedy argmaxes. Recomputed ON first-score errors against the quantized source are:

| Case | RMS | Maximum absolute error |
|---|---:|---:|
| Table | 6.0679675075e-7 | 4.7683715820e-6 |
| Form | 6.5491990011e-7 | 3.8146972656e-6 |
| Arabic line | 5.7790449440e-7 | 3.8146972656e-6 |

Eight selected form first-chunk arrays (gate, up, fused activation and down for OFF and ON) were also rehashed and compared with the immutable source trace. All eight match source bytes exactly. The first-score, teacher-row and greedy-row-zero hashes agree in the producer manifests. Recomputed first-score RMS/magnitudes differ from the producer by at most two double ULPs; maximum absolute errors match exactly, within the existing eight-ULP recomputation allowance. No model numerical budget changed.

The independent audit checks the OFF token/text/summary identities against the prior fused-ON producer, plus 637 recorded OFF stage/row manifest identities (300 table, 249 form, 88 Arabic). The controls receipt, tested executable/library identities, single-gate intervention, declared unchanged layers0–2 and historical source-response hashes also close. These manifest comparisons do not claim a fresh read of the complete raw inventories. Seventeen selected small vectors totaling 9,906,176 bytes were checked; complete raw rows, later hidden layers and runtime artifacts remain remote and were not independently rehashed by this audit. The earlier component controls retain their explicit decoder-compose build-only limitation.

The public audit publishes counts, error metrics and hashes, without recognized strings or token sequences. The original producer and failures remain immutable. Every experimental gate remains default-off; this observed calibration panel creates no release, alias, application upgrade, speed result or held-out qualification.
