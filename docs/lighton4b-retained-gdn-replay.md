# LightOn4B retained synthetic GDN replay

Original component run [41923599391](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41923599391)
at `52275dd1fbe07a1920aa409ca2ad5f3f7ee51d01` is preserved as a numerical
negative. All four cases failed standalone `full_prefix.delta.output`; two also
failed final recurrent state. Grouped/projected full-prefix stages passed.

The test helper `data(const std::vector<float>&)` collided with `std::data`
through argument-dependent lookup. For mutable vectors, `std::data(C&)` was a
better overload. `previous_delta.clear()` retained its allocation, so the next
fresh-prefix standalone recurrence consumed stale cached state instead of null.
Renaming the helper to `optional_data` fixes this test-only routing error.
The new `empty_cache_is_null` control explicitly checks a cleared allocated
vector. Production recurrence arithmetic and original arrays are unchanged.

Retained source artifact13367509290 is named
`lighton4b-components-41923599391`, 230,669,375 ZIP bytes, with service digest
`e90d303546d53e0d7352f43ae2e8362063dbb9bb901c0a2b7613a59af5e65532`.
The independently reviewed original component report has SHA256
`8709072c0f080a47a5d1a8b52b34c3eb6eba9bee9152de2c87872aa2c4e88fa0`.
[Replay41924863763](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41924863763)
at `cf18dc728` verifies the archive digest, pinned original report/source/run,
all four GGUF pairs and every original raw tensor before native build/execution.
It imports NumPy2.3.5 only for array reading; no original framework import or
forward executes. Original-reference and ordinary-control jobs are skipped.

[Independent hosted report review41925188488](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41925188488)
at `f6034d733` validates all four cases, each complete ordered52-stage panel,
all authored controls, frozen source payloads and false qualification fields.
The retained replay report SHA256 is
`69f5ebc795e5cf8dc1ffc6fe39496a10476a5bde94a9f9536e1112d096b05ce3`.

| Synthetic case | Passed stages | Maximum absolute error | Minimum row cosine |
|---|---:|---:|---:|
| equal16-t65 | 52 | 3.397464752e-6 | 0.999999996995 |
| grouped16-32-t65 | 52 | 2.861022949e-6 | 0.999999945275 |
| grouped16-32-t1 | 52 | 1.549720764e-6 | 0.999999999502 |
| grouped2-4-t3-b2 | 52 | 7.152557373e-7 | 0.999999999784 |

Budgets remain absolute0.002 and row cosine0.99999. All208 comparisons pass;
no tolerance changes or regenerated reference vectors were used. The initial
new-file manual-dispatch attempt returned404 before execution because that
workflow was absent from the default branch. The successful replay and report
review used temporary registered `controls.yml` dispatch jobs, now removed;
the replay/checker scripts and historical workflow source remain recoverable
at the named commits. Large reference/native artifacts remain remotely retained.

This qualifies only bounded synthetic CPU/F32 components at hidden width16,
including16:32 head geometry and width128. It does not qualify full hidden width,
checkpoint payload, model loading, export, quantization, OCR text, EOS/document
quality or a public alias. Existing development0.17.15 remains unreleased.
