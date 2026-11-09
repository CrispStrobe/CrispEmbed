# Physical GDN decoded A/B: negative

**Keep `CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS` default-off and unqualified. Do not ship this path.** Exact source-fed components did not establish decoded quality. Global Q4 repack remains off; no default, alias or release change follows.

The bounded [Kaggle run](https://www.kaggle.com/code/chr1s4/lighton4b-gdn-full-ab-v1) completed at source `f6a1e60d3cea5a1f6841079697734906f70d8b40` in 1,118.98 seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-full-ab/v1/gdn-full-ab.json) SHA256 is `40b20127fe0a0ffbbbc227f40a847a31a5ac6c0c8c231edff24aa17ce1c6cdc5`. Runtime, CMake and reference CLI match tested source `cd50e9e691530525fb998edda8a449cc0725be91`. Physical model SHA `843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43` is unchanged.

| Observed case | OFF → ON tokens, all EOS | Character edits against original BF16 | Word edits | First-score RMS error against upstream | Child wall seconds |
| --- | --- | --- | --- | --- | --- |
| Table | 282 → 294 | 349 → **364** /607 | 97 → **104** /106 | 0.188629 → 0.152752 | 210.16 → 234.78 |
| Form | 167 → 167 | 175 → 175 /656 | 45 → 45 /90 | 0.105889 → 0.100611 | 143.72 → 173.25 |

Form raw IDs and text remain exactly unchanged. Table is worse despite its improved first-score error. Actual upstream table remains348/607 character edits and97/106 word edits, so neither native table arm meets the predeclared floor. Native versus upstream table edits worsen from9 characters/7words to27/17. Observed RSS peaks are approximately4.28–4.32GB. Decode lengths differ on table; this is not a balanced speed benchmark.

OFF closes all282/167 vocabulary rows, all38 text anchors, vision/processor arrays, IDs and EOS against the immutable baselines. Only the GDN gate changes; ON keeps actual pixels, merged vision, prompt/modalities, embeddings and positions identical. Every emitted row is finite with actual greedy argmax. Native OFF/ON first differs at layer0: table RMS2.80664e-5/max0.000924185, form4.54486e-5/0.00196648. At layer31 these differences are0.0676861/4.19354 and0.0671365/4.60565. These are paired native stages, not actual upstream hidden states.

The independent review rehashes14 selected closed files, validates actual summaries/IDs/EOS, recomputes raw character/word edit counts against both references, confirms the sole gate difference and authenticates the unchanged physical pack. Large score/stage arrays and exact binary/backend bundle remain remote; their complete numerical/hash checks are producer checks, not a second independent full-array replay. This run has one teacher row plus independent greedy cap512, not a177-position teacher panel. Original BF16 and upstream models were not rerun. Held-out cases were not used or rerun; their historical-receipt negative remains binding.
