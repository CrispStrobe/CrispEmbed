# Josev owned task-quality v1 label review

These20 new requests form10 EN/DE scenario pairs and60 field labels. Labels were authored before model outputs. They do not replace the old four smoke references. The panel is CC0-1.0 and is a bounded owned holdout, not a broad representative benchmark.

| Scenario (EN and DE) | Type of fact | NOUL gold | Choice gold | Completed checks / ordinal gold |
| --- | --- | --- | --- | --- |
| fact-key | factual | true | a | 1 |
| fact-badge | factual | false | z | 0 |
| negation-depot | negation | true | m | 2 |
| negation-letter | negation | false | a | 1 |
| correction-time | temporal-correction | true | z | 2 |
| correction-shelf | temporal-correction | false | m | 0 |
| order-finish | relative-order | true | a | 1 |
| order-visits | relative-order | false | z | 1 |
| count-lamps | explicit-count | true | m | 2 |
| count-room | explicit-count | false | a | 0 |

Each request includes literal state facts, one truth question, one three-way selection and an ordered0/1/2 count question. A state rationale is included with each label in the JSON. Option and question insertion orders vary before outputs; no row is dropped for difficulty or model disagreement.

Review every English/German state, question, option description and label in [the frozen panel](../tests/regression/josev2b/task-quality-v1.json) before authorizing the new original-only GPU capture. No execution is authorized by this draft.

The coordinator independently reviewed every full state, question, option and all60 labels before any model output; the separate [review receipt](../tests/regression/josev2b/task-quality-label-review-v1.json) records that source review. This is agent review, not human adjudication. The immutable panel SHA256 is `2b1a3c8d0c20be4a64a7f21695b80da8f5f727dd8b50a4d8fd00523aa20efb18`.

The new [versioned protocol](josev2b-task-quality-v1.json) reuses the authenticated official checkpoint, publisher Python3.13/CUDA F32 environment and audited57-wheel installation. It changes neither original mathematics nor the old four requests or thresholds. The [new original producer](../tools/dump_josev2b_task_quality_v1.py) first freezes all20 actual tokenizer records, masks, ordered fields and spans, rejects state clipping and more than8000 total tokens, and only then permits checkpoint payload download. Capture requires ordinary/observed/repeat neutrality and all74 declared arrays per case,1480 total. Original mistakes remain in the report; requests are never dropped.

The GPU bootstrap at [tools/kaggle/josev2b-task-quality/bootstrap.py](../tools/kaggle/josev2b-task-quality/bootstrap.py) is a source template. Render its `@COMMIT@` and `@QUALITY_PROTOCOL_SHA@` literals only after reviewing the committed source and protocol. Hardware capability≥7.5, CUDA13 driver≥580.65.06,14GiB total/11GiB free GPU memory,25GiB scratch and3GiB result-space barriers precede installation or payload access. Unsupported hardware is conclusive:false with no automatic retry. All arrays and logs stay in remote results; only small reports need review. No GPU execution has been dispatched by this source preparation.

The [hosted native caller](../tools/ci-heavy/josev2b_task_quality_v1.py) and [new20 comparator](../tools/josev2b_task_quality_panel_v1.py) are prepared separately. Native prerequisites are deliberately null and reviewed flags false: an explicit reviewed original20 receipt/raw manifest, current old-four Q4 candidate, exact F32/Q4 model hashes and runtime Git-object closure are required before model access. The source base used to author this harness is not a qualified Q4 runtime. The owner must fill those pins in a separately reviewed source change; no latest/main fallback exists. Native inputs are staged by that hosted owner; this caller has no downloader and performs no original recapture.

When activated after review, build with `tests/josev-task-quality` and `REVIEWED_RUNTIME` set to that exact checkout. The root portable CPU profile uses GGML_LLAMAFILE=ON, explicit root ISA flags and contraction OFF for probes, and the production shared library must exclude observer symbols. The caller authenticates all NPY1 hashes, shapes, extents and finite policy before NumPy. It compares1360 mapped stages through actual `crispembed_diff::Ref` plus independent NumPy measurements and retains120 original position/rotary/mask arrays as unmapped evidence. Those arrays are not120 native numerical comparisons. IDs, masks, spans, lexical IDs and occurrence order must match exactly. Public C and byte-identical production Python bindings run all20 raw requests, with repeat, rejection and recovery checks. F32 must pass first before any Q4 forward.

F32 requires every mapped stage max absolute≤0.002 and minimum row cosine≥0.99999, raw logits≤0.002, raw probabilities≤0.0001 and ordinal expected-index error≤0.001. Q4 keeps all those strict F32 flags in the report separately. New20 Q4 acceptance instead requires all60 discrete decisions agree with the original, no accuracy loss versus original or F32 by type/language, accuracy≥90% overall and≥85% per type/language, every public probability error≤0.01 and every ordinal expected-index error≤0.01. Score units are original criterion indices0..2; a normalized0..1 score is rejected. Total variation, margins, all misses, error magnitudes and first strict-stage failures remain visible. The old-four thresholds remain unchanged.

An original or native failure stays negative and cannot be repaired by dropping rows, replacing gold or changing these budgets after outputs. A failed Q4 task gate requires an explicitly new critical-precision recipe and reevaluation retaining prior results. Passing this small correlated holdout alone does not authorize publication, default gates, broad representative quality or a release/app upgrade.

The hosted native invocation, after all pins are explicitly reviewed, is:

```sh
python tools/ci-heavy/josev2b_task_quality_v1.py \
  --runtime REVIEWED_RUNTIME --reference AUTHENTICATED_ORIGINAL20 \
  --candidate-report REVIEWED_OLD4_Q4_REPORT \
  --f32-model PINNED_PACKED_F32 --q4-model PINNED_PACKED_Q4 \
  --build FRESH_HOSTED_BUILD --out FRESH_HOSTED_REPORTS
```

Staging and remote artifact service identity are the hosted caller's responsibility; this worker independently rehashes receipt/array/model bytes, the reviewed runtime Git objects and the compiled binary inventory in its fresh loader child. Its current pending flags make this command reject before any array/model/build access. No hosted native workflow is registered by this source-only preparation. Keep the full result trees, build/source logs, adapters, original arrays and native stages remotely for90 days; preserve small reference, freeze, raw manifest, protocol, source and command receipts for review. Do not put raw arrays or model files in Git. Public C and Python contexts are loaded sequentially to avoid two simultaneous2B weight allocations.
