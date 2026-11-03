# Document evaluation for new OCR profiles

`tools/ocr_document_eval.py` introduces a separate, versioned document record and
diagnostic scorer. It leaves the frozen `tools/ocr_quality.py` policy, its source
pins, old scores and rejected Ovis/LightOn runs unchanged. The new controls use
authored data and execute no model.

This is a common interchange and diagnostic layer. It is not an implementation
of OmniParseBench's pass/fail tests, OmniDocBench's TEDS/CDM metrics, or a new
accuracy claim. Run the official scorer on its pinned dataset and protocol to
report an upstream benchmark result. The export shape follows
[OmniParseBench `views.py`](https://github.com/datalab-to/omni_parse_bench/blob/04d72ec396113c8227e91cbdc0b277985dede68c/omni_parse_bench/views.py)
and [layout contract](https://github.com/datalab-to/omni_parse_bench/blob/04d72ec396113c8227e91cbdc0b277985dede68c/omni_parse_bench/harness/layout.py).

## The three separate questions

1. **Native fidelity:** compare the original checkpoint and CrispEmbed with the
   same prepared inputs, processor, prompt, generation policy and model revision.
   Capture intermediate arrays, decisive logits and free-generation IDs/text/EOS.
   Apply the current [precision policy](precision-acceptance.md), retaining the
   strict diagnostics and decoded differences.
2. **Document accuracy:** compare both outputs against independently annotated
   text, ordered blocks, boxes, HTML tables and formulas. Missing annotation is
   unavailable, not an empty/perfect target. Record incomplete pages and parsing
   failures without excluding them from the panel.
3. **Deployment cost:** measure cold/warm latency, throughput and peak RSS on the
   target hardware with the same timing span and pipeline. A source model's H100
   throughput says nothing about CrispEmbed CPU or Metal latency.

The document scorer always emits `model_qualified: false`. Its exactness flag
is a diagnostic, not the repository's precision acceptance decision. No alias,
publication, default change or app upgrade follows from this tool's success.

## Record contract

Both gold and prediction use `schema: crispembed-ocr-document-v1`, a `page_id`,
and `protocol_sha256`. Freeze the protocol before inference; hash its actual
bytes in the producing runner. The protocol must identify its dataset revision,
split, selection, licensed inputs/annotations, rendering, model arms, prompts,
caps, scoring source and intended comparison. Keep development/selection pages
separate from a subsequent independent accuracy panel.

`input` contains `source_sha256`, `rendered_sha256`, positive `width`/`height`,
and `rendering: {tool, version, settings}`. All must match between gold and
prediction. Hash the exact rendered bytes and record the color/orientation/DPI/
long-edge recipe in `settings`; do not silently resize to make a page pass.

`document` contains these required fields:

| Field | Meaning |
|---|---|
| `raw_text` | Unmodified UTF-8 emitted by the model/pipeline. Never overwrite it with cleaned text. |
| `text` | The producer's explicit text view in reading order; preserve its transformation in the protocol. |
| `html` | Actual HTML output, or `null` if unavailable. |
| `blocks` | Ordered `{label, bbox, text}` objects, or `null`. `bbox` is normalized top-left `[x0,y0,x1,y1]`, or explicit `null` if geometry is unavailable. |
| `tables` | HTML table strings in emitted order, `[]` for known no tables, or `null` if unannotated/unavailable. |
| `formulas` | Formula strings in emitted order, `[]` for known no formulas, or `null`. |

Block labels use the pinned OmniParseBench vocabulary: `text`, `heading`,
`page_furniture`, `list`, `caption`, `footnote`, `equation`, `code`, `table`,
`form`, `figure`. Each model adapter needs a reviewed mapping. Unknown labels,
nonfinite/inverted/zero-area/out-of-range boxes fail validation; the interchange
does not silently clamp, sort, infer boxes or drop blocks.

Gold adds `annotation: {source, license, method}`. This states provenance; the
tool cannot independently prove that a human annotation is correct or that a
license is suitable. Use the already approved corpus or review actual source
terms before importing another one. OmniParseBench itself has per-source terms.

Predictions add:

- `run`: `engine`, `task`, `precision`, `backend`, `build_mode`, immutable
  `runtime_revision`, and `components`. Each component has a distinct `name`,
  `repo`, exact 40-character checkpoint `revision`, and SHA-256 for `weights`,
  `processor` and `tokenizer` (`weights_sha256`, etc.). For multiple files, hash
  a sorted manifest of relative names and individual SHA-256 values. For an
  inapplicable tokenizer, hash an explicitly empty manifest; never use the OCR
  model's tokenizer hash for a separate layout detector. Include the layout
  model and all other learned pipeline stages.
- `run.expected_calls`: every generation call ID from the pipeline trace,
  including layout, crops and retries. This must equal the generation record
  IDs. A runner must derive it from its trace, not its successful calls alone.
- `generation`: every call's `id`, exact `prompt`, `max_new_tokens`,
  `generated_ids`, `eos_token_ids` and `status`. `eos` requires actual final EOS
  ID evidence; `cap` requires exactly the declared cap without EOS. `error` and
  `unknown` are incomplete. `generated_ids: null` is allowed only when completion
  is not asserted. EOS-looking text is not evidence. A multi-crop page completes
  only when every declared call completes.
- `performance`: `elapsed_ms`, `peak_rss_bytes`, and a named `span`. Use explicit
  `null` when a measurement is unavailable. Do not substitute zero for missing
  latency or RSS.

SHA/revision syntax validation is not remote attestation. The producer must
actually hash the artifacts it loads, verify expected source pins, and retain
its portable build receipt. Keep the original record next to any exported view.

## Diagnostics and limitations

Raw and ordered text each receive Unicode-character CER and whitespace-word
WER. Neither view applies case folding, Unicode normalization, punctuation
removal or fuzzy matching; insertion errors may exceed 1.0.

Blocks are compared by their emitted position. Report label agreement, text
errors, count differences and position-indexed box IoU. This deliberately exposes
reading-order changes. It is not detection matching, mAP or OmniParseBench layout
accuracy. Unavailable boxes produce null geometry, never zero or perfect IoU.

Tables preserve explicit cell coordinates, `th`/`td`, and positive `rowspan` /
`colspan`. Cell text collapses whitespace only. The parser rejects nested tables,
overlapping spans, row-span overflow, holes/ragged grids, duplicate attributes,
scripts, arbitrary attributes and malformed HTML. Supported inline formatting
is `b`, `strong`, `i`, `em`, `sub`, `sup`, `span`, `p`, `div` and `br` without
attributes. Unsupported HTML remains visible as a per-table parse error. No
browser repair or score substitution occurs. This exact cell/span comparison
is not TEDS/GriTS.

Formulas receive literal text errors/exactness. `x^2` and `x^{2}` differ; the
tool does not pretend to measure mathematical equivalence or CDM.

`exact_complete_document` is true only for complete generation and exact text,
blocks, tables and formulas; false for an observed mismatch/incomplete output;
null when annotation/prediction/geometry is unavailable and no observed mismatch
already makes it false. Raw serialization differences stay separate. A broad
benchmark summary must retain unavailable/error/capped counts instead of averaging
only the successful documents.

## Commands

```sh
python tools/ocr_document_eval.py validate prediction.json
python tools/ocr_document_eval.py validate gold.json --gold
python tools/ocr_document_eval.py score gold.json prediction.json > diagnostics.json
python tools/ocr_document_eval.py export-opb prediction.json > opb-output.json
python tests/test_ocr_document_eval.py -v
```

Validation/export return 0 on a valid record. Scoring returns 0 for an exact
completed annotated document, 1 for a mismatch/incomplete/unavailable diagnostic,
and 2 for invalid input/provenance. Always retain the emitted JSON, including
nonzero results. The export contains only observed `html`/`blocks`; it never
creates layout for a text-only model. Calling this export is not an official
benchmark run. Full dataset execution and representative licensed annotations
remain separate, hosted work.

The executable, authored positive/negative examples are in
`tests/test_ocr_document_eval.py`, included in the shared model-free controls
workflow. They establish the record/scorer behavior, not model quality.
