# TeleOCR retained validation evidence

**Experimental and disabled by default. Strict numerical qualification remains
false.** Use the [validation report](../../../docs/teleocr-validation.md) for
interpretation and the [usage guide](../../../docs/teleocr.md) for portable
build, conversion and test commands. The [license audit](../../../docs/teleocr-license.md)
and [NOTICE](../../../NOTICE.teleocr) identify source and redistribution terms.

The machine-readable [final matrix](summary.json) independently aggregates the
24 recorded CPU cases from their original reports and frozen references.
It preserves separate decoded, numerical, coverage and confidence outcomes.
An exact output does not make an intermediate numerical failure pass. The
[independent final review](final-matrix-peer-review.json) confirms all 24
reports, their original receipts, the threshold classifications and the
separately scored Q4 code-output difference.

## Evidence index

| Scope | Primary records | Interpretation |
| --- | --- | --- |
| Original HF model | [Reference panel](reference/reference-results.json), [producer receipt](reference-producer.json), [independent review](reference-peer-review.json) | Six actual original-module CPU/F32 executions, frozen before native comparison; full IDs and text, 75 observation names per case |
| Native F32 | [Six-case run](native-f32/summary.json), [independent review](native-f32-peer-review.json) | Complete decoded sequences and numerical comparisons |
| Native F16 | [Four smaller cases](native-f16-small/summary.json), [code/layout](native-f16-large/summary.json) | Same frozen prompts, images and generation caps |
| Native Q8_0 | [Four smaller cases](native-q8-small/summary.json), [code/layout](native-q8-large/summary.json) | Protected vision/embedding; 196 decoder matrices quantized |
| Native Q4_K | [Four smaller cases](native-q4-small/summary.json), [code/layout](native-q4-large/summary.json) | Actual protected Q4_K recipe; numerical and decoded outcomes remain independent |
| Public interfaces | [Suite](public-f16/suite.json), [individual checks](public-f16/public.json), [compiled binding build receipt](binding-compilation/build-checks.json) | Actual C/Python/orchestrator/CLI/HTTP and compiled Rust/Dart consumers; interface consistency against a native anchor |
| Document parser | [Suite](parser-f16/suite.json), [detection](parser-f16/detection/report.json), [segmentation](parser-f16/segmentation/report.json) | Native layout/crop generations, typed JSON, Markdown, OTSL/HTML and confidence/cap checks; bounded smoke scope |
| Production control | [Initial full-text control](production-observer-control-f16/report.json), [final production control](production-post-review-f16/report.json) | Normal CLI without test-only observers; frozen text, 66 IDs and EOS |
| Loader | [Original 26-case suite](teleocr-loader-final.json), [final 43-case suite](teleocr-loader-post-review.json) | Header-only malformed-file subprocess tests, including final norm/bias dtype rejection |
| Final source/build | [Source delta](final-source-review/source-delta.json), [dtype preflight](final-dtype-guard-preflight.json), [build](build-post-review.json), [installed notices](install-post-review.json) | Final edit changes admission checks only; old snapshots and binary hashes remain unchanged |
| Model containers | [Source hashes](source-manifest.json), [conversion manifests](model-manifests/), [Q4 audit](quantize-q4-final-audit.json), [F16 final conversion](convert-f16-final-verification.json) | Full 700-tensor inventories, exact model hashes and recipe provenance |
| Model-free contracts | [Python tests](python-unit-tests-final.log), [quantizer tests](quantizer-provenance-tests.json), [preprocessing](preprocess-final.log), [Pillow pixels](preprocess-pillow13/results.json) | Converter/parser/API contracts, metadata and exact input patch comparisons |
| Shared regressions | [Environment gates](env-gate.log), [BPE](bpe-pretokenize.log), [Qwen tokenizer](qwen-pretokenize.log), [engine selectors](engine-selectors.log), [exit convention](clean-exit-final.log) | Bounded existing shared-component regressions |
| Numerical diagnosis | [Paired magnitudes](f32-paired-activation-advisory.json), [portable diagnostics](portable-diagnostics/receipt.json), [probe instructions](../../../tools/teleocr_diagnostics/README.md) | Same-input/original-module propagation and F32/F64 diagnostics; strict failures preserved |
| PDF ingestion | [Rendering receipt](pdf-ingest.json) | One-page PDFium input/rendering check; PNG model inference is a separate scope |

Each native run directory retains per-case JSON and logs beside its summary.
The parser directories retain raw model generations, interpreted JSON and
Markdown. Source snapshots and executable hashes document what actually ran;
later source is not substituted into old receipts.

## Preserved initial failures

- [Initial F32 text](native-f32-initial/summary.json): input-pixel mismatch and
  one missing observer boundary, despite exact decoding.
- [First corrected F32 text](native-f32-pillow-fix/summary.json): exact pixels
  and complete observation coverage, with numerical failures still present.
- [Initial loader gate assumption](teleocr-loader-initial-gate-assumption.json):
  the test initially treated the nonempty string `false` as disabled; the
  corrected test follows the existing repository convention.
- [Loader/build overlap](teleocr-loader-build-race.json): transient dynamic
  loader errors from testing during relinking. The completed sequential
  suites are retained separately.
- [Initial Q4 metadata audit](quantize-q4-initial-audit.json): stale precision
  recipe and incorrect file-type metadata; all final tensor payloads are
  unchanged.
- [Initial source-header check](final-source-review/initial-header-check.json):
  a comparison retained whitespace after stripping an added comment. The
  source-delta review distinguishes comment-only header changes from code.

## Reproduce and verify

The [matrix producer](summarize-matrix.executed.py) rechecks report hashes,
duplicated receipt fields, reference strings/IDs, EOS, coverage, confidence
counts and edit-distance scores without running inference. From the project
root:

```sh
python tests/results/teleocr/summarize-matrix.executed.py \
  tests/results/teleocr tests/fixtures/teleocr/synthetic-text.json
```

Its `summary_producer_sha256` records the exact producer bytes. CER denominators
use reference Unicode characters; WER denominators use reference whitespace
tokens. Source/native agreement on publisher examples is distinct from the
authored synthetic fixture's transcription gold.

The [initial collection inventory](collection-inventory-initial.json) records
the first 269 copied files and 63 source/receipt hash checks. It intentionally
predates the final long quantized runs and final loader checks. The delivery's
`_delivery/SOURCE-MANIFEST.json` and `SHA256SUMS` cover the complete final source
and evidence inventory, including later additions.

Raw receipts retain their original execution paths and bytes. Checkpoint
weights, generated model GGUFs, large activation arrays, build outputs and
toolchains are excluded. The original-module dumper and pinned source can
regenerate references; the source archive includes the complete pinned ggml
code and the owned fixture. CPU validation timings may overlap other bounded
work and are diagnostic, not a controlled performance comparison. No GPU,
other-platform, broad-corpus or publication qualification is implied.
