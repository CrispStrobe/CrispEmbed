# Frozen OCR quality contract v1

This is a small Qwen held-out comparison, not OCRBench, OmniDocBench, olmOCRBench
or an estimate of general accuracy. It separates original-model agreement from
recognition against independently authored annotations. Neither exact Python
output nor a high cosine waives existing numerical failures.

## Sources and split

[panel.json](../tests/regression/ocr-quality/panel.json) freezes source/gold SHA-256,
selection, derivative geometry, prompt and cap before reference/native generation.
The two real-document transcripts come unchanged from the existing high-confidence
[manual annotations](../tests/regression/images/cc0/ground_truth.json), whose
provenance says no OCR output was consulted. The owned inventory grid and its
annotation were authored together before any model output; it is a controlled
fixture, not a scan or independent natural-document corpus.

| Evaluation case | Source terms | Explicit prepared size | Nominal 16-pixel patches | Gold |
|---|---|---|---:|---|
| Historical supermarket receipt | [CC0 original TIFF](https://commons.wikimedia.org/wiki/File:Alexander%27s_Supermarket_receipt,_late_twentieth_century.tif) | 352×704 | 968 | Existing full receipt transcript; TIFF/annotated PNG RGB equality required |
| Two-column book/test page | [CC0 source](https://commons.wikimedia.org/wiki/File:Test_OCR_document.jpg) | 448×576 | 1008 | Existing full transcript, title then left column then right column |
| Workshop inventory | [Owned MIT SVG](../tests/regression/ocr-quality/inventory.svg) | 448×448 | 784 | Heading, location/date, 5×3 grid, signature/batch |

Full pages are explicitly resized by this evaluation recipe, with no crop. This
can discard detail; results apply to these derivatives only. Larger/native-size
pages remain a separate task. Nominal patch counts are preparation checks, not
proof of the actual AutoProcessor grid. The model audit must check its original
processor output and reject more than1024 patches; never silently shrink again.

The four historical Qwen cases, including the completed receipt/prompt experiment,
remain **development**. The six frozen native-imatrix synthetic examples remain
**calibration**. Preparation rejects identical source byte hashes and
whitespace-normalized transcripts across these named sets and within evaluation.
It also checks original annotation hashes and unchanged text. This is not proof
against semantic overlap, arbitrary derivative copies, pretraining contamination
or use in other engines' earlier evaluations. Once these three cases are used to
choose a recipe, they become validation data; independent test data is needed
before a generalization or publication-quality claim. Do not tune to the test.

## Frozen inference policy

Use `lightonai/LightOnOCR-3-0.8B` at
`36d8636925327c671bd8e479c517a004f7566bd9`, its actual processor/chat template,
Transformers5.5.4/tokenizers0.22.2, F32 eager original inference/four CPU threads.
Use the existing compact dumper plus full-dumper anchor control. Freeze all
original outputs before any evaluation-native arm. Do not feed gold, original
embeddings or reference arrays into ordinary native generation.

The preselected prompt is upstream image-only (`""`) for all cases; the generation
cap is1024, greedy sampling off. EOS must be established from generated IDs, with
matching IDs/stop metadata validated by the existing decoded harness. Cap outputs
retain scores but are incomplete. Do not select a prompt or larger cap after seeing
which variant wins; revise the contract explicitly and retain the first results.

Compare existing F16-text/F32-vision, Q8, attention-Q8/FFN-Q6 and Q4 recipes as
separate arms; preserve calibration/precision alternatives and their exact tensor
manifests. Keep production precision defaults unchanged. Explicitly enable the
validated parallel-F16 conversion gate for comparable native timing, with other
precision gates and profiling off except separately named diagnostics. Retain
all earlier stage budgets and failures. New full score vectors use the shared
`crispembed-diff`/`qwen35_compare` gates; log errors, RMS/norms and the earliest
new differing stage before attributing a quality change to quantization.

For every arm, retain full original/native IDs, decoded UTF-8, EOS/cap, characters,
lines, actual per-token/mean confidence, raw and secondary quality scores and
matching-span timing. Probability alignment ends at the first ID divergence.
Record source/weight/tensor hashes, processor grid and all flags. Do not call
teacher-forced agreement independent free-generation quality. No thresholds,
precision defaults, HF aliases or app pins change under this evaluation contract.

## Scoring policy

[tools/ocr_quality.py](../tools/ocr_quality.py) emits all views independently:

- **Raw:** unchanged Unicode-character CER and whitespace-word WER from the
  existing comparator. Keep actual text and line/character counts. No trimming,
  case folding, Unicode normalization or markup removal in this view. CER can
  exceed1 when output adds characters.
- **Ordered content:** a versioned secondary parse of a restricted HTML fragment
  or simple Markdown pipe table/ATX heading. Convert supported serialization to
  visible text, decode HTML character references, then collapse whitespace only
  on both sides. Preserve case, accents, punctuation, repeated words and reading
  order. No bag-of-words matching, sorting, fuzzy gold alignment or optional
  omission of headings/footers. Report CER/WER; this view does not measure line
  wrapping, bounding boxes or geometric reading order.
- **Tables:** compare rectangular cell grids in emitted table/row/column order.
  Retain parsed grids and expected/actual table counts. Count different, missing
  and extra position-indexed cells; row shifts can produce multiple cell errors.
  This is not TEDS, GriTS or a rate normalized across different document sizes.
  Text-only output may have zero content error while failing table structure.
  The two real transcripts have no independent cell-grid annotation (`tables:null`):
  table scores and overall exact-quality are unavailable, not zero/perfect. Keep
  `content_exact` and all actual parsed grids; only the owned grid has structural gold.
- **Completion:** `exact_quality` requires established EOS, exact ordered content
  and exact annotated tables. With unavailable table gold it is null for a completed
  output (false for incomplete output). It is a stringent diagnostic, not a requirement that imperfect
  original OCR suddenly become accurate. Report native-versus-original parity and
  native/original-versus-gold accuracy separately. Quantization quality requires
  interpreting all per-document changes, not optimizing one aggregate pass bit.

Supported HTML tags: `div`, `p`, `h1`…`h6`, `b`, `strong`, `i`, `em`, `br`, `table`,
`thead`, `tbody`, `tfoot`, `tr`, `th`, `td`. Tables are top-level, cells explicit,
rows rectangular; only unit `rowspan`/`colspan` attributes are accepted. Other
attributes, comments, declarations, nested tables/repair, non-unit spans, fenced code
and unsupported markup produce `parse_ok=false`, null secondary scores and a
failed exact-quality diagnostic. Keep raw scores; never substitute zero or silently
exclude that document from an average. A future format extension requires its own
policy version and negative controls; keep v1 outcomes.

Simple pipe tables require outer pipes, a complete separator row (at least three
hyphens per column, optional alignment colons) and consistent column counts.
Escaped pipes, entities or embedded HTML inside pipe cells are outside v1. Inline
Markdown styling is retained literally rather than guessed away. These limitations
must be reported if original outputs fall outside the parser's scope.

CLI example (the caller must validate EOS metadata, not infer it from text):

```sh
python tools/ocr_quality.py tests/regression/ocr-quality/inventory.gold.json output.txt --complete eos
```

Exit0 means exact completed content/table quality; nonzero retains the JSON
measurements for imperfect, incomplete or unsupported outputs.

## Executable preparation and remaining evidence

[OCR quality contract CI](../.github/workflows/ocr-quality.yml) runs model-free
positive/negative controls and prepares derivatives on Ubuntu24.04 with
Pillow11.3.0/CairoSVG2.8.2. It records source revision, policy/scorer/panel/image/RGB
and DejaVu font hashes. Its small artifact contains only three PNGs and a report;
no weights/reference probes enter production artifacts. Once the prepared report is retained, CI automatically rejects changes to image/
RGB/source/gold/font/split and scorer hashes before inference. Rendering/library/font changes
require comparing pixels/hashes before claiming the same frozen inputs.

Preparation and visual review are required before model inference. Original/native
F16 and quantized runs, actual processor grids, decoded accuracy, confidence,
timing and comparative quality are still pending. Do not describe a green
preparation workflow or scorer control as model parity or recognition improvement.


Preparation and review are complete:
[regular38879605915](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38879605915)
passes all13 controls; [ARM38879334940](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38879334940)
produces bit-identical PNG/RGB and matching font/source/gold/split hashes.
[The retained report](../tests/results/ocr-quality-prepared.json) preserves raw
fields and report hashes plus review notes for all three actual PNGs. The receipt
is faint/source-clipped and the book text small; all inputs/gold stay unchanged.
These are preparation/visual observations, not model recognition scores.

## Held-out F16 audit

`tools/ci-heavy/qwen35_ocr_bootstrap.py --quality` checks the retained report before
heavyweight imports/downloads: successful public workflow, exact panel/scorer/
source/gold/split hashes and all three reviewed PNG hashes with notes. It rerenders
and compares frozen PNG/RGB/font hashes before model work. Preparation replay also
rejects a changed scorer hash; this closes a validation gap before held-out runs.
Missing evidence or changed rendering fails immediately. Model runs stay hosted.

The driver freezes all three original captures before any evaluation-native arm.
It reuses the compact/full anchor, strict prefill and generated-score comparisons,
ordinary owning-API generation and actual confidence. Native precision, weights
and runtime source hashes must match the previous F16-text/F32-vision baseline;
all other precision gates remain off and parallel-F16 conversion is explicitly1.
Actual processor grids must equal the frozen derivative grid and stay within1024
patches. Grid mismatch or a cap output stays visible rather than resizing or
extending the cap after seeing output. All full decoded text and raw/secondary
scores are retained; separate counters keep numerical, output, completion and
unsupported-format failures distinct. Original OCR accuracy is measured separately
from native/original agreement. Historical924/56 prefill debt stays archived, not
rerun or forgiven here.

All22 model-free controls pass on hosted
[38882346793](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38882346793)
at `4c4b3ee4`; replay reports `frozen_derivatives_exact=true` with all hashes
unchanged. [The retained evidence](../tests/results/ocr-quality-prepared.json)
includes the full replay/report SHA and successful
[secret scan](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38882346446).
[F16 audit38882488689](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38882488689)
at the same source is running; no held-out model-quality conclusion is claimed.
