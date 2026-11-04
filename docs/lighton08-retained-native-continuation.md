# LightOnOCR-3-0.8B retained-original native continuation

Prepared 4 November 2026; design only. No new driver, original forward, native
execution, qualification, publication or release is recorded by this document.
Use [the reviewed continuation](ocr-benchmark-expansion.md) and preserve
[the failed exact comparison](../tests/results/ocr-benchmark-expansion/lighton08-original-diff-41810175624.json).

## Immutable first input

The first case is `inventory-table`, alone. Its producer is
[41810175624](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41810175624),
source `2f1bbdb11c3612acf7399e31b33ac715027f3184`, workflow
`.github/workflows/heavy-cpu.yml`, event `workflow_dispatch`, completed with
**failure**. Require that exact failure rather than accidentally requiring a
successful producer. Native execution never started: `quant.frozen_original`
rejected the repeated original dictionary before the build.

Authenticate artifact `13312356863`, name `heavy-41810175624`, ZIP size
`3459070891`, SHA256
`7517525e72d29d1e535257458cc11942d3df360a6fb7b1ab0182ce8fff2138da`.
Verify GitHub run/artifact repository, source, workflow, run association, size,
digest and expiry through the Actions API before transfer; stream and hash the
whole ZIP before extraction. Use `extract_ocr_nov4_evidence`'s redirect handling,
verified streaming and safe-member checks; reject duplicate names, traversal,
symlinks, unexpected files and decompression budgets before opening GGUFs.
The observed expiry was 18 November 2026; preserve the original archive remotely
before expiry, with identical size/hash and an immutable preservation receipt.

The small extraction is independently pinned by
[retained-review-20261104.json](../tests/results/ocr-benchmark-expansion/retained-review-20261104.json):
extractor run `41869721715`, source
`15c48029d342392d5d346f5f274b5e8a0905c695`, artifact `13344312377`,
`690326` bytes, SHA256
`583828f2a02c14757af2598997b81d718deddaafb07901e26202a82cb475ed6e`.
Its `lighton08-inventory-table/archive-directory.json` and
`selected-text-sha256.json` authenticate selected text and directory observations;
they do not constitute downloaded/validated numerical arrays.

All following paths are internal to the large artifact, beneath
`lighton08-portable-inventory-table/`:

| Member | Bytes | Purpose |
|---|---:|---|
| `original/official-delta.gguf` | 2100273216 | Actual document's 905 arrays; native comparison reference |
| `original/official-reference.json` | 129730 | Actual document metadata, IDs, text, probabilities, CPU controls |
| `inventory-table.png` | 25700 | Exact image; no rendering |
| `export-manifest.json` | 163277 | Frozen F16 model export census |
| `report.json` | 149730 | Producer identity, phase and retained failure |
| `anchor-full.npz` | 878855072 | Separate original full anchor |
| `anchor-full.json` | 162076 | Anchor metadata |
| `anchor-compact/official-delta.gguf` | 42642624 | Separate compact anchor |
| `anchor-compact/official-reference.json` | 1387 | Compact anchor metadata |
| `official-delta.gguf` | 879015264 | Separate full-anchor export; never substitute for document reference |

The archive directory contains the first case's required image and document
reference; no missing-input blocker is apparent for this case. Per-member SHA256
and the actual 905-tensor census remain **unverified** until hosted archive
inspection. ZIP CRC32 and declared `captured_arrays` are insufficient substitutes.

## Historical failure and separate comparison identity

Keep the original exact guard failed, including its sole pre-existing ignored
field `generation_seconds`. The 1759 differences comprise 1510 CPU-control
metrics, 248 selected probabilities and one mean probability; maximum selected
probability difference is `1.7289565493738834e-6`, mean difference
`1.0336023148482809e-8`. All other reference fields match. Never broaden ignored
fields, rewrite historical JSON, or repeat original inference to obtain a pass.

A new receipt must explicitly identify its target as the authenticated **retained
41810175624 original**, separately from the older frozen original. Preserve the
whole historical diff and both metadata dictionaries. A native match to this
retained target does not establish reproduction of old CPU metrics/probabilities.
Record `original_forward_calls=0` and historical/current verdicts separately.

## Source closure and input barrier

Implement a new driver/protocol in an isolated worktree after merging current
main. Do not call `lighton08_portable.run()` or its shell wrapper: they execute
full/compact/document original forwards. Reuse only its named pure preparation,
conversion, environment, native and comparison helpers. Keep the protocol and
all frozen files in `tests/fixtures/lighton08-portable/protocol.json` unchanged.
Before dispatch pin the new complete source SHA, clean tracked tree, ggml Git
object, CMake/workflow, all `src` inputs, native harness/parity headers, converter,
fixture, parsers/scorers and new driver/protocol hashes. Fail before acquisition
on drift; report source amendments without weakening original pins. Reference
observers remain test targets, excluded from ordinary production artifacts.

Before native model use, retain a report of every archive member and hash, then
read the document GGUF with `gguf.GGUFReader`. Require architecture
`qwen35_delta_reference`, exactly 905 unique arrays, F32 stored type, valid
nonempty shapes/byte counts, finite contents and no duplicate/unexpected names.
Write a sorted immutable name/shape/type/bytes/SHA256 inventory and whole-member
SHA256. Validate every actual tensor name against the pinned original producer's
capture schema (`dump_qwen35_ocr_page_reference.py` and its installed vision/CPU
probe helpers); do not invent an inferred 905-name list from a count alone.
This hosted schema review/freeze is the pre-inference barrier: stop for missing
or unexplained arrays. Authenticate all 905, even probes the native harness does
not consume, and report the consumed/unconsumed inventory explicitly.

Cross-check `generated.ids`, `generated.scores.0` through
`generated.scores.247`, `input.decoded_utf8` and `input.decoded_length` against
metadata. Require each full score vector to have vocabulary 248320, all positions
present once, finite values and argmax equal to the declared original ID. Require
248 IDs, no intermediate EOS, final EOS `248046`, `reached_eos=true`, `stop=eos`,
cap1024 and empty prompt. Authenticate image SHA256
`1e7a3073bd158dc87613463e45eafc52cec0c48f3655a6937a583767e26f4b4f`,
size448×448, grid `[[1,28,28]]`, patches784, exact rendered prompt and input
IDs/pixels/geometry against actual arrays and pinned preprocessing. Also require
the historical prepared PNG and decoded RGB checks from `lighton08_portable`.
Compare raw UTF8 bytes before any text normalization. Preserve CPU controls and
selected confidences exactly as retained; recompute selected softmax confidence
from all original logits using the documented F64 normalization and compare at
the declared printed/serialization precision without creating a new budget.

## Hosted native-only first case

Use GitHub Actions `.github/workflows/heavy-cpu.yml`, `ubuntu-latest`, portable
CPU, one document per job. No shared-host build, suite, inference or large
artifact download. Check available RAM≥10GiB and scratch≥15GiB plus retained
output≥8GiB (≥23GiB if shared), and account separately for the 3.46GB original
ZIP plus extracted anchors/reference/model/build before transfer; raise the
capacity bound if those simultaneous allocations exceed it. Retain resource
reports and exact command/timeouts. Use four inference threads, two build jobs,
CPU-only Torch and pinned `fixture.VERSIONS`; record resolver/freeze and pip check.

Reuse `ovisocr2_lighton_fixture` source/license authentication and export helpers:
model `lightonai/LightOnOCR-3-0.8B`, revision
`36d8636925327c671bd8e479c517a004f7566bd9`; base `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. Verify every SOURCE_FILES member,
checkpoint `1706030528` bytes/SHA256
`ae46aa6fd7136d57297ac006ed9ec345b7a2083feb2e99501bedcb014ca3efdf`
and base license before conversion. Call frozen converter
`convert(local, baseline, fixture.MODEL, fixture.REVISION, dtype="f16",
keep_f32=fixture.receipt_vision_recipe(receipt, converter))`; use
`fixture.verify_export`. Require F16 `1918139328` bytes/SHA256
`41409fc871e76a6bc13e9988d0efd2706e06d0c7c77e2ffb8746c28348ba7bd4`.

Configure/build as the existing portable driver, with both native flags OFF,
CUDA/METAL/BLAS OFF (also explicitly Vulkan OFF), Release, targets
`test-qwen35-quant` and `crispembed-quantize`; inspect actual CMakeCache and hash
both executables. Run `test-qwen35-quant --request-controls` first.
Quantize using `crispembed-quantize BASELINE CANDIDATE q8_0 --tensor-type 'v.*=f32'`.
Require frozen Q8 SHA256
`6d5a9e94ded5207ed551704c09d87940e6e6d1f02b9a2db3571371baf5d3929a`
and recipe bytes from `page-parity.json`'s `q8-vision-f32`; use
`quant.tensor_audit` on actual baseline/candidate tensor bytes and shapes/types.
Keep protected vision tensors byte-exact; no new recipe/default/quantization.

For each model F16 then Q8, reuse `native_environment(prior)` (clear all inherited
CRISPEMBED variables then apply frozen environment; QWEN35_OCR=1,
QWEN35_QUANT=1, portable and four threads). Invoke separate processes:

```text
test-qwen35-quant MODEL IMAGE DOCUMENT_REFERENCE "" 1024 --prefill-only
test-qwen35-quant MODEL IMAGE DOCUMENT_REFERENCE "" 1024 --full-quality
```

`--full-quality` already teacher-forces the original prefix at every score
position, then invokes an independent owning context for free generation; retain
its admission/rejection/recovery controls. Literal QUANT=0 is only the frozen
negative admission control. Do not use `--scores-only` or reference embeddings
as an ordinary production substitute. Preserve observer/numerical and ordinary
free-generation results separately; do not claim an additional independent
observer-off replay unless explicitly executed and recorded.

Reuse `quality.strict_prefill` with frozen stage names; `parse(log, reference)`,
`validate(..., reference,1024)`, `add_confidences`, `teacher_quality` and
`quality.score(old.gold,...)`; require full-quality's first77 metrics equal the
independent prefill metrics. Use `review.case_review`/`summarize` only after the
new retained-source authentication barrier. Record all248 ordered teacher rows
and full-vocabulary score metrics, strict failures, absolute/mean/RMS errors,
reference/native magnitudes and earliest actual divergent stage.

First-case acceptance requires both complete arms, minimum final-logit cosine
strictly >0.98 F16/>0.97 Q8 over **all248** positions, independent free output
IDs/text exact to retained original, EOS/no cap, observer/request controls,
confidences and scored gold CER/WER/structure reported. Keep strict budgets and
every failure intact; a precision-bar pass alone does not qualify decoded quality
or waive numerical debt. Save failures even if command exit is nonzero.

## Remaining cases and stopping rule

Do not dispatch `historical-receipt` or `two-column-page` before independent
review of the complete positive inventory-table artifact and source closure.
Their expected output counts are473 and683 respectively. The authenticated
inventory-table archive contains neither remaining case's original arrays/image;
the small extraction does not supply them. **Remaining-case blocker:** identify
and authenticate exact retained archives (or immutable remote preservation)
with each complete original GGUF/metadata/image, full-position logits and source
before scheduling either job. Historical summary JSON alone cannot fill this gap.
Do not solve missing references by new original calls under this lane.

Actions API discovery on 4 November 2026 found the following concrete historical
sources. Each run-specific `/actions/runs/RUN/artifacts` endpoint returned
`{"total_count":0,"artifacts":[]}`; no surviving artifact ID, API digest, byte size
or expiry is exposed, so do not fabricate a download contract from report hashes.

| Historical producer | Source | Role |
|---|---|---|
| `38882488689` | `4c4b3ee4fb8a92bd6b6c8991452f388a6100e874` | Original three-case F16 quality panel; completed failure; original473/683/248 IDs |
| `38870347237` | `a07e83fbd02e1c8ff73b3f4f9a224ab0cba265c3` | Receipt/conversion qualification linked by `receipt.json` |
| `38674281196` | `d6c8941662e84fcb410de7838036afc7ba971e9b` | Earlier page-parity producer |
| `38928823914` | `fdf83339f0a99fdc0648eb4a9f00c261c14aae64` | Q8 historical-receipt; possible repeated original carrier, unavailable |
| `38931861995` | `a42a28b76570c9468add0affd44576fe5ee96881` | Q8 two-column-page; possible repeated original carrier, unavailable |

The exact `4c4b3ee4` workflow uploaded `heavy-${github.run_id}` for14 days.
The empty API response establishes absence now; it does not reveal whether
expiration or explicit deletion removed it. `quality.json`'s
`artifact_report_sha256=a76edcfbefa36ddb381c0a871440e82b2f560785a8319289d0f8d897cd6dc14d`
authenticates the report only, not missing full arrays.

A complete paginated repository artifact metadata listing (4340 artifacts at
inspection) found a separate surviving90-day LightOn **format1 fixture**:
producer `40046789339`, source
`da0c8bd5c13991b77238195035df781888ddb0d2`, artifact `12542163496`,
name `lightonocr3-format1-receipt-40046789339`, `1339726582` bytes,
API SHA256 `08eadc6369156359eb002e297d24e9196903bc223d5054a5c548550835b860f4`,
`expired=false`, expiry `2027-01-22T00:48:28Z`. Its small companion
`12542746015`, `lightonocr3-format1-reports-40046789339`, `2971` bytes,
SHA256 `80ec07780b95630aa7a0ae837d48b8aa0aa7715a50a258f4bec0965feed54bdd`,
has the same expiry. These are model-format/converter fixture artifacts from
`ovisocr2_lighton_fixture`, not authenticated carriers of the473/683-position
original document arrays; no raw contents were downloaded in this discovery.
They must not be substituted for either missing case or for the distinct905-array
inventory-table capture from41810175624. The other preservation matches were
Ovis/UltraBERT evidence, not LightOn three-page originals. The original-case
array blocker therefore remains after discoverable Actions metadata checks.

The independently pinned prepared PNG transport remains available as a separate
input source: `ovisocr2-quality-prepared-preservation.json` points to immutable
Hub revision `057e12668bf8109ec83c65929c9b8d6cb99ca566`, path
`ocr-quality-v1/prepared-38879605915/prepared.zip`, `724951` bytes, SHA256
`a890c3ad8151706948d3d89529e5f5716405e5f790a3749436f36d78f20ca9cc`.
This can supply authenticated historical images after the existing archive/RGB
checks; it cannot reconstruct missing original logits/activations. Metadata-only
discovery does not establish current blob readability or verify archive members.

The missing raw references block this **retained-native lane** for the other two
cases; they do not prohibit a separately declared future evaluation. After the
first inventory-table native result is reviewed, an alternative next action is
to write and review a separately named **current-CPU original capture protocol**
for historical-receipt and two-column-page. That protocol must preserve unchanged
image/prompt/cap/model-source hashes and gold; freeze exact CPU/backend/thread
identity and complete array inventories; and complete/authenticate the original
capture before native execution. It must retain the failed historical exact guard
and distinguish a current-CPU reference from historical reproduction. Attribute
source/operator CPU differences using retained evidence before selecting any
changed comparisons. Never silently substitute old gold/reference identities or
broaden ignored fields. This alternative remains pending design and review: this
retained-native plan authorizes no new original calls.

Retain original archive/reference/model hashes, new binaries/cache/source,
commands, logs, complete metrics, rejection controls and summaries remotely;
download only review-sized reports. Preserve successful first evidence when
resuming remaining cases. Aggregate only identical new source/model/environment
closures, all three cases and both precisions. Keep historical exact failure,
original strict rejections and book/receipt quantization negatives unchanged.
`current_model_qualified=false` and `publication_qualified=false` remain until
separately reviewed full task/public-surface/backend/publication gates; no alias,
runtime release or downstream app pin follows from this continuation.
