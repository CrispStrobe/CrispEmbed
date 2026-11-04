# OCR benchmark expansion — 4 November 2026

This is the owner-authorized implementation follow-up to the current
OmniParseBench / OmniDocBench model review. Source support, port fidelity,
document accuracy and published consumer support are separate states. No upstream
score below is a CrispEmbed measurement.

## Verified source checkpoint

All15 hosted workflows pass on
[`7ac5b594`](https://github.com/CrispStrobe/CrispEmbed/commit/7ac5b594e63a34b53a753d9520fccff4ceb888d0),
including the full native desktop/mobile build, both WASM builds and existing
browser smoke checks, Rust, and the expanded model-free controls. The
[CI record](../tests/results/ocr-benchmark-expansion/source-ci-7ac5b594.json)
retains exact run/source identities and earlier failures. All162 focused OCR
Python controls pass. The implementation was merged through
[PR138](https://github.com/CrispStrobe/CrispEmbed/pull/138) as `2f1bbdb1`.
The first five manual jobs have run; ordinary CI does not substitute for those
model-specific checks, and no new model is qualified yet.

## First hosted results

The [initial-run receipt](../tests/results/ocr-benchmark-expansion/initial-hosted-20261104.json)
records the five jobs at the same merge source, their immutable artifact
identities and the original failures. Artifacts expire on18 November2026.

| Lane | Observed result | Next action |
|---|---|---|
| MinerU2605 | Run41810161887 failed; detailed reports remain in its745MB archive | Extraction41869721715 authenticates the reports: receipt F32/F16/Q8 pass their final-logit bars; Q4 rejects144 undeclared Q4_0 fallbacks. Aligned Q4_K/Q8 retry41871894063 is running |
| LightOn3-1B | Run41810166994 failed dependency resolution before inference | Correct the incompatible Hub/tokenizers pins; rerun the same F32/F16 protocol |
| Xiaomi0 | Run41810171225 passed package checks, failed processor metadata validation | Correct list/tuple representation handling while preserving exact numeric settings; rerun metadata before inference |
| LightOn3-0.8B | Run41810175624 completed the original capture, failed frozen-original reproduction before native execution | Extraction41869721715 finds1759 exact differences:1510 CPU-control fields,248 selected probabilities and the mean. IDs/text/other fields match. Preserve the negative guard; scope a retained-capture continuation before more inference |
| Nemotron frontend | Run41810180035 passed11/11 exact geometry/packing cases and9/9 negative controls | Completed bounded milestone; native resize, positions and the neural backend remain separate |

## Reviewed retained reports and further repairs

[Extraction41869721715](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869721715)
completed successfully at `15c48029d`. Both original ZIPs were authenticated on
hosted CI; all84 selected text members were independently rehashed during review.
The690,326-byte small artifact13344312377 is SHA256
`583828f2a02c14757af2598997b81d718deddaafb07901e26202a82cb475ed6e`.
The [extraction receipt](../tests/results/ocr-benchmark-expansion/retained-extraction-41869721715.json),
[independent numerical review](../tests/results/ocr-benchmark-expansion/retained-review-20261104.json)
and [all1759 LightOn differences](../tests/results/ocr-benchmark-expansion/lighton08-original-diff-41810175624.json)
are retained publicly. No original inference was repeated by extraction.

- MinerU's receipt covers208 complete original generated positions. F32 minimum
  final-logit cosine is0.9999999999961997, F16 is0.9999999999974624 and Q8 is
  0.9993957618183006; all three native decodes match the request. Strict failures
  remain: first F32/F16 `vis_layer_31` max0.135742188/0.1875 and first Q8
  `vis_layer_0` max0.169651985; worst Q8 stage cosine0.399607867. F16's measured
  warm latency is1.331x original on this run. Same-neural-backend helper comparison
  passes exact JSON/Markdown for8 blocks/9 requests; it is separate from original
  Python neural parity. Q4 stops before its native execution: the C quantizer
  emits144 Q4_0 fallbacks for896-wide rows, correctly rejected by the declared
  parser. Correction `1bb0ddb1d` explicitly uses Q8 for those matrices, retaining24
  genuine Q4_K down projections,273 Q8 and384 F32 tensors. Every role is checked;
  the parser still rejects Q4_0. The same bounded receipt retry is
  [41871894063](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41871894063).
  Review it before dispatching the other seven frozen cases.
- LightOn0.8's original exact guard remains negative. All reference fields other
  than CPU-control metrics and selected/mean probabilities match. Maximum selected
  probability delta is0.0000017289565493738834; mean delta is about1.034e-8.
  Native never ran. Do not add ignored fields or repeat the original blindly.
  Next design a separate native-only continuation against the retained905 arrays,
  authenticating archive, source, image, prompts, caps, IDs and full score coverage.
  It must record the failed historical exact guard and keep existing original/gold
  records immutable. If old-original numerical reproduction itself is required,
  scope CPU/backend attribution with retained values before any new reference run.
- Xiaomi's processor correction passes on actual installed packages. The first
  retry [41869734698](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869734698)
  exposes reversed dimensions in the test caller, not production preprocessing.
  Correction `f505b242a` (format-only successor `3a5a9a1dc`) passes actual
  [metadata/frontend41870593556](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41870593556):
  all4 task streams,270336 resized values and540672 patch values match exactly,
  with zero image errors at the unchanged1.2e-7 patch budget. Separate two-image
  F32/F16 [inference41871383235](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41871383235)
  is executing; no neural parity or document qualification follows yet.
- LightOn1B [41869728304](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869728304)
  passes dependency checks and builds but rejects loading before any forward:
 311 missing/310 unexpected keys. The pinned framework aliases config type
  `mistral3` to LLaVA and inserts an unwanted language-model namespace. Correction
  `5297c4cc8` uses its class-specific identity-registration API for the two actual
  LightOn classes, checks exact checkpoint/model key inventories, allows only the
  declared tied-head omission, and retains strict complete-loading checks. Both
  loader source files are additionally SHA-pinned. No checkpoint or model equations
  change. The actual F32/F16 retry is
  [41871466965](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41871466965).

These are bounded case results and executing jobs, not new registry aliases or
publication. Query each exact run before retrying it; preserve every initial
negative report. Large original archives still require preservation before
18 November2026 if the raw capture is needed beyond that deadline.

## Priority and implementation lanes

| Priority | Model | Work in this expansion | Remaining evidence |
|---|---|---|---|
| 1 | MinerU2.5-Pro-2605-1.2B | Exact source/converter/runtime profile, separate default-off gate, original-first final-logit and document-helper runner; 2604 stays available | All selected F32/F16/Q8/Q4 positions, original 2.0.7 helper comparison, full panel and immutable publication |
| 2 | LightOnOCR-3-1B | Exact Pixtral/Qwen3 checkpoint profile and grounding/transcription work through the existing LightOn engine | Processor/template/patch parity, all-stage original/native output, table/box fidelity and publication |
| 3 | Xiaomi-OCR-0 | Exact Qwen3.5-0.8B profile/frontend reuse, public API task controls, pinned external PPDocLayoutV3 producer and bounded document assembly | Hosted original-first native and two-stage execution; full-page OCR alone cannot inherit the vendor's two-stage score |
| 4 | LightOnOCR-3-0.8B | [Retained full-logit policy review](lighton08-precision-next.md) is complete; F16/Q8 clear current numerical bars on the old three-page panel, Q4 does not | Prepared separate current-source portable F16/Q8 job; hosted execution on all three cases, preserved text differences and consumer evidence |
| 4 | OvisOCR3 | [Separate source-pinned Markdown task audit](ovisocr3-document-task.md) with raw/presentation output and completion controls | Independent licensed input/gold, original task-specific captures and native task admission. The old capped two-column plain-OCR failure still blocks its old qualification/publication decision |
| 5 | Nemotron 3.5 Parse | [Exact source and 510-tensor census](nemotron-parse.md); bounded native input-geometry/patch frontend work | New C-RADIO/causal mBART graph, full original/native vectors and decoded structured output, then precision/public API/publication |
| Conditional | dots.mocr | Model/base/architecture and actual supplemental terms reviewed | Project license-policy decision before implementation/distribution; then custom vision/processor work and Qwen2 decoder reuse |
| Later | LightOnOCR-3-4B | Dense Qwen3.5 source candidate | Separate 4B profile, memory/long-page/GPU protocol after small-model parity |
| Later | Infinity-Parser2-Flash / Pro | Dense Flash and larger MoE Pro tracked separately | Flash can follow dense Qwen3.5 work; Pro needs a 40-layer/256-expert Qwen3.5 MoE implementation and total-weight storage planning |
| Optional restricted | Chandra-OCR-2 / Surya 2.5 | Separate licensed product options | Explicit terms/eligibility review, isolated distribution and new integration; existing `surya-det` is not full Surya OCR |

The implemented lanes have separate protocols: [MinerU 2605](mineru25-pro-2605.md),
[LightOn3-1B](lightonocr3-1b.md), [Xiaomi-OCR-0](xiaomi-ocr0.md), and
[Nemotron frontend](nemotron-parse.md).
Every profile retains its existing family behavior and has a separate source
identity. A shape-compatible checkpoint is not automatically admitted as another
model, and a community GGUF is not proof of CrispEmbed parity.

## Benchmark interpretation

The reviewed [OmniParseBench dataset snapshot](https://huggingface.co/datasets/datalab-to/omni_parse_bench/tree/8bedaececd7f816638247782cfa44e8578a85a5c)
has separate text/table/layout tests. Its adapters choose rendering, prompts,
vendor postprocessing and retry modes, not just model weights. The reviewed
[harness revision](https://github.com/datalab-to/omni_parse_bench/tree/04d72ec396113c8227e91cbdc0b277985dede68c)
therefore belongs in any attempted reproduction.

The useful leading comparisons from that snapshot are Chandra2 90.1, dots.mocr
87.9, MinerU 2605 87.1, Infinity Pro 87.0 and Surya 2.5 85.9. Those values guide
candidate selection, not native acceptance. LightOn3-0.8B is 81.6, LightOn3-4B
82.9, Nemotron's default single mode 80.0 and Infinity Flash 78.4. Nemotron
greedy/single/triple settings differ. There is no 1B LightOn row to inherit, and
Xiaomi's text/table entries do not establish a headline layout score.

[OmniDocBench](https://github.com/opendatalab/OmniDocBench) measures a different
combination of text edit distance, table structure and formula accuracy. Do not
combine its versions, a vendor's self-reported result and OmniParseBench numbers
into one ranking. Record the exact dataset, evaluator revision and pipeline.

## Rights are attached to exact materials

| Model/material | Observed terms and integration treatment |
|---|---|
| MinerU 2605 checkpoint | [Apache-2.0 model card](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B/tree/08aaea840498d49ce16247b6263196cf02814885). Audit the model and `mineru-vl-utils` separately from the full MinerU application. It is not excluded from this permissive model lane merely because another application component has different terms. |
| LightOn3 family | [Official model family](https://github.com/lightonai/LightOnOCR) declares Apache-2.0. The 1B architecture follows LightOn2; 0.8B/4B use Qwen3.5. Exact source profiles still differ. |
| Xiaomi-OCR-0 | [Official checkpoint](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) and declared Qwen3.5-0.8B-Base are Apache-2.0. Audit the layout component and helper code as distinct dependencies. |
| Nemotron 3.5 Parse | Pinned OpenMDW-1.1 model/source and MIT tokenizer; retain applicable notices. See the [source audit](nemotron-parse.md). |
| dots.mocr | The [MIT-labelled checkpoint](https://huggingface.co/dots-studio/dots.mocr/tree/e539fbb52280393adc081b289ec597430a0f9031) also has a [supplemental agreement](https://github.com/studio-dots-ai/dots.mocr/blob/23f3e5612fb8066d4034d5ecfc8f33a9243533eb/dots.mocr%20LICENSE%20AGREEMENT). It permits commercial/proprietary/SaaS uses but adds use/privacy obligations, modified-weight attribution and an amendment/migration clause. Record it as MIT plus supplemental terms, not plain MIT or categorically research-only. No terms are accepted by this roadmap. |
| Infinity Parser2 | [Flash](https://huggingface.co/infly/Infinity-Parser2-Flash/tree/cfddc4106b0abc4706f05575a164f7dd35d09c46) and [Pro](https://huggingface.co/infly/Infinity-Parser2-Pro/tree/b27d470100514329fc6439aada8f16ccea5f9e2a) declare Apache-2.0. Pro's active-parameter count is not its storage/RAM requirement. |
| Chandra2 / Surya2.5 | Separate code and weight terms. [Chandra2 weights](https://huggingface.co/datalab-to/chandra-ocr-2/tree/af93b47dba1b47b6640c86ccf487ed2260ab9a09) and [Surya2.5 announcement](https://www.datalab.to/blog/surya-2-5) impose additional commercial eligibility/use terms; preserve separate explicit eligibility decisions. |

The existing TeleOCR `qwen-research` classification and acknowledgement, OXR
terms, and removal of the old dots.ocr registry entry remain unchanged. dots.mocr
is a distinct candidate. Indra remains public-code-analysis-only under its
existing selected scope. The benchmark dataset uses per-source terms; an open
scorer does not make every source document a permissively redistributable fixture.

## Shared evaluation and release sequence

The new [document evaluation contract](ocr-document-evaluation.md) preserves
raw/ordered text, normalized typed blocks, reading order, HTML table spans,
formulas, measured latency/RSS, and every generation call's IDs/cap/EOS status.
Forced EOS at the cap is incomplete. It exports observed HTML/blocks to the
pinned OmniParseBench shape without inventing layout for a text-only output.

For each model, freeze source/license/processor/template/corpus hashes, run the
original first, then compare native decisive logits and free generation. Use
portable CPU builds for precision decisions. Preserve raw strict errors, decoded
differences and source-quality failures even when a newer precision policy
accepts a numerical result. Retain failed artifacts and report missing annotation
or unavailable capabilities explicitly.

Only a model with the required native/task/public-interface evidence proceeds
to quantized artifact upload, empty-cache readback, SHA-pinned registry entry and
consumer validation. Source implementation does not upgrade the immutable app
runtime or imply browser/WASM support for large VLMs. Release/tag/default work
stays in its existing lanes.

## Corrected dispatch continuation — 4 November 2026

GitHub CLI authentication works. Dispatching the original PR141 head
`4d18bd178f48a8382bee6e8c163042343dab4c26` exposed a separate workflow defect:
GitHub returned HTTP422 before creating any of the three manual runs because
`runner.temp` is unavailable in job-level `env`. The zero-job push failure
[41860949379](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41860949379)
also belongs to that invalid workflow source, not model inference.

Correction `15c48029d` initializes the two evidence directories in a runner step
through `GITHUB_ENV`. It changes no model code, package, protocol, archive pin or
qualification threshold. `actionlint`1.7.7 and the26 extraction controls pass.
All three hosted jobs now exist on that exact correction source:

| Job | Run | Review gate |
|---|---|---|
| Retained MinerU/LightOn0.8 reports | [41869721715](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869721715) | Verify complete extraction and small-file hashes, diagnose the first failed stage/changed fields before another model run |
| LightOn3-1B F32/F16 | [41869728304](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869728304) | Dependency/source/native controls, original-first full panel and explicit precision/decoded results |
| Xiaomi0 metadata/frontend | [41869734698](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869734698) | Review actual metadata and frontend before dispatching separate `--inference` |

Do not duplicate these runs. Re-query their status; dispatch acceptance or green
model-free CI is not model qualification. Large archives and model execution stay
on hosted runners. The initial failure archives still expire18 November2026.

## Hosted execution queue

All commands below use the existing manual `heavy-cpu.yml` workflow and the
reviewed correction source on `fix/ocr-qualification-nov4`. The original batch
already ran; dispatch only the corrected lanes after reviewing the retained
failure evidence. Start with the declared bounded input, retain unsuccessful
evidence, then complete the frozen panel.
No command publishes models or changes runtime defaults.

The correction source fixes LightOn1B's dependency intersection with exact
Hub1.31.0 and adds resolver/freeze/pip-check evidence. Xiaomi's image processor
now compares exact channel values across the original list-to-tuple
normalization, records actual container types and checks the pinned base-class
source. Twenty LightOn controls,29 Xiaomi controls and26 archive/diagnostic
controls pass. Actual hosted retry results remain pending.

Dispatch these three bounded jobs first, with `build_mode=portable`,
`runner=ubuntu-latest`, empty extra `pip`, and the correction branch selected:

| Order | Script | Arguments | Completion needed |
|---|---|---|---|
| 1 | `tools/ci-heavy/extract_ocr_nov4_evidence.py` | Empty | Both frozen ZIPs verified and small reports copied; review MinerU's failure and LightOn0.8's exact changed fields |
| 2 | `tools/ci-heavy/lightonocr3_1b.sh` | `--formats f32,f16` | Dependency check, original/native full panel and explicit precision results |
| 3 | `tools/ci-heavy/xiaomi0_reference.py` | Empty | Actual metadata and native frontend success before the separate `--inference` job |

The extraction selector enters a dedicated Actions-read job, with no model
secrets, package installation, model execution or native build. It accepts only
the two recorded producer runs/artifacts, checks full archive size/SHA-256 and
retains bounded text members in `ocr-nov4-retained-text-<run_id>` for90 days.
Its LightOn diagnostic uses the existing exact equality and sole
`generation_seconds` exclusion; it cannot change the frozen guard or verdict.
MinerU and LightOn0.8 should not be rerun until these reports justify the next
experiment. Nemotron's completed processor job needs no duplicate execution.

The model protocols below define the remaining panel inputs after those gates:

| Lane | Script | Initial arguments | Scope |
|---|---|---|---|
| MinerU 2605 | `tools/ci-heavy/mineru2605_qualify.sh` | `receipt f32,f16,q8_0,q4_k 2048 3 1` | One frozen request, complete final logits at all four precisions, symmetric original/native helper traces; repeat as separate jobs for the remaining seven cases |
| LightOn3-1B | `tools/ci-heavy/lightonocr3_1b.sh` | `--formats f32,f16` | Three frozen images, transcription/grounding, full-vocabulary original/native comparisons and actual GGUF census |
| Xiaomi OCR0 | `tools/ci-heavy/xiaomi0_reference.py` | `--inference` after the metadata-only run without arguments | Two owned source/native controls. The external detector and document adapter need their separate frozen document experiment; no vendor benchmark claim |
| LightOn3-0.8B | `tools/ci-heavy/lighton08_portable.sh` | `inventory-table` | New current-source portable F16/Q8 replay; then `historical-receipt` and `two-column-page` at the same source commit |
| Nemotron frontend | `tools/ci-heavy/nemotron_parse_frontend.sh` | None | Eleven original-processor size/packing cases plus negative controls; no weights or neural inference |

Use `build_mode=portable` and a distinct `isolated_lane` per immutable input.
The scripts keep weights/builds in scratch and raw references/comparisons in
hosted artifacts. Their preflights declare RAM/disk requirements; an insufficient
runner must stop before model work, not reduce the protocol's input or cap.
