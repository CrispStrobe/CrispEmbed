# OCR benchmark expansion — 4 November 2026

The [current4 November qualification continuation](ocr-qualification-2026-11-04.md)
supersedes historical queued/running states and dispatch tables below. It records
actual completed panels, the Pixtral epsilon/cache corrections and remaining gates.

This is the owner-authorized implementation follow-up to the current
OmniParseBench / OmniDocBench model review. Source support, port fidelity,
document accuracy and published consumer support are separate states. No upstream
score below is a CrispEmbed measurement.

## Current coverage and qualification state

The 4 November audit checked main
[`2f748825`](https://github.com/CrispStrobe/CrispEmbed/tree/2f748825ff3b56b8f2b89d156986b94218ae28c3)
and was reconciled with main
[`17c96639`](https://github.com/CrispStrobe/CrispEmbed/tree/17c96639a5507c4cee51898b9eeb05ad58d2ebdc)
after PR141 merged. Use the [current qualification continuation](ocr-qualification-2026-11-04.md)
for the completed cache panel and precisely bounded next scopes.
The PR138 implementation is merged. Its [historical source-CI record](../tests/results/ocr-benchmark-expansion/source-ci-7ac5b594.json)
retains the earlier source and failures; current model execution is no longer
accurately described as wholly pending.

The [registry at the audited main](https://github.com/CrispStrobe/CrispEmbed/blob/2f748825ff3b56b8f2b89d156986b94218ae28c3/examples/cli/model_mgr.cpp)
already includes PaddleOCR-VL-1.6 (Q8/Q4), GLM-OCR, olmOCR-2-7B-1025,
FireRed-OCR, Nanonets-OCR2-1.5B, DeepSeek-OCR-2, MiniCPM-OCR, Unlimited-OCR-2,
SmolDocling and gated OvisOCR2. OXR also exists under its explicit license gates.
The `lightonocr` alias remains LightOnOCR-2-1B; published MinerU aliases target
2604. Exact new versions and task protocols must be checked separately.

| Existing work | Latest bounded evidence | Remaining work |
|---|---|---|
| MinerU2605 | Eight cases / 32 precision arms complete; all EOS. F32/F16 match 8/8 decodes; Q8 and mixed Q4/Q8 match 6/8 | Retained numerical and quantized-layout differences, including rotated Q4, prevent blanket qualification/publication |
| LightOn3-1B | Full cache+epsilon panel passes all 12 corrected outputs; the earlier epsilon-only F16 language-change failure remains historical evidence. Strict failures and eight capped outputs remain | Separately scoped paired CPU-default replay, then EOS/document and precision/publication gates; both correction gates remain default-off |
| Xiaomi-OCR-0 | Exact frontend and passing final-logit/decode checks on four F32/F16 arms | All four outputs hit 16 tokens; full-document/EOS and quantized qualification remain |
| LightOn3-0.8B | Native path exists; latest exact replay has probability differences with identical IDs/text | Declared retained-native continuation, preserving the failed historical guard |
| OvisOCR3 | Gated plain OCR plus separate Markdown task/source audit | Original/native document-task evidence; retain the old plain-OCR capped-page failure |
| Nemotron3.5 | Source/tensor census and bounded geometry/packing frontend | Native resize, learned positions, neural graph and complete OCR/publication path |

The following first-batch records retain their original failures and observations.
Their running/queued language and dispatch suggestions are historical; the current
qualification continuation above supersedes them. Do not repeat completed panels.

## Historical source checkpoint

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

## Historical first hosted results

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
  completed successfully. The [full report](../tests/results/ocr-benchmark-expansion/xiaomi0-inference-41871383235.json)
  and [independent review](../tests/results/ocr-benchmark-expansion/xiaomi0-inference-review-41871383235.json)
  preserve all4 arms/160 stage comparisons and16 full248320-wide logit rows per
  arm. Minimum final cosines are0.9999999999994335/0.9999999999999927 at F32 and
  0.9999999999999366/0.9999999999999754 at F16, with exact IDs/text/stops.
  Six strict stages fail: first `vision.layer.11` in every arm (max0.03125–0.0518799)
  plus `decoder.norm` in the text arms (max0.013526/0.008469). Both originals and
  all native arms hit the declared16-token cap; full-document/publication flags
  remain false. Freeze a separately reviewed EOS/document/precision protocol
  before extending caps, adding Q8/Q4 or publishing an alias.
- LightOn1B [41869728304](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41869728304)
  passes dependency checks and builds but rejects loading before any forward:
 311 missing/310 unexpected keys. The pinned framework aliases config type
  `mistral3` to LLaVA and inserts an unwanted language-model namespace. Correction
  `5297c4cc8` first suppressed that alias but its strict inventory check rejected
  the actual saved text prefix, before any forward. Inspecting the71712-byte
  checkpoint header establishes532 tensors:310 under `language_model.model.*`
  and222 vision/projection tensors under `model.*`. The exact header fixture
  SHA is028ca2de0140abe05988620d440826309657747b2387767fe6b592dc74340cfe.
  Successor `4476be6eb` maps only the saved text prefix to `model.language_model.*`
  through the framework's class-specific registration API and the candidate
  converter, preserves the vision names, and rejects unknown namespaces/collisions.
  Strict complete-loading checks and the declared tied-head exception remain.
  Both loader sources are SHA-pinned;22 controls pass. No checkpoint value or model
  equation changes. Actual F32/F16 retry
  [41873656972](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41873656972)
  remains executing; review it before any further run.

These are bounded case results and executing jobs, not new registry aliases or
publication. Query each exact run before retrying it; preserve every initial
negative report. Large original archives still require preservation before
18 November2026 if the raw capture is needed beyond that deadline.

## Priority and implementation lanes

| Priority | Model | Why it merits work | Next bounded action / gate |
|---|---|---|---|
| Finish existing | MinerU2605, LightOn3-1B/0.8B, Xiaomi, Ovis3 | Convert the existing implementation/evidence into accurately bounded support | Follow the exact outstanding protocols above; PR141 retains ownership of its current norm/cache work |
| Next new checkpoint; active | LightOnOCR-3-4B | Apache-2.0 difficult-text/handwriting/math/degraded-scan option | [Pinned source preflight passes hosted controls](lightonocr3-4b.md); next are original runtime capture and explicit dimension, grouped-head, tokenizer and memory changes |
| Highest conditional new capability | Logics-Parsing-V3 | 0.8B Qwen3.5 model with multi-page hierarchy, continuation and document state | Explicit publisher weight grant first; its Apache-2.0 code permits a separate document-state/DSL contract without loading weights |
| Next permissive custom backend | Qianfan-OCR | Document understanding, key information extraction, layout reasoning and charts | Audit its Qwen3 decoder, custom vision/projector and conflicting processor defaults; not a stock Qwen3-VL alias |
| Conditional quality/multilingual candidate | dots.mocr | Strong missing document quality candidate | Decide on actual supplemental terms, then custom Dots vision/processor plus Qwen2 decoder work |
| Footprint-driven | Nemotron 3.5 Parse / OpenDoc-0.1B | Compact deployment options if total memory and latency improve | Nemotron needs the full new neural composition; OpenDoc includes PP-DocLayoutV2 plus enhanced UniRec. Count the whole pipeline |
| Specialist / later | OCRVerse / Dolphin1.5 | Visual/code reconstruction / compact fallback | OCRVerse weight license clarification and exact text/full-model selection; Dolphin1.5 must justify a new Swin/mBART composition with measured savings |
| Opportunistic / server tier | Infinity-Parser2-Flash / Pro | Dense-family specialty / large MoE deployment | Flash after qualified dense reuse; Pro requires a 40-layer/256-expert MoE port and total-weight storage planning |
| Optional restricted | Chandra-OCR-2 / Surya2.5 | Separate licensed product options | Resolve commercial/competition terms; `surya-det` is not Surya2.5 recognition/document OCR |

The implemented lanes have separate protocols: [MinerU 2605](mineru25-pro-2605.md),
[LightOn3-1B](lightonocr3-1b.md), [LightOn3-4B source preflight](lightonocr3-4b.md), [Xiaomi-OCR-0](xiaomi-ocr0.md), and
[Nemotron frontend](nemotron-parse.md).
Every profile retains its existing family behavior and has a separate source
identity. A shape-compatible checkpoint is not automatically admitted as another
model, and a community GGUF is not proof of CrispEmbed parity.

## Benchmark interpretation

The current [OmniParseBench README/harness revision](https://github.com/datalab-to/omni_parse_bench/tree/74629e1597259105ed13ae62aded01ece65b3290)
is dated 4 November. The separately observed [dataset tree](https://huggingface.co/datasets/datalab-to/omni_parse_bench/tree/74f8d391683d69813a6c9bfbb5f2d53b273658e9)
lags the new Ovis result, so record both identities. Adapters select rendering,
prompts, helpers and retry modes. Ovis3's headline uses two calls per page;
MinerU uses layout then crop recognition. These are benchmark-maintainer results,
not CrispEmbed measurements; Datalab also supplies Chandra and Surya.

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

LightOn4B's role is supported by the [same OmniParse table](https://github.com/datalab-to/omni_parse_bench/blob/74629e1597259105ed13ae62aded01ece65b3290/README.md):
versus 0.8B, handwriting improves 57.1 to 65.4, math 87.1 to 96.4 and degraded
pages 76.8 to 87.8, while tables decline 72.1 to 70.8. Avoid a universal-upgrade claim.
The current OmniDocBench table is labelled `v1.6_full`: Qianfan is 93.90 versus
already-integrated PaddleOCR-VL-1.6 at 96.34. Choose Qianfan for extraction tasks;
Logics v2's 93.33 must not be assigned to V3. V3 multi-page results are publisher
reports. OCRVerse's listed evaluation points to the text checkpoint, not the full
visual/code model. OpenDoc's advertised recognizer size excludes its layout stage.

## Rights are attached to exact materials

| Model/material | Observed terms and integration treatment |
|---|---|
| MinerU 2605 checkpoint | [Apache-2.0 model card](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B/tree/08aaea840498d49ce16247b6263196cf02814885). Audit the model and `mineru-vl-utils` separately from the full MinerU application. It is not excluded from this permissive model lane merely because another application component has different terms. |
| LightOn3 family | [Official model family](https://github.com/lightonai/LightOnOCR) declares Apache-2.0. The 1B architecture follows LightOn2; 0.8B/4B use Qwen3.5. Exact source profiles still differ. |
| Qianfan-OCR | [Official weight LICENSE](https://huggingface.co/baidu/Qianfan-OCR/blob/main/LICENSE) and [NOTICE](https://huggingface.co/baidu/Qianfan-OCR/blob/main/NOTICE) declare Apache-2.0 with identified MIT InternVL components. Audit its actual processor path before porting. |
| Logics-Parsing-V3 / v2 | [GitHub code](https://github.com/alibaba/Logics-Parsing) is Apache-2.0. The [official V3 weight tree](https://huggingface.co/Logics-MLLM/Logics-Parsing-V3/tree/e6d324e30e0f64f1a0dfb652efcb83a4b244591e) and card have no explicit weight license; v2 also needs clarification. Third-party GGUF labels do not establish a publisher grant. |
| OCRVerse / ABot-OCR | Public code/weights do not establish unrestricted weight rights. [OCRVerse full model](https://huggingface.co/DocTron/OCRVerse) lacks a model card/license declaration; [ABot](https://huggingface.co/acvlab/ABot-OCR) also needs explicit licensing. Keep conditional. |
| OpenDoc / Dolphin1.5 | [OpenOCR code](https://github.com/Topdu/OpenOCR) and [original UniRec weights](https://huggingface.co/topdu/unirec-0.1b) declare Apache-2.0; audit the enhanced OpenDoc checkpoint and layout component separately. [Dolphin1.5](https://huggingface.co/ByteDance/Dolphin-1.5) declares MIT. |
| Dolphin-v2 / Youtu / Hunyuan | Dolphin-v2's [weight LICENSE](https://huggingface.co/ByteDance/Dolphin-v2/blob/main/LICENSE) is Qwen Research, unlike 1.5. [Youtu](https://huggingface.co/tencent/Youtu-Parsing/blob/0d8c591f63cb4eb137d1824715a1b2e5dbca23dc/LICENSE.txt) adds an EU limitation; [Hunyuan](https://huggingface.co/tencent/HunyuanOCR/blob/350e760afa32195a9c78a809b8c36b003074b662/LICENSE) excludes EU/UK/South Korea. |
| Xiaomi-OCR-0 | [Official checkpoint](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) and declared Qwen3.5-0.8B-Base are Apache-2.0. Audit the layout component and helper code as distinct dependencies. |
| Nemotron 3.5 Parse | Pinned OpenMDW-1.1 model/source and MIT tokenizer; retain applicable notices. See the [source audit](nemotron-parse.md). |
| dots.mocr | The [MIT-labelled checkpoint](https://huggingface.co/dots-studio/dots.mocr/tree/e539fbb52280393adc081b289ec597430a0f9031) also has a [supplemental agreement](https://github.com/studio-dots-ai/dots.mocr/blob/23f3e5612fb8066d4034d5ecfc8f33a9243533eb/dots.mocr%20LICENSE%20AGREEMENT). It permits commercial/proprietary/SaaS uses but adds use/privacy obligations, modified-weight attribution and an amendment/migration clause. Record it as MIT plus supplemental terms, not plain MIT or categorically research-only. No terms are accepted by this roadmap. |
| Infinity Parser2 | [Flash](https://huggingface.co/infly/Infinity-Parser2-Flash/tree/cfddc4106b0abc4706f05575a164f7dd35d09c46) and [Pro](https://huggingface.co/infly/Infinity-Parser2-Pro/tree/b27d470100514329fc6439aada8f16ccea5f9e2a) declare Apache-2.0. Pro's active-parameter count is not its storage/RAM requirement. |
| Chandra2 / Surya2.5 | Read the actual [Chandra2 LICENSE](https://huggingface.co/datalab-to/chandra-ocr-2/blob/main/LICENSE) ($2M revenue/funding thresholds) and [Surya2.5 LICENSE](https://huggingface.co/datalab-to/surya-ocr-2.5/blob/main/LICENSE) ($5M). Both add competing-product restrictions and output provisions; eligibility is a separate product decision even below a threshold. |

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

## Active source-only implementation lane

Branch `feat/ocr-next-models-20261104` / [PR143](https://github.com/CrispStrobe/CrispEmbed/pull/143)
begins with LightOnOCR-3-4B. The [executable source preflight](lightonocr3-4b.md)
now checks the actual 723 BF16 tensors, independently derived names/shapes and
contiguous byte intervals, with identity/shape/unsupported-feature negatives.
It retains the exact original metadata/header plus separate bounded tokenizer
prefix evidence. Native support, export, document quality and publication remain
unqualified. Small-file and header acquisition is distinguished from full
checkpoint hashing or model inference. [Hosted source controls](../tests/results/lightonocr3-4b/source-ci-c41469b2.json)
pass at `c41469b2`: 23 new cases, 236 tests in the full controls job and 51 CI
selection tests. Original runtime capture is the next gate.

The current [Qwen3.5 runtime](../src/qwen35_ocr.cpp) assumes 473 tensors,
24 decoder / 12 vision layers and 1024-wide text features. 4B needs 32 decoder /
24 vision layers and width 2560. Its linear attention has 16 key heads and
32 value heads, while the existing linear/delta code assumes equal head counts.
The 248320-by-2560 vocabulary matrix also exceeds `project_cpu`'s existing
element limit. The converter's historical `vision-f32` recipe covers only
12 vision layers. Parameterization, grouped-head recurrence, bounded vocabulary
projection and a distinct precision recipe require separate implementation and
original-module evidence; increasing limits or exposing an alias is insufficient.

This bounded lane can prepare source contracts and hosted control jobs without
occupying the model-qualification lane. Full builds, original/native captures,
corpus panels and large artifacts remain on hosted CI or authorized remote
compute under the repository instructions. No version/release/default change
follows from source-only tooling. Work is pushed incrementally to its feature
branch as requested by the owner.


The dispatch record below describes the initial repaired batch, not a current
queue. Its original failures remain preserved; use the current continuation for
all subsequent completion and artifact-preservation results.

## Historical dispatch correction — 4 November 2026

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

## Earlier hosted execution queue — consult current receipts first

The commands below document the original PR138 manual jobs. Several now have
completed results in PR141; use its linked continuation before scheduling work.
Do not repeat completed originals/native panels or use this historical branch
as the current source. Freeze the reviewed source and outstanding scope first.
These scripts do not publish models or change runtime defaults.

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

