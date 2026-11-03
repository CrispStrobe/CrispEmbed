# OCR benchmark expansion — 3 November 2026

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
Python controls pass. The new checkpoint profiles have not yet run their
original/native model qualifications; ordinary CI does not substitute for those
manual jobs. The implemented code and prepared jobs are in
[draft PR138](https://github.com/CrispStrobe/CrispEmbed/pull/138).

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

## Hosted execution queue

All commands below use the existing manual `heavy-cpu.yml` workflow and the
reviewable `feat/ocr-benchmark-integration` branch. Start with the declared
bounded input, retain unsuccessful evidence, then complete the frozen panel.
No command publishes models or changes runtime defaults.

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
