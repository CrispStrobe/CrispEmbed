# EmbeddingGemma 2 typed decisions

Status: complete opt-in CPU encoder/head support passes original-Python checks
on Linux ARM64 and macOS 12, including actual Python, CLI/HTTP, Rust and Dart
contracts. FP32 and Q8 artifacts are published with verified SHA-256 pins.
Reference agreement is measured separately from task accuracy.

Source: [mrm8488/embeddinggemma-2-ft-typed-decision](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision)
at `fcc165b7d7d97827e241e09752e83fd11165a74b`. Both its card and
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
declare Apache-2.0. Reference implementation: Transformers 5.19.0 and the
checkpoint's complete custom configuration/model code.

The text backbone has hidden size 512, 768-dimensional projected token outputs,
24 bidirectional layers, projection-only per-layer inputs and a learned decision
head. Local attention uses radius 512 and head dimension 256; full-attention layers
use head dimension 512 with one KV head. Query/key/value normalization and unit
attention scaling differ from Gemma3. Source norms multiply their weights directly;
the Gemma3 `1 + weight` convention does not apply.

Every state, question and option is encoded with
`task: classification | query: `, including the prompt/BOS/EOS in mean pooling.
The normalized embeddings feed a schema-conditioned prior plus gated learned
correction. Stored temperatures calibrate choice, score and boolean probabilities.
The fine-tuned encoder and head are both required; substituting the base encoder
changes decisions. This is inference support, not a training/collation API.

## Interfaces

The CPU reference backend currently requires `CRISPEMBED_GEMMA_DECISION=1`.
Build current source including this backend. Convert a locally
available pinned source snapshot to GGUF:

```sh
python models/convert-gemma-decision-to-gguf.py \
  --model "$GEMMA_DECISION_SOURCE" --output "$GEMMA_DECISION_F32" --dtype f32
CRISPEMBED_GEMMA_DECISION=1 crispembed -m "$GEMMA_DECISION_F32" \
  --decide '{"state":{"task":"Delete old branches","constraint_violations":1},"questions":{"review":{"type":"noul","instructions":"Does this require human review?"}}}'
```

Python uses the same native encoder, schema renderer and head:

```python
import os
from crispembed import CrispEmbed

os.environ["CRISPEMBED_GEMMA_DECISION"] = "1"
model = CrispEmbed(model_path)
answers = model.decide(state, questions, workflow="agent_trace_observability")
embeddings = model.encode(["text one", "text two"])
```

Questions support `choice` (criteria mapping/list), `score` (ordered levels or
numeric-key criteria mapping) and `noul` (yes/no). Answers contain `kind`, `label`,
`value`, `confidence`, `probabilities` and `expected_level`. Values are strings,
integers and booleans respectively. Score expectation uses the actual numeric
option labels, including negative/nonconsecutive labels. Multiple options are
scored jointly and repeated rendered texts reuse their embeddings within a request.

C exposes `crispembed_has_decision`, `crispembed_decide_json` and
`crispembed_decision_error` on the ordinary opaque embedding context. Returned JSON
belongs to the context. Rust provides `has_decision` / `decide_json`; Dart provides
`hasDecision` / `decide`. HTTP `POST /decide` accepts `{state,questions,workflow?}`;
`/health` includes `typed_decision: true` when active. The shared model mutex covers
serialization as well as inference. Ordinary embedding/token routes also work.

The lower-level C/Python/Rust `decision_scores` (Dart `decisionScores`) interface scores precomputed
normalized state/question/options with 768 coordinates; it returns raw logits and
calibrated probabilities. Rendering preserves JSON object order and the Python
state formatting used in training. Request bounds: 1 MiB JSON, nesting depth 64,
1–128 questions, 1–4096 options per question. Invalid requests fail rather than
returning stale answers. Per-context use is not thread-safe outside server locking.
Floating JSON values must be finite. Integer state text supports Python's default
4300-digit bound; numeric score labels fit a signed 64-bit integer. Float spelling
uses pinned Ryu conversion, retaining the project's macOS 12 / iOS 15 build
targets. List/dict string representations use CPython 3.12 / Unicode 15.0
printability; the existing tokenizer category tables remain unchanged.

## Precision and validation

The backend uses FP32 tanh-GELU rather than GGML's CPU FP16 lookup table.
`CRISPEMBED_GEMMA_DECISION_LEGACY_GELU=1` selects the table as an A/B control.
An activation regression uses nearby FP32 inputs that the table collapses, checks
the precise values against independent double-precision arithmetic and keeps
other model backends unchanged.

FP32 is the reference artifact. The exporter does not offer FP16: the publisher
reports overflows for this model in that dtype. Q8_0 conversion preserves the
head, output projection, scalar/norm/bias tensors and calibration in FP32.
`CRISPEMBED_GEMMA_DECISION_F32_DOT=1` optionally keeps activation operands in FP32
for quantized linear weights using bounded row-dequantization. Quantized quality
must be reported separately from FP32 port correctness. No GPU or speed claim.

Hosted validation: [heavy CPU workflow](https://github.com/CrispStrobe/CrispEmbed/blob/main/.github/workflows/heavy-cpu.yml),
`tools/ci-heavy/gemma_decision_validate.py`. It performs source download/conversion,
original-Python reference dumping, every-layer comparisons (including PLE), exact
token IDs, norms/absolute errors and embeddings. Probes include multilingual text,
whitespace, special/multimodal placeholder tokens and a sequence beyond the local
attention radius. It then tests exact schema rendering, the head on frozen Python
embeddings, calibrated probabilities, typed values, API/CLI/HTTP calls, invalid
inputs and repeats. Q8 is measured with explicit approximation reporting.

```sh
gh workflow run heavy-cpu.yml --ref main \
  -f script=tools/ci-heavy/gemma_decision_validate.py \
  -f runner=ubuntu-24.04-arm \
  -f 'pip=torch==2.14.1 transformers==5.19.0 gguf safetensors numpy sentencepiece'
```

[Linux ARM64 run 38352165925](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38352165925)
passes all 145 stage comparisons (five probes × 29 stages), including exact IDs
for the 549-token window probe. Across stages, maximum relative L2 error is
1.481e-5; the final normalized embeddings differ by at most 2.17e-6 relative L2.
Decision-bank embeddings and complete typed answers give:

| Artifact | Bytes | Minimum bank cosine | Answers matching Python | Maximum probability error |
|---|---:|---:|---:|---:|
| FP32 | 1,102,182,464 | 0.999999940 | 8/8 | 0.00000482762 |
| Q8_0, FP32 activation operands (portable CPU baseline) | 307,492,928 | 0.999956429 | 8/8 | 0.00282478324 |

These eight questions cover choice, score and boolean schemas; they are a
numerical reference suite, not an independent accuracy benchmark or calibration
dataset. The checkpoint's fitted temperatures are retained without refitting.
Actual API/CLI/HTTP, schema rendering, head-only scores, invalid requests and
repeat calls pass. Independent
[macOS run 38352829721](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38352829721)
also passes all 145 stages and eight questions: the same minimum bank cosines,
maximum FP32 probability error 3.87479e-6 and Q8 error 0.00282531946.
The updated [macOS 12 run 38359820734](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38359820734)
passes the same model checks plus exact Python rendering for 5,537 finite doubles,
Unicode/list representations and empty/whitespace state failures. Both updated
platform runs configure `GGML_NATIVE=OFF`; Linux's corresponding portable-baseline
metrics above come from [38359818133](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38359818133).
The final [publication run 38374087175](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38374087175)
at `17c08b3c` passes all native/reference checks, the separate Python15 renderer,
actual Rust/Dart shared-library scores, all eight typed answers, invalid requests
and repeat calls. Dart analysis is clean. The published GGUFs and original-Python
fixtures were accepted into their public repositories and their remote hashes verified.
[Recorded measurements](../tests/results/gemma-decision/arm-validation.json)
and [decoded FP32 answers](../tests/results/gemma-decision/arm-f32-decisions.json)
retain the original/native results. The
[HF repository](https://huggingface.co/cstr/embeddinggemma-2-ft-typed-decision-GGUF)
contains both validated model files, the source license/card and full reports.
[Validated artifact identities](../tests/results/gemma-decision/validated-artifacts.json)
record both exact file hashes/sizes. The default alias
`embeddinggemma-2-ft-typed-decision` downloads FP32 (1,102 MB);
`embeddinggemma-2-ft-typed-decision-q8` downloads Q8 (307 MB).
Both downloads enforce the recorded SHA-256 pins. This text-only model does not belong in the OCR
orchestrator or CrispCalc OCR catalog.

Published alias example (build current main; existing releases predate this backend):

```sh
CRISPEMBED_GEMMA_DECISION=1 crispembed -m embeddinggemma-2-ft-typed-decision \
  --decide '{"state":{"task":"Delete old branches","constraint_violations":1},"questions":{"review":{"type":"noul","instructions":"Does this require human review?"}}}'
```

For the Q8 alias also set `CRISPEMBED_GEMMA_DECISION_F32_DOT=1` to reproduce
the measured arithmetic. [Publication provenance](../tests/results/gemma-decision/publication.json)
records model/fixture revisions, workflow, source commit and both file identities.
