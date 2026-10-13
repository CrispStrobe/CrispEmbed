# EmbeddingGemma 2 typed decisions

Status: opt-in CPU implementation passes original-Python validation on Linux
ARM64 and macOS. Publication checks are running; download aliases are not yet
registered. Reference agreement is measured separately from task accuracy.

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
Build from the feature branch while integration is pending. Convert a locally
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
gh workflow run heavy-cpu.yml --ref feat/embeddinggemma2-typed-decision \
  -f script=tools/ci-heavy/gemma_decision_validate.py \
  -f 'pip=torch transformers==5.19.0 gguf safetensors numpy sentencepiece'
```

[Linux ARM64 run 38352165925](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38352165925)
passes all 145 stage comparisons (five probes × 29 stages), including exact IDs
for the 549-token window probe. Across stages, maximum relative L2 error is
1.481e-5; the final normalized embeddings differ by at most 2.17e-6 relative L2.
Decision-bank embeddings and complete typed answers give:

| Artifact | Bytes | Minimum bank cosine | Answers matching Python | Maximum probability error |
|---|---:|---:|---:|---:|
| FP32 | 1,102,182,464 | 0.999999940 | 8/8 | 0.00000482762 |
| Q8_0, FP32 activation operands | 307,492,928 | 0.999956489 | 8/8 | 0.00282493224 |

These eight questions cover choice, score and boolean schemas; they are a
numerical reference suite, not an independent accuracy benchmark or calibration
dataset. The checkpoint's fitted temperatures are retained without refitting.
Actual API/CLI/HTTP, schema rendering, head-only scores, invalid requests and
repeat calls pass. Independent
[macOS run 38352829721](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38352829721)
also passes all 145 stages and eight questions: the same minimum bank cosines,
maximum FP32 probability error 3.87479e-6 and Q8 error 0.00282531946.
[Recorded measurements](../tests/results/gemma-decision/arm-validation.json)
and [decoded FP32 answers](../tests/results/gemma-decision/arm-f32-decisions.json)
retain the original/native results. Publishing and SHA-256 registry pins follow
the remaining checks. This text-only model does not belong in the OCR
orchestrator or CrispCalc OCR catalog.
