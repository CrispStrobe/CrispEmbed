---
license: apache-2.0
base_model: mrm8488/embeddinggemma-2-ft-typed-decision
base_model_relation: quantized
pipeline_tag: text-classification
tags: [gguf, crispembed, embeddinggemma, typed-decisions, calibration]
---

# EmbeddingGemma 2 fine-tuned for typed decisions — CrispEmbed GGUF

Complete encoder plus schema-conditioned head from
[mrm8488/embeddinggemma-2-ft-typed-decision](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision),
pinned to fcc165b7d7d97827e241e09752e83fd11165a74b. Both this fine-tune and its
[Google base](https://huggingface.co/google/embeddinggemma-2) declare Apache-2.0.
This custom GGUF architecture requires CrispEmbed; stock llama.cpp support is not claimed.

| File | Bytes | Precision |
|---|---:|---|
| embeddinggemma-2-ft-typed-decision-f32.gguf | 1102182464 | original FP32 weights |
| embeddinggemma-2-ft-typed-decision-q8_0.gguf | 307492928 | Q8 encoder matrices; FP32 head/norms/scalars/output projection/calibration |

CPU validation: five every-layer comparisons with exact token IDs, norms and
embedding checks against actual original Python FP32, plus API/CLI/HTTP/schema/head
and decoded typed-value contracts. FP32 decision-bank embedding minimum cosine:
0.999999940; maximum option probability error:
4.82762369e-06.
Q8 embedding minimum cosine: 0.999956429.
Q8 agrees on 8/8 reference labels;
its measured probability changes are in results/parity-q8.json. This is a small
numerical reference suite, not an independent task accuracy benchmark. No GPU or speed claim.
The exporter does not offer FP16 because the publisher reports overflow in that dtype.

Build CrispEmbed at 17c08b3c25da81940ddb3d1a6634cf54c6dd50f6 or a later commit including this backend, enable the CPU reference path,
and use your downloaded GGUF path:

```sh
CRISPEMBED_GEMMA_DECISION=1 crispembed -m model-f32.gguf --decide '{"state":{"task":"Delete old branches","constraint_violations":1},"questions":{"review":{"type":"noul","instructions":"Does this require human review?"}}}'
```

For Q8 also set CRISPEMBED_GEMMA_DECISION_F32_DOT=1 to reproduce the reported
arithmetic. Native inference adds the original classification prompt automatically,
mean-pools the projected token outputs and applies the trained head plus per-type
temperatures. Never substitute base-model embeddings, retrieval prompts or plain
cosine scoring for the full typed-decision pipeline.

Python: `CrispEmbed(model_path).decide(state, questions, workflow="...")`.
Questions support choice, score and noul (boolean); responses preserve string,
integer and boolean values, confidence, calibrated option probabilities and score
expectation. Rust, Dart, the C ABI and POST /decide expose the same pipeline.

[Usage and limits](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/embeddinggemma2-typed-decision.md).
results/ includes full reports, source/workflow revisions and artifact SHA-256 hashes.
[Python layer references](https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/tree/main/gemma-decision)
are hosted separately. The base model is a multilingual backbone; the head was
trained on English schemas. Reference parity is not validation on your own tasks.
