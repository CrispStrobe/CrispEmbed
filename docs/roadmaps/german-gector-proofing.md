# German GECToR: conditional Q4_K backend

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: roadmap only; native implementation waits for Dart qualification.
Base: `b34a132c6c204b71f5c58ba8bc1705b8769edb9b`.

The [canonical Quds roadmap](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/roadmaps/german-gector-proofing.md) owns the frozen model,
backend decision, acceptance criteria and application integration.
Coordinate this bounded lane through PLAN.md; follow AGENTS.md/CLAUDE.md.

## Activation and scope

First benchmark onnx_runtime_dart and its separately named ONNX INT4 candidate.
Activate this native lane when measured latency/memory or model-distribution
requirements justify it. Reuse the shared Dart word preparation, alignment,
edit decoder and guards. Implement only inference and exact conversion here.

- Convert the pinned Proofbird v2.1 fine-tuned BERT encoder and BOTH trained
  heads: logits_labels[1,T,5000] and logits_d[1,T,2].
- Start from exact source tensors, preserving transpose/layout, tokenizer,
  special positions, layer normalization and GELU behavior.
- Expose prepared IDs with explicit sequence/mask semantics. Existing single
  sequence BERT raw execution assumes unpadded input; validate all-ones masks
  unless proper masking is implemented.
- Reuse run_encoder_raw and ggml matrix multiplication for both classifiers.
  Preserve raw/debug outputs for parity; return compact predictions normally.
- Add a default-off value-parsed core_env gate and document it if runtime code
  is introduced. Keep existing inference paths unchanged.

## Q4_K recipe and gates

Q4_K is the preferred candidate for eligible encoder matrices. The existing
native quantizer supports it and tensor-type overrides. Preserve sensitive
embeddings, normalization/bias tensors and the two output heads at F16 or Q8_0
when required; record every tensor's actual precision. ONNX MatMulNBits is a
different format and is not a Q4_K import path.

First prove an exact F32/F16 conversion against frozen original ONNX inputs,
both heads and final decoded corrections. Then compare Q4_K/mixed precision
against that control with no new clean-control edits or lost baseline fixes.
Do not qualify from cosine, file size or tokenizer identity alone.

Run builds/model batches on hosted CI in accordance with repository guidance.
Keep large GGUF/raw reference artifacts remote and retain small hashed receipts,
all failures, earliest differing stages, corrected text, latency and RSS.
No release, model publication, registry default, consumer pin or main merge is
part of roadmap publication. PLAN records open work; HISTORY/PERFORMANCE receive
completed evidence. Existing source guards and unrelated lanes stay intact.
