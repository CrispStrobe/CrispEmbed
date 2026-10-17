# UltraBERT native-support feasibility — 15 October 2026

**Feasible, but not currently supported.** The first useful port is the retrieval
checkpoint UltraBERT-Embed-150M, followed by the150M and380M masked encoders.
They need an explicit architecture path; treating them as stock BERT, ModernBERT
or an unmodified bidirectional decoder would give different math and masks.
The initial feasibility audit read public metadata, tokenizer assets and Python
source only. Conversion/reference implementation is now in progress, as described
below; no native support or measured parity is claimed.

## Pinned upstream sources

| Checkpoint | Revision | Published tensor parameters | Width / FFN / heads |
|---|---|---:|---|
| [UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/tree/367d5343b5a0164d57b810cedf5de4b8983d5984) | `367d5343b5a0164d57b810cedf5de4b8983d5984` | 151,032,576 | 768 /1024 /6 |
| [UltraBERT-150M](https://huggingface.co/dmis-lab/UltraBERT-150M/tree/f821af159246ab4ac84669d42533281366bc55ab) | `f821af159246ab4ac84669d42533281366bc55ab` | 189,960,960 | 768 /1024 /6 |
| [UltraBERT-380M](https://huggingface.co/dmis-lab/UltraBERT-380M/tree/7789071ca161cb99e03ec1006655d24ad8c1b0e1) | `7789071ca161cb99e03ec1006655d24ad8c1b0e1` | 429,966,336 | 1024 /3072 /8 |

All three cards declare Apache-2.0, and the authors explicitly describe the
family as Apache-2.0 in their [release article](https://huggingface.co/blog/dmis-lab/ultrabert).
The pinned file lists contain no standalone LICENSE; retain the declaration and
standard license/notice provenance before distributing converted artifacts.
Parameter totals come from HF safetensors metadata, not loaded weights. The base
model names count the encoder: the serialized MLM checkpoints additionally hold
an untied vocabulary head and two prediction projections. Published benchmark
claims are upstream results and have not been reproduced in CrispEmbed.

## Actual inference contract

The [published implementation](https://huggingface.co/dmis-lab/UltraBERT-150M/blob/f821af159246ab4ac84669d42533281366bc55ab/modeling_ultrabert.py)
and configuration specify24 pre-norm layers, head dimension128, ordinary weighted
RMSNorm epsilon1e-6, bias-free Q/K/V/O projections and SwiGLU
`down(silu(gate(x))*up(x))`. There is a final RMSNorm and no embedding LayerNorm,
absolute position table or Q/K normalization. Attention is bidirectional.

Layers0/1 use local attention with radius128 **on each side**, inclusive; layer2
is global, repeating through layer23. Local RoPE theta is10,000 and global theta
2,560,000. RoPE uses split-half rotation. Eager masks combine the local band with
the key padding mask. FlashAttention's forwarded local-window argument differs
by one in the Python adapter; compare boundary behavior instead of guessing from
the argument name. Existing ModernBERT code in CrispEmbed uses a different first
global layer and interprets its configured window as a full width, so aliasing
the metadata would silently change both scheduling and locality.

The tokenizer is GPT-2 ByteLevel BPE:49,152 tokens/48,900 merges, no normalizer,
no prefix space, ordinary ByteLevel regex. A single sequence is
`<|endoftext|> A`; a pair is `<|endoftext|> A <|sep|> B`, with no terminal EOS.
BOS/EOS/pad share ID0; separator is15 and mask16. The actual BOS remains attended
despite sharing the pad ID; preserve the explicit mask/sequence lengths. The
declared32,768-token limit is not proof that a native32K path fits or is correct.
The embedder's tokenizer additionally serializes right-side LongestFirst
truncation at32,768; the base tokenizer has no serialized truncation rule. Their
vocabulary/merges are equal but their file hashes differ. Preserve and test that
distinction, including an explicit error policy for any smaller native bound.

The embedder's [module graph](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/367d5343b5a0164d57b810cedf5de4b8983d5984/modules.json)
contains Transformer plus **CLS pooling**, with prompt tokens included and no
Normalize module. Preserve raw CLS and optional L2-normalized output separately.
Its default prompt name is null. The named retrieval-query prompt is
`Instruct: Given a question, retrieve passages that answer the question\nQuery: `;
documents have no prefix. Do not silently prefix ordinary encode calls or replace
CLS with mean pooling. Task-specific instructions and symmetric-task behavior
remain explicit caller choices.

The base masked-LM checkpoints apply
`cls(gelu_tanh(linear2(gelu_tanh(linear1(hidden)))))`, with a separate untied
vocabulary projection. Training used decoupled heads, but the published inference
class exposes one head; do not invent an inference-time masking-ratio selector.
Generic classification/QA wrapper classes are not trained task heads merely
because they appear in `auto_map`.

## Fit with CrispEmbed

`src/decoder_embed.cpp` already supplies RMSNorm/SwiGLU, bidirectionality and
split-half RoPE primitives. `src/crispembed.cpp` supplies encoder pooling,
tokenizer and local-window infrastructure. Neither path supplies this complete
contract unchanged: the decoder embedder currently passes one attention mask
across layers; the ModernBERT encoder has different normalization, global-layer
offset and window semantics. Prefer a distinct `ultrabert` profile with explicit
per-layer window/theta metadata, sharing primitives rather than changing existing
model defaults. Do not route it through the causal decoder or Qwen3.5 GatedDeltaNet.

Suggested scope is several engineering days for initial bounded CPU embedding
parity, then additional head, binding, quantization and long-context work. This
is a planning estimate, not measured completion time or a claim of native support.

## Approximate storage

Shape-based decimal estimates, excluding tokenizer/GGUF overhead and unvalidated
precision retention:

| Artifact | F16 matrices approximately | Q8 approximately | Q4_K lower estimate / critical precision allowance |
|---|---:|---:|---:|
| Embed-150M | 0.302GB | 0.160GB | 0.085GB / roughly0.10–0.11GB |
| Base150M with MLM head | 0.380GB | 0.202GB | 0.107GB / roughly0.14–0.15GB |
| Base380M with MLM head | 0.860GB | 0.457GB | 0.242GB / roughly0.29–0.30GB |

Q8 uses34 bytes/32 weights; Q4_K144 bytes/256 weights. Norms remain floating
point; preserving embedding/head/prediction tensors can increase sizes beyond
these allowances. No GGUF was produced. A dense32K F16 mask alone is2GiB, before
activations, scores, workspaces or weights; model storage is not runtime RAM.

## Pending implementation and acceptance

- [ ] Freeze the full source/base/license/tokenizer provenance and inspect the
  actual weight schema before selecting tensor mappings or dropping any heads.
- [ ] Add a strict GGUF converter with per-layer windows/thetas, token templates,
  pooling/query metadata and fail-closed shape/type coverage.
- [ ] Add a default-off `core_env::on` native architecture path; preserve all
  existing encoders and literal0 behavior. Bound supported context honestly.
- [ ] Run the inspected original custom Python source at its pinned revision on
  hosted CI, initially F32 eager. The source was tested upstream with
  Transformers5.16.1/Sentence Transformers6.0.1; pin compatible dependencies
  separately from the existing Qwen5.5.4 environment.
- [ ] Compare exact IDs, input embeddings, every attention/FFN/residual/final
  norm, raw CLS, optional normalized vectors and query/document handling.
  Include padding/batching, Unicode, empty text, pairs, repeated/error cases,
  and window distances127/128/129 plus the first global-layer transition.
- [ ] Add full-vocabulary/multi-mask head comparisons and decoded predictions
  for base models; do not interpret generic wrapper logits as trained classifiers.
- [ ] Wire C ABI, CLI/server and Python/Rust/Dart; publish only after broader
  reference/quality checks, quantization calibration and remote hash verification.
- [ ] Validate long context and memory incrementally on appropriate resources;
  no32K, GPU, speed or MTEB-reproduction claim before measured evidence.

The text-only family does not belong in the OCR orchestrator or OCR catalog.
[Machine-readable source audit](../tests/results/ultrabert/source-audit.json)
records immutable revisions, metadata totals and inspected-file hashes.

## Conversion/reference lane — 16 October 2026

The pinned retrieval safetensors header contains218 F32 tensors and151,032,576
parameters, with no prediction head. The file's published LFS identity is
604,153,320 bytes, SHA256
`da8be29564b69e4f90fc8f2683e1f369c3d3575e75f7fe798643e28de02e2b9c`.
The exact header hash, shapes, types and offsets are frozen in
[the weight schema](../tests/results/ultrabert/embed-weight-schema.json).
Payload hashing is a hosted conversion gate, separate from header inspection.

[The converter](../models/convert-ultrabert-to-gguf.py) accepts only a local
snapshot at the pinned revision. It checks every inspected source/tokenizer hash,
full weight-file hash and complete218-tensor schema before loading weights; it
rejects extra heads, missing tensors, changed shapes/types and overlapping payloads.
It emits the distinct `ultrabert` architecture, per-layer radius/theta arrays,
split-half layout, full tokenizer/templates and explicit CLS/query metadata.
F32 and F16 matrix storage retain F32 norms. Neither format is yet a validated
native artifact. No runtime path, registry entry or release is added by this lane.

[The original-Python dumper](../tools/dump_ultrabert_reference.py) observes the
unchanged hash-checked custom source at F32/eager, with forward hooks on all24
layers: actual masks/positions/RoPE tables, norms, Q/K/V projections, attention,
FFN and residual outputs, final hidden state, raw CLS and optional L2. Every
hooked output must equal an ordinary unhooked inference exactly. The frozen
cases cover empty/BOS, document, explicit query, Unicode, pairs, local-window
boundaries and a padded batch. Actual original masks must enforce inclusive
127/128/129 boundaries, the first global layer and attended BOS despite its
shared pad ID. GGUF stage names feed `crispembed_diff::Ref`; post-RoPE Q/K and
additional operator probes may be needed when native attribution begins.

The [hosted driver](../tools/ci-heavy/ultrabert_reference.py) verifies complete
F32/F16 tensor round-trips and saves original reference archives remotely.
Small [model-free negative controls](../tests/test_ultrabert_contract.py) run
before downloads and in [contract CI](../.github/workflows/ultrabert-contract.yml).
Hosted results are pending. Reference captures and storage round-trips do not
establish native parity, SentenceTransformer wrapper parity, retrieval quality,
32K support, speed or quantized quality. The512-token reference guard rejects
overflow without silently truncating; it is not a native context claim.

Native integration must explicitly preserve CLS: the existing decoder loader's
`pooling_type` fallback routes CLS to last-token pooling. Its graph also supplies
one mask across layers. Its per-layer theta scheduling can inform a distinct
profile, but does not supply UltraBERT's complete contract. The converter writes
CrispEmbed's recognized `gpt2` pre-tokenizer name and both the standard
`separator_token_id` and existing `sep_token_id` compatibility key, both15; exact
original pair IDs remain the acceptance gate. Reusing loader primitives does not
justify accepting the architecture before the default-off native gate exists.

Current hosted validation (17 October): the original model dispatch
[38969507642](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38969507642)
failed before inference because the initial safetensors0.7.0 pin conflicts with
Transformers5.16.1's >=0.8.0 requirement. The
[retry38997289196](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38997289196)
uses the same immutable source0dddbd2b with `torch==2.14.0 transformers==5.16.1
gguf==0.19.0 safetensors==0.8.0 numpy==2.2.6`. No model settings, pixels, masks,
reference code or gates change. Model results remain pending. PR69 stays draft
until actual conversion/reference evidence is reviewed.

## Prepared-ID ggml prototype — validation pending

The original14 source/schema controls pass in
[38969505679](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38969505679)
at `0dddbd2b61740c0f25482b2afb428b815b5c85ac`; the actual log reports14 tests.
The corrected original-model retry remains pending; no reference inference or
native cosine result is claimed.

[The test-only graph](../tests/test_ultrabert_diff.cpp) implements all24 pre-norm
blocks, per-layer local/global masks and split-half RoPE, RMSNorm/SwiGLU, final
norm and raw/optional normalized CLS. It takes the original prepared IDs and
explicit masks, including padded query positions. `CRISPEMBED_ULTRABERT=1`
enables this executable; unset and literal0 are checked before model IO. It is
not compiled into a production model backend and adds no public API/registry.

[The hosted native driver](../tools/ci-heavy/ultrabert_native.py) checks that every
original/conversion source byte matches the immutable pre-port source0dddbd2b,
then compares292 numerical stages and50 exact input/mask/position controls per
case using `crispembed_diff::Ref`. Numerical gates remain0.99999 cosine and0.002
maximum absolute error, with no environment override. Reports retain mean/RMS,
both norms and the first failure in original execution order. All F32/F16 cases
are collected before failed numerical gates return nonzero. Eight additional
model-free controls reject missing/duplicated stages, forged green verdicts,
incorrect totals and negative errors, while retaining nonfinite/exact failures.
All22 controls and the hosted prototype build pass;
[the retained build record](../tests/results/ultrabert/native-build.json)
separates compilation/gates from unmeasured model parity.

This prototype covers every block's projection, attention, normalization, FFN
and residual outputs. It does not yet expose the48 original RoPE-table stages
or post-RoPE Q/K; extend original probes if attribution requires them. Prepared
IDs do not establish native tokenization, prompts, public encoding or independent
retrieval quality. Production integration and bindings remain separate work.

Full native runs additionally require `--frozen-report` pointing to the reviewed
complete original audit at0dddbd2b. All original reference arrays/metadata and
complete conversion records must replay exactly before native comparisons start.
The full path fails closed while that original job has not completed; build-only
CI exercises compilation and default/unset/literal0 gates without model inference.
Do not waive a reference replay drift caused by a different CPU backend.
