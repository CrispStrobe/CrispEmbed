# UltraBERT support contract — updated 17 October 2026

**Feasible, but not currently supported.** The first useful port is the retrieval
checkpoint UltraBERT-Embed-150M, followed by the150M and380M masked encoders.
They need an explicit architecture path; treating them as stock BERT, ModernBERT
or an unmodified bidirectional decoder would give different math and masks.
The initial feasibility audit read public metadata, tokenizer assets and Python
source only. Conversion and original-reference verification are complete, as described below.
The prepared-ID native prototype has a complete initial parity audit with
retained intermediate failures. No public native backend is supported yet.

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
Initial parameter totals came from HF metadata. The retrieval checkpoint now
also has complete hosted source/schema/payload verification. The base
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
these allowances. No quantized GGUF has been validated. Actual F32/F16 export sizes are
609,437,696/307,447,808 bytes, as recorded in [PERFORMANCE](../PERFORMANCE.md).
A dense32K F16 mask alone is2GiB, before
activations, scores, workspaces or weights; model storage is not runtime RAM.

## Pending implementation and acceptance

- [x] Freeze source/tokenizer and actual complete weight schema/payload identity.
  All218 retrieval tensors are mapped; no head is discarded.
- [ ] Complete license/notice packaging for converted-artifact distribution.
- [x] Add a strict GGUF converter with per-layer windows/thetas, token templates,
  pooling/query metadata and fail-closed shape/type coverage.
- [ ] Add a default-off `core_env::on` native architecture path; preserve all
  existing encoders and literal0 behavior. Bound supported context honestly.
- [x] Run the inspected original custom Python source at its pinned revision on
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
The hosted original reference and conversion checks pass. Native comparisons
retain183 F32/1,294 F16 stage failures. Reference captures and storage round-trips do not
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

Current verified baseline: [38997562520](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38997562520)
at `0573f1847db68914052a98341a22fceeb1358518`. All seven cases,2,730 original
stages, hook-neutrality/mask controls and complete conversions pass. The
[reviewed report](../tests/results/ultrabert/reference-audit.json) preserves all
raw fields and hashes; tensor arrays stay in the remote run artifact. The earlier
dependency and inference-tensor negative-control failures are retained in HISTORY.
The corrected environment is `torch==2.14.0 transformers==5.16.1 gguf==0.19.0
safetensors==0.8.0 numpy==2.2.6`. PR69 remains draft while native evidence is reviewed.

## Prepared-ID ggml prototype — initial parity and operator audits complete

The original14 source/schema controls pass in
[38969505679](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38969505679)
at `0dddbd2b61740c0f25482b2afb428b815b5c85ac`; the actual log reports14 tests.
The complete original model reference passes at0573f184. The first native
panel completes all14 arms; detailed metrics and retained failures are in
[PERFORMANCE](../PERFORMANCE.md).

[The test-only graph](../tests/test_ultrabert_diff.cpp) implements all24 pre-norm
blocks, per-layer local/global masks and split-half RoPE, RMSNorm/SwiGLU, final
norm and raw/optional normalized CLS. It takes the original prepared IDs and
explicit masks, including padded query positions. `CRISPEMBED_ULTRABERT=1`
enables this executable; unset and literal0 are checked before model IO. It is
not compiled into a production model backend and adds no public API/registry.

[The hosted native driver](../tools/ci-heavy/ultrabert_native.py) checks that every
original/conversion source byte matches the approved reference source0573f184,
then compares292 numerical stages and50 exact input/mask/position controls per
case using `crispembed_diff::Ref`. Numerical gates remain0.99999 cosine and0.002
maximum absolute error, with no environment override. Reports retain mean/RMS,
both norms and the first failure in original execution order. All F32/F16 cases
are collected before failed numerical gates return nonzero. Eight additional
model-free controls reject missing/duplicated stages, forged green verdicts,
incorrect totals and negative errors, while retaining nonfinite/exact failures.
The expanded24 model-free controls pass at a8b129f4, including original-manifest
byte preservation and the six-record operator parser. The expanded test-only
executable compiles and runs all14 ordinary/operator arms at the same source.

[the retained build record](../tests/results/ultrabert/native-build.json)
separates compilation/gates from unmeasured model parity.

This prototype covers every block's projection, attention, normalization, FFN
and residual outputs. It does not yet expose the48 original RoPE-table stages
or post-RoPE Q/K; extend original probes if attribution requires them. Prepared
IDs do not establish native tokenization, prompts, public encoding or independent
retrieval quality. Production integration and bindings remain separate work.

Full native runs require `--frozen-report` pointing to the reviewed complete
original audit at0573f184. `--frozen-artifacts` consumes its saved tensors directly:
producer identity, raw report bytes, all archive/tensor hashes and regenerated
GGUF records/bytes must match before native inference. This avoids changing
original arrays through a fresh CPU floating-point run. Exact Python replay is
retained as an alternative and fails on any reference drift.

The [manual hosted audit](../.github/workflows/ultrabert-native.yml) downloads
only that declared original run on a hosted runner; its token has read-only
contents/actions permissions. It then runs all14 F32/F16-case arms and preserves
every failed stage before returning nonzero. [First native audit38997900303](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38997900303)
completes all arms and fails the unchanged numerical gates after retaining all
measurements. All700 exact controls pass. F32 raw CLS cosines round1.0 with worst
error0.000067234; intermediate failures still start at layer14 FFN output.
The [separate original-input operator audit38998432137](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38998432137)
completes with unchanged ordinary metrics/exits and all84 local records retained.
F32 passes41/42 probes: Unicode whole MLP from original norm has error0.0048828125,
reduced to0.000671386719 using original gate/up. Norm probes pass on all cases;
projection-error amplification is narrowed, not a single operator defect proven.
F16 retains24/42 local failures. Next capture original SiLU/product/down inputs
and inspect projection rounding while preserving the initial frozen references.
[Raw operator report](../tests/results/ultrabert/operator-audit.json),
[magnitudes and limits](../PERFORMANCE.md).
No numerical gate is waived by final-vector, reference or build success.


The next opt-in `linear_audit` extends the hosted audit with a separate
[test-only original MLP observer](../tools/dump_ultrabert_linear_reference.py).
It feeds frozen layer14 norm values into the original model's actual MLP and
hooks its SiLU and down-projection input. Original gate/up/output anchors must
match bit-for-bit, including signed zero; hooked/unhooked outputs must be equal.
Additional arrays stay in separate remote diagnostic archives, never replacing
the frozen encoder reference. Three independent native replays then compare
SiLU from original gate, product from original gate/up, and down projection from
original product. The ordinary graph and six earlier probes stay intact.
Compilation, three weight-free capture controls, the expanded25 model-free
controls and complete linear measurements are pending at this revision.
