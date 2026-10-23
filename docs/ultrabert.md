# UltraBERT support contract — updated 23 October 2026

**Experimental default-off CPU/F32 text support is merged; general release/model qualification remains pending.** The first useful port is the retrieval
checkpoint UltraBERT-Embed-150M, followed by the150M and380M masked encoders.
They need an explicit architecture path; treating them as stock BERT, ModernBERT
or an unmodified bidirectional decoder would give different math and masks.
The initial feasibility audit read public metadata, tokenizer assets and Python
source only. Conversion and original-reference verification are complete, as described below.
The prepared-ID native prototype has a complete initial parity audit with
retained intermediate failures. PR74 adds the bounded public backend below; numerical audit failures remain.

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
safetensors==0.8.0 numpy==2.2.6`. These tools and the gated CPU/F32 architecture dispatch are integrated on main;
remaining acceptance work is in the executable handoff.

## Current test-only implementation and evidence

The default-off `CRISPEMBED_ULTRABERT` test graph implements all24 blocks,
per-layer masks/thetas, split-half RoPE, RMSNorm/SwiGLU, residuals and CLS.
Unset/literal0 are checked before IO. Each case compares292 numerical stages
and50 exact controls with `crispembed_diff::Ref`;48 original RoPE tables and
post-RoPE Q/K are not yet compared. That prepared-ID audit added no production
dispatch; the later PR74 public path is documented below.

| Surface | Result | Evidence |
| --- | --- | --- |
| Original source/conversion | 7 cases/2,730 captures,218 tensors per F32/F16 export pass | [reference report](../tests/results/ultrabert/reference-audit.json) |
| Initial native panel | All14 arms/700 exact controls complete;183 F32/1,294 F16 numerical failures retained | [initial native](../tests/results/ultrabert/initial-native-parity.json), [PERFORMANCE](../PERFORMANCE.md) |
| Original-input operators | F32 local checks41/42 pass; F16 retains24 failures; ordinary metrics/exits unchanged | [operator report](../tests/results/ultrabert/operator-audit.json) |
| Isolated original replay | Rejects empty gate anchor drift2.0265579223632812e-6 before new native comparisons | [initial capture failure](../tests/results/ultrabert/initial-linear-capture-failure.json) |
| Full original replay | Hook neutrality passes;287 empty-case old payloads differ, first Q0 error5.960464477539062e-7 | [all full-replay differences](../tests/results/ultrabert/full-forward-capture-failure.json) |
| Native BPE/test template | All8 rows exact IDs/masks; terminal-SEP and zero-ID-padding controls each fail8/8;4 log controls pass | [39007901207](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39007901207), [report](../tests/results/ultrabert/tokenizer-audit.json) |
| Same-input local operators | 39009703362 at f6f84aba completes14 arms; all21 F32 local probes pass, six F16 down failures; ordinary failures unchanged | [run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39009703362) |

The strict replay failures do not establish a CPU/backend cause. Preserve their
original arrays, source identity and exact checks. The new `linear_local_audit`
mode calls original SiLU/down modules on fixed frozen gate/up inputs, providing
identical replayed products to Python/native down. `local_down` is a separately
labelled local operator target, never a replacement for frozen FFN/model output;
its discrepancy with that old output is reported. Strict replay alternatives,
ordinary model comparisons and all original failures remain intact.

[The native driver](../tools/ci-heavy/ultrabert_native.py) verifies the original
producer, report/archive/tensor hashes and byte-exact regenerated conversions
before comparisons. `--frozen-artifacts` avoids recomputing original arrays;
strict Python replay remains a separate alternative that fails on drift. The
[manual workflow](../.github/workflows/ultrabert-native.yml) uses read-only
contents/actions access and keeps arrays remote. Do not treat a reference or
local-component success as full-model/independent retrieval quality.

[Native tokenizer test](../tests/test_ultrabert_tokenizer.cpp) uses the real
BPETokenizer with an explicit test-only UltraBERT template. The generic BERT pair
API adds a terminal separator; the later PR74 single-text profile explicitly
selects GPT-2 splitting and CLS while leaving public pairs excluded. The passed metadata-only fixture is not public
GGUF model loading. Initial compile/API failure remains retained alongside its
successful repair. Exact inputs/gold were unchanged.

[Executable U1–U4 tasks](next-agent-handoff.md) define the next result review,
numerical attribution, production surfaces and independent quality/publication
gates. All model/tokenizer/control claims must retain their exact source/run
scope; no32K, GPU, latency, MTEB or quantized-quality claim follows from this panel.

## M1 evidence and the bounded F32 output lane — 23 October2026

The [completed M1 audit](../tests/results/ultrabert/m1-validation.json) runs the
original Python CPU model against frozen Python, with pinned source/dependencies,
identical selected weight/input hashes and all four arms/seven cases retained.
Nonempty cases first differ at original RoPE cosine by5.96e-8; long/padded
final-hidden error reaches0.005791664 (Linux0.007454872). All arms are identical;
MKLDNN is unavailable. Raw CLS max6.52e-5/normalized6.41e-7 are much smaller.
[Full producer JSON](../tests/results/ultrabert/reproducibility-m1.json) and
[independent structural review](../tests/results/ultrabert/m1-independent-review.json)
retain the evidence; no new inference was run during report review.

This supports distinguishing original-platform reproducibility from native
implementation defects. It does not prove that every failed native stage is
benign, identify one backend cause or establish task accuracy. Preserve the
original strict replay and0.99999/0.002 full-model audit without new references.

Existing native F32 pooled outputs pass both raw/normalized fixed-vector gates in
all seven cases: max6.72e-5/5.94e-7, reported cosine1.0. F16 raw CLS max0.0185
fails all seven cases, despite its normalized cosine0.999999642. Do not combine
these precision conclusions or promote F16 from cosine alone.

The merged default-off **F32 pooled-embedding profile** separately qualifies:
actual text IDs/masks, prompts/normalization, raw and normalized CLS, repeat/batch/
context isolation, C API/Python/CLI/HTTP and actual Rust/Dart model execution
(39979622930, independently reviewed). Independent retrieval inputs/gold were frozen before the completed original-
first wrapper/native smoke. Keep
all183/1294 intermediate failures visible. Despite bounded API/smoke success,
this remains an experimental integration lane, with no supported registry alias,
release or HF quantized publication. Further original attribution can continue
independently using a distinct hypothesis, without repeating the completed arms.

## Experimental public F32 text profile — merged and bounded checks pass

PR74 merged at3f3d023a after all source4be9f864 checks, including ten Build/
Flutter39943915528 jobs. It adds a default-off CPU/F32 backend to normal
`crispembed_init`/`encode`/`encode_batch` and per-token calls. Enable
`CRISPEMBED_ULTRABERT=1`; unset/literal0 reject this architecture. No registry
alias, artifact publication, release or default-model change is included.
The backend validates the pinned architecture/source, full218-weight F32 schema,
per-layer schedules and tokenizer contract before allocating the weight backend.
Native context is512 tokens including attended BOS; overflow is rejected rather
than truncated. F16/quantized, GPU, Matryoshka, inputs containing the literal `<|` prefix
and public pair embedding remain outside this bounded profile.

Ordinary public dense encoding is L2-normalized, as required by the shared ABI.
Original SentenceTransformer raw CLS remains separately accessible through
`crispembed_encode_tokens_raw`'s first row; Python uses
`ids, hidden = model.encode_tokens(text, normalize=False)` then `hidden[0]`.
Documents/ordinary encode have no implicit prefix. Set the declared query prefix
explicitly through `set_prefix`; the existing CLI auto-query-prefix convention
applies, with `--prefix ""` selecting unprefixed documents. Python/Rust/Dart,
CLI and server use their existing generic ABI routes without a new model class.

The hosted `ultrabert-api.yml` workflow verifies original preserved arrays and
exact regenerated conversion bytes before running public text/ID and pooled
`crispembed-diff` comparisons. Six original single/batch cases (seven rows) are
covered; the original pair case remains in the separate prepared-ID audit.
Repeated calls, two contexts, explicit prefixes, overflow/failure recovery,
model-free schema controls, Python binding, CLI and HTTP surfaces are included.
Actual Rust/Dart39979622930 passes all seven frozen rows/42 pooled checks and
consumer/gate/error contracts; its test-integration PR75 awaits final CI. Public
pair support remains separate.
All fixed pooled gates remain0.99999/0.002. The original full-stage audit and its
183/1294 failures remain unchanged; this new test does not relabel them as passed.
API39943940607 passes; original-first SentenceTransformer/native39950831201
also passes the owned8-query/24-document smoke with all three metrics1.0,
all32 IDs/64 vectors and8/8 top1 equal. Raw/normalized maxima7.34e-5/5.51e-7,
minimum cosine0.999999999999019. [Full scores/review](../tests/results/ultrabert/retrieval-smoke.json).
Actual Rust/Dart execution passes in the separately reviewed
[wrapper audit](../tests/results/ultrabert/public-bindings.json). Representative
retrieval and release/publication
remain separate gates; the authored smoke is not broad benchmark evidence.
