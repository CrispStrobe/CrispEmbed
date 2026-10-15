# UltraBERT native-support feasibility — 15 October 2026

**Feasible, but not currently supported.** The first useful port is the retrieval
checkpoint UltraBERT-Embed-150M, followed by the150M and380M masked encoders.
They need an explicit architecture path; treating them as stock BERT, ModernBERT
or an unmodified bidirectional decoder would give different math and masks.
This audit read public metadata, tokenizer assets and Python source only. No
weights were downloaded, remote Python executed or inference/benchmark run.

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
