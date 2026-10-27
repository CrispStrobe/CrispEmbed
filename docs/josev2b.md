# Josev-2B source audit and prepared reference lane

The original four-case CUDA/F32 capture, package provenance audit, all251-array
preservation and exact122-tensor F32 head conversion are completed and reviewed.
[Original evidence](../tests/results/josev2b/original-review.json) and
[component conversion](../tests/results/josev2b/head-artifact-review.json) preserve
source identities and limits. Native CPU/CUDA head parity, text backbone,
composed typed outputs, task quality and release qualification remain pending.
The existing Gemma decision API does not load Josev. The source and owned inputs
are frozen in [the versioned contract](josev2b-source-contract.json) and
[four authored text cases](../tests/regression/josev2b/text-v1.json).

## Source, license and provenance

The checkpoint is [mayflowergmbh/Josev-2B at
6f0a31c8](https://huggingface.co/mayflowergmbh/Josev-2B/tree/6f0a31c8ed683aba79097eb3f48df2c9ab560afd).
Its actual [LICENSE](https://huggingface.co/mayflowergmbh/Josev-2B/blob/6f0a31c8ed683aba79097eb3f48df2c9ab560afd/LICENSE)
is Apache-2.0. Preserve that license, applicable notices and modification notices
when distributing derivatives; sections4 and6 retain attribution obligations and
exclude a trademark grant. The card names `unsloth/Qwen3.5-2B` as its base, with
merged language/vision adapters and a new Clef joint head. It does not identify the
exact training-base commit. The inspected [Unsloth source](https://huggingface.co/unsloth/Qwen3.5-2B/tree/fffbc0d8711cf87df6d1f97938c5e50ef94aadc7)
and [Qwen upstream](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
both carry the same Apache license bytes. These revisions are inspection anchors.

The published `joint_schema_model.py` is byte-identical to
[Cloudflare/clef-flash at f0b7d3cd](https://huggingface.co/Cloudflare/clef-flash/blob/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f/joint_schema_model.py),
whose Apache license was also inspected. SHA256s, Git blobs and file sizes are in
the contract. Large backbone/head/tokenizer hashes come from pinned service LFS
metadata; they have not been locally downloaded or payload-verified. The hosted
recipe verifies every payload before loading it. Checkpoint licensing does not
relicense training datasets: the model card separately describes mixed upstream
data terms. Publisher quality and calibration figures remain publisher claims.

## Actual inference contract

The `image-text-to-text` pipeline tag does not describe this decision API.
The publisher's `systemone` performs one complete causal backbone prefill with
`use_cache=False`, then returns per-option probabilities from a learned head.
It neither samples nor generates answer tokens; `usage.output_tokens` is zero.

The original serializer constructs its own chat markers and state/schema/suffix;
it does not call `chat_template.jinja`. Strings remain literal. Other JSON values
use compact UTF-8 JSON with sorted object keys. Questions retain request insertion
order. An absent, null or empty instruction falls back to the question ID.
Question spans cover instruction tokens; option spans cover compact semantic
JSON containing the option ID and optional description. Only state tokens are
right-clipped to the remaining budget; an oversized schema fails. Original
collation right-pads, despite the tokenizer's configured left padding. The first
reference lane uses batch1, so padded model arithmetic remains unqualified.

The three types have different semantics:

| Type | Head option order | Public answer |
| --- | --- | --- |
| `noul` | `true`, then `false`; optional description overrides | Probability of `true`, rounded to4 decimals |
| `choice` | Lexicographically sorted string IDs | Argmax and probabilities in original criteria order; that original order resolves exact ties |
| `score` | Ordered list positions `0..N-1` | Expected ordinal index, maximum probability, index-to-description legend |

The text backbone has24 layers, hidden2048/FFN6144:18 gated-delta linear layers
and6 full-attention layers, every fourth layer. Full attention has8 query heads,
2 KV heads, head dimension256 and64 rotary dimensions, theta10,000,000 and mRoPE
sections11/11/10. Full text positions/masks must be captured from the real model.
The configuration's262144 maximum is not a validated context claim.

The separate60,957,454-byte joint-head file is essential. Its width512 head
normalizes every final hidden row with LayerNorm, computes instruction and option
span means, and reads lexical option vectors from the output embedding weights.
Six projections feed two evidence-routing cross-attention blocks and four
pre-norm TransformerDecoderLayers. Head field self-attention is bidirectional:
all requested fields can affect one another. It is not the causal backbone mask.
The final logits combine a scaled lexical prior, a scaled field/option cosine
and a gated residual MLP. Preserve LayerNorm epsilon1e-5, exact GELU, normalize
epsilon1e-12 and cosine epsilon1e-8 separately from backbone RMSNorm epsilon1e-6.

The official loader and `systemone` never read per-type temperatures from
`unsloth_decision_config.json`. The card says the global1.5578663349151611
temperature is already folded into the head. The initial reference applies no
additional temperature. The optional Unsloth vector is ordered choice/score/noul,
which differs from head type IDs noul/choice/score. Supporting that alternate API
would require its own pinned implementation and reference, rather than silently
applying this vector to the official path.

The publisher also provides [Josev-2B-GGUF at8e5c6920](https://huggingface.co/mayflowergmbh/Josev-2B-GGUF/tree/8e5c69202ccd920c4f17c14087c0b4e01f1270a1).
Its card says the GGUF stores per-type temperatures. This route therefore needs
a distinct reference profile even before quantization differences. Small source
inspection confirmed an existing [MIT-licensed llama.cpp Clef graph](https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/src/models/clef.cpp),
[converter](https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/conversion/clef.py)
and [SystemOne server code](https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/tools/server/server-decision.cpp)
at b11541. They are useful graph/conversion references; retain their MIT notices
if adapting code. No GGUF weight payload or author-server output was downloaded
or verified. Do not substitute the author's quantized/calibrated outputs for the
initial official-PyTorch capture.

## Bounded original-first acceptance

Text-v1 accepts finite JSON,1–8 nonempty question IDs and at most24 options,
at most65536 UTF-8 request bytes/512 total tokens, batch1, and no images/videos/media arguments or user
`<|` special-token surfaces. Choice requires string-keyed criteria; score requires
an ordered list; `noul` overrides only true/false descriptions. These are declared
subset exclusions. Text rendering is otherwise unchanged, including whitespace,
Unicode and null. The long owned case must exercise state-only clipping; the
original serializer determines actual tokens and any schema-overflow rejection.
No input is dropped or rewritten to fit a result.

Before any backbone/head weight load or forward, the dumper freezes every actual
original tokenizer ID, attention mask, semantic span, option ID and request.
It captures embedding, each24 backbone layers, final norm, actual rotary inputs
and selected layer-mask inputs, head span/lexical/residual inputs, projections,
routing/field layers, logits, float32 softmax and official typed answers. It
requires ordinary, observed and repeated logits to be bit-exact and the captured
logit conversion to match the real `systemone` response. Hook capture is evidence
only after that observer check. Raw arrays stay in remote artifacts; only manifests,
source/environment fingerprints, input freeze and small reports should be fetched.
The source inspector also checks the unchanged original serializer/answer
declarations with a byte-ID tokenizer stub. Those controls establish source
semantics, and make no claim about actual Qwen tokenization or inference.
Ten lightweight controls passed against the inspected source, including
altered/missing source rejection and excluded-input/size/order boundaries:

```sh
JOSEV_SOURCE_DIRECTORY="$JOSEV_SOURCE" \
  python -m unittest discover -s tests -p test_josev2b_contract.py -v
```

Without the optional already-fetched source directory, the source-semantics test
skips; the other controls require only the standard library. No test downloads
a tokenizer or imports a model framework.

Future native comparisons keep the fixed vector/stage0.002 maximum absolute
error and0.99999 cosine budgets, logits0.002, probability error0.0001 and ordinal
score error0.001. IDs, spans, option ordering, choice winners and zero generated
tokens must agree. Record every four-decimal answer difference rather than
claiming rounded JSON parity from a cosine. Require finite activations and
probabilities; attention masks may contain negative infinity. These new owned
cases establish serialization and numerical coverage, not representative German
decision quality. A licensed task-quality panel remains a separate dependency.

## Completed original GPU reference and head parity

The frozen original uses workerb2edf8c83c9aca1f8a9210198d47798ac13828b8,
bootstrap5e938857368820b0d4d49bd977561e3f418ed9c0, and receipt
SHA256`51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`.
Its57-package audited Python3.13 environment uses PyTorch2.13.0,
Transformers5.5.0, tokenizers0.22.2 and the exact versioned lock. The capture
uses CUDA/F32, eager backbone attention, math SDPA for the head, disabled TF32/
MHA fast paths and no optional FLA/causal-convolution extensions. All four
ordinary/observed/repeat cases pass. Actual reference time39.46s and peak CUDA
allocation9,082,022,400 bytes are original-only session measurements.

The current approved CUDA13 original profile requires compute capability≥7.5,
driver≥580.65.6,14GiB total/11GiB free GPU memory and25GiB scratch before
installation/payload access. Unsupported hardware rejects without automatic
resource retry. The original is frozen; do not repeat it to test a new native
implementation. Initial Python/pip/tokenizer-version/metadata-selector failures
and exact old sources remain in HISTORY and the retained negative receipts.

All251 arrays and native adapters are preserved by
[40429513945](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40429513945).
The122-tensor head converts losslessly into121,900,352 bytes. The actual
[head CPU/CUDA audit](../tests/results/josev2b/head-cpu-cuda-review.json) passes
123 stages per backend and all four complete typed answers; worst absolute
errors1.14441e-5/7.62939e-6. This uses original hidden/lexical inputs and does not
qualify native backbone/composed inference, public API, representative quality
or a new release. See [the GPU protocol](josev2b-head-gpu.md).

The exact checkpoint index contains320 text tensors among632 total keys and no
separate output head. Their1,881,825,088 elements imply7,527,300,352 F32 payload
bytes before GGUF overhead. The approximately9.1GB estimate describes the full
conditional checkpoint including unused vision. Storage calculations are not
measured native peak memory; actual backbone conversion and parity remain gates.

## Incremental native port

1. Review the original capture, input/span contract, memory/timing and calibration
   before native outputs. Preserve any first failure and its exact source.
2. Add a dedicated default-off Josev tensor/profile loader and converter, retaining
   the separate head and output-embedding semantics. The existing Ovis
   `map_small_decoder` explicitly requires1024/3584; Josev is2048/6144. Preserve
   that small-model guard. Shared full-attention, gated-delta, convolution and
   RMSNorm kernels are candidates for reuse only after stage comparison.
3. Introduce an explicit hidden-only full-prefill decoder interface. The existing
   decoder always computes last-token vocabulary logits, which this decision path
   does not use. Compare embeddings, first differing layer, final norm and actual
   text position/mask behavior before adding the joint head. No generation/cache
   optimization is needed for the initial original path.
4. Implement the distinct512-wide joint head and trace its routing, cross-field
   attention, lexical prior and residual. The existing Gemma typed-decision path
   independently embeds state/questions/options into a different256-wide head;
   its arithmetic/temperature pipeline cannot substitute. Reuse bounded JSON
   parsing, result ownership/error handling and binding scaffolding after defining
   whether the public API exposes SystemOne or an explicit common adapter.
5. Add converter/registry hash pins, C ABI, CLI and Python/Rust/Dart coverage with
   literal0 gate and negative-input controls. Only then qualify realistic licensed
   decision quality and portable artifacts. No app pin, release or default changes
   follow from this source audit.

Image/video support requires separately frozen Qwen3-VL processor outputs,
patch/merge order, media placeholders, positions/masks and composed decision
outputs. Ovis's OCR image preprocessing/prompt is not that processor contract.
Long context, batching/padding, BF16, alternate calibration, quantization and
prefill caching also remain separate experiments with original-first references.

## Retained setup history

The completed setup provenance and initial rejected recipes are archived in
HISTORY. Current work uses the completed original and head receipts above;
old commands describing an undispatched profile or tokenizers0.23.0 are
historical records, not instructions to repeat model capture.

## Public API integration boundaries after numerical parity

The existing decision surface is Gemma-specific in several material ways:
`crispembed_decision_scores` and Python `decision_scores` accept separate
normalized768-dimensional vectors and kinds choice0/score1/noul2. Josev instead
uses joint2048-dimensional backbone token rows plus semantic spans, with kinds
noul0/choice1/score2. Its field decoder scores questions jointly. Do not adapt
these differences by relabeling dimensions or reusing the Gemma kind enum.

The existing `crispembed_decide_json` contract accepts state/questions/workflow
and returns a flat per-question result. The official Josev `systemone` requires
a string model identifier and returns `{model,answers,usage}`, including rounded
choice probabilities, score legend/expectation and noul probability. Python
`decide` currently omits the model field. A dedicated SystemOne request entry
point, capability query and matching C/Python/CLI/Rust/Dart/HTTP wrappers can
preserve both contracts; settle their exact public names after composed parity.
Keep the Gemma scoring entry point scoped to its documented model and reject
unsupported component calls explicitly.

The ordered `Value.fields` parser in `src/core/decision_schema.h` is a potential
syntax primitive, but its Gemma rendering and kind/criteria handling are not the
Josev serializer. Preserve question insertion order, lexical choice encoding
order, original criterion order for tied answer selection, compact sorted-key
JSON for non-string semantic values, and state-only truncation. Verify native
UTF-8 serialization/token IDs/spans against the actual original input freeze.
Do not use an ordinary map for the insertion-ordered question sequence.

Match Python's four-decimal answer rounding, including exact half cases such
as0.03125→0.0312; ordinary `std::round` uses a different tie rule. Retain F32
softmax probabilities as reference values before formatting and compare decoded
answers separately from numeric stage budgets. Add schema/order/tie/rounding,
missing-model, oversized-schema, media-rejection and unsupported-component
controls across the real public surfaces. No public API work or answer parity
is established by this source analysis.

The backbone component must expose all unpooled final hidden rows for the
explicit2048/6144 profile. Preserve the existing1024/3584 profile guard; avoid
unused vocabulary projection or token generation in the new hidden-only route.
Freeze actual original positions/masks and compare the earliest differing
backbone/head stage before composing typed outputs.
