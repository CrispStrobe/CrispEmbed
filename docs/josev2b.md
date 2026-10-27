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

## Prepared GPU recipe; not dispatched

Use an authorized free GPU slot after source/resource review. Do not submit this
2B reference to a CPU matrix. The proposed original profile is CUDA/F32 with
PyTorch2.13.0, Transformers5.5.0, tokenizers0.23.0, safetensors0.8.0,
huggingface-hub1.33.0, NumPy2.2.6, accelerate1.13.0, torchvision0.28.0 and
Pillow12.3.0 on Python3.13. The latter two are needed to construct the original
processor even for this text-only path; they do not establish image parity. It pins the
actual Qwen implementation bytes. It uses eager backbone attention, math SDPA
for the head, deterministic algorithms, disabled TF32/MHA fast paths, and an
isolated environment without FLA or causal-conv1d extensions. This is an explicit
F32 audit profile, distinct from the publisher's usual BF16 execution. No original
GPU result has yet established its memory use, runtime or neutrality.

Resource preflight requires14GiB total/11GiB free CUDA memory and18GiB scratch.
Compute capability must be at least7.0. The recipe checks `nvidia-smi` before
framework import or payload access; a P100 draw writes `conclusive:false`,
`retry_permitted:false` and exits. Do not retry the allocation hoping for a
different GPU. The dumper repeats the capability guard before reading payloads.
Allow30 minutes for capture, at most2GiB array payload, and preserve failed
outputs. The F32 backbone alone needs approximately9.1GB of parameter storage;
this estimate is not a measured peak. A16GiB GPU is a candidate only if actual
preflight passes. The recipe fails on CPU or missing resources. Installation and
download time are separate from the30-minute capture bound. Review availability
of the pinned CUDA wheel and retain the actual driver/CUDA/package fingerprint.

After installing those exact dependencies in the approved GPU environment and
checking out the reviewed commit, use public service paths and fresh output
directories supplied by the execution environment:

```sh
python tools/kaggle/josev2b_reference.py \
  --reviewed-source REVIEWED_FULL_COMMIT \
  --reviewed-contract-sha256 REVIEWED_CONTRACT_SHA256 \
  --scratch "$JOSEV_SCRATCH" --out "$JOSEV_OUTPUT"
```

The helper downloads only the exact pinned files, checks their payload hashes,
then switches offline before capture. It does not submit a job, configure an
account, publish artifacts or clean shared storage. Preserve the Git commit,
contract SHA, complete logs, `input-freeze.json`, `reference.json`, license/source
notices and raw-array archive plus archive/member SHA256s in remote storage.
Record the service run ID and expiry only after the actual run exists.
Review reports for private environment details before adding public evidence;
keep complete original logs in the authorized remote archive.

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

### Concrete private GPU bootstrap

The [single-script bootstrap](../tools/kaggle/josev2b-original/bootstrap.py)
pins its original worker commit and contract digest explicitly. The current
prepared worker profile is `cuda-f32-python313-v3-locked`; its arithmetic, source files,
tokenizer, four owned requests and acceptance gates are unchanged. Its own script
version is separately recorded; pushing other repository commits
does not update a submitted kernel. Account substitution stays in private execution
metadata; no account, token dataset or credentials are embedded in this recipe.

The initial bootstrap at[b6e74739](https://github.com/CrispStrobe/CrispEmbed/commit/b6e747395f56847a72f7f4fb9b7023374285924f)
(SHA256 `9f3d77cfb41eda1a68ab287e0c01fb3e837d719a8a539e86e4377c92a4858ff3`)
used worker89264a279cd589d622b60f5ca7d1605693cf52d8 and contract
SHA7f141896f708871d5beb01760a649a38030488851026ef10b858d5f33b96027a.
It rejected the actual Python3.13 environment because that first profile required
Python3.11. The rejection occurred before hardware inspection, dependency
installation, checkpoint/tokenizer access or any reference output. It was
inconclusive and says nothing about the assigned GPU or model. The initial
profile, source, failure and journal are retained; there is no gold recapture.

Before any reference output, this revision declares Python3.13 explicitly.
Primary [torch2.13.0 release metadata](https://pypi.org/pypi/torch/2.13.0/json)
and [torchvision0.28.0 metadata](https://pypi.org/pypi/torchvision/0.28.0/json)
confirm the cp313 Linux wheels and their SHA256s. NumPy2.2.6 also publishes a
cp313 Linux wheel. Wheel payloads were not downloaded during source preparation.
The bootstrap records the actual Python version and inspects hardware before
checking its Python profile, so unsupported hardware is recorded immediately.
The Python3.13 profile remains unexecuted until a reviewed bounded run occurs.
 These wheels use CUDA13.0.3; they are not available in
the inspected CUDA12.8 wheel index. The bootstrap therefore checks Turing or newer
(capability7.5), driver580.65.06 or newer and14/11GiB total/free GPU RAM before
framework installation. It requires25GiB initial scratch and3GiB output space,
then the original worker independently checks18GiB scratch before checkpoint
access. Unsupported hardware exits inconclusive with no automatic retry.

Installation uses a new isolated environment and the complete57-wheel hashed
lock described below. `pip check`, a tiny exact F32 CUDA matrix product, compiled GPU architecture
reporting, contract/source controls and actual original source hashes precede
model access. Setup/download and the30-minute capture have separate
bounds. Only reports/logs/source notices and bounded reference arrays are retained
as kernel outputs; checkpoints, tokenizer and environment stay in task scratch.
Failures retain their bootstrap journal, command status/log SHA256 and partial
reference manifest. This prepared bootstrap is not an actual reference result.

The corrected Python3.13 attempt at142cc75c passes actual two-T4 capability,
driver580.178.04 and memory preflight, then its venv `ensurepip` child fails before
framework installation or any tokenizer/checkpoint access. Preserve that complete
bootstrap journal and four successful Git command logs. The subsequent setup
repair reuses the established Ovis recipe: create a fresh venv without ensurepip,
then use base pip's `--python` option to install SHA-pinned pip26.2.2 into that
venv. User pip configuration is disabled for this task. No global environment or
model arithmetic changes; supported-GPU evidence is separate from original
inference, which still has not run. Unsupported hardware still stops without retry.

### Complete locked environment before the first original output

The third setup attempt, using the v2 profile and bootstrapf5a0ac06, successfully
creates isolated pip26.2.2. Installation then rejects unpublished
`tokenizers==0.23.0` before any checkpoint/tokenizer access or reference output.
It downloads the pinned framework wheels on the GPU service but does not complete
framework installation. The preceding `sitecustomize` missing-`wrapt` warning is
nonterminal; no `wrapt` package is added to hide it. Keep this failure separate
from the earlier Python-profile and ensurepip failures.

The new pre-reference profile is `cuda-f32-python313-v3-locked`, with
`tokenizers==0.22.2`. Its cp39-abi3 Linux wheel is compatible with CPython3.13
and the pinned Transformers5.5.0 requirement. The other eight package pins, original model/tokenizer files, four owned inputs,
arithmetic and numerical gates are unchanged. CUDA13's capability7.5 minimum is now enforced by both the worker
and bootstrap, preventing a standalone worker from reaching payloads on Volta.

The [complete lock](josev2b-environment-lock.json) fixes57 distribution versions,
exact wheel URLs, published byte counts and SHA256 values. The
[requirements file](../tools/kaggle/josev2b-original/requirements.lock) installs
that closure with `--no-deps --require-hashes --only-binary=:all:`; the GPU setup
must not run an unchecked dependency resolver. It includes all CUDA-toolkit
extras selected by Torch, with2,790,174,915 published wheel bytes in total.
No wheel/sdist, framework, tokenizer, checkpoint or corpus was downloaded or
imported during the local metadata-only preparation.

Existing pip25.3 performed a binary-only, ignore-installed dry run for Linux
CPython3.13 under a tested HTTP transport guard. The guard rejects every wheel,
sdist and unrecognized URL before transport, including redirect targets; it
permits bounded PyPI indexes/release JSON and PEP658 `.whl.metadata` only.
The resolver completes114 metadata requests, followed by114 independent primary
release/PEP658 checks. Every selected URL/hash matches its PyPI release file.
An independent marker/version/extras walk verifies the entire closure for the
resolver's Python3.13.11 and the observed GPU Python3.13.15. These are dependency
metadata results, not installed ABI, CUDA execution or original-inference proof.

Aggregate release JSON is not always the selected wheel's exact metadata:
CUDA-toolkit uses PEP440-equivalent13.0.3/13.0.3.0 and different Windows-machine
markers, and psutil uses underscore/hyphen spellings in a development extra.
Those differences are retained. Actual selected-wheel metadata agrees with pip;
Linux CPython3.13 active requirements agree with primary JSON for every provided
extra. No active Linux dependency/version conflict is ignored.

Before model access, the locked worker verifies the expected venv prefix,
distinct base prefix, disabled user site, absence of external package paths,
all57 installed versions and byte-exact installed wheel METADATA. The bootstrap
also checks loaded framework versions and origins inside that venv. `pip check`,
original Qwen source hashes, absent FLA/causal-conv extensions and the existing
exact CUDA F32 smoke remain mandatory. Runtime CUDA/driver compatibility and the
first original capture are still pending; no new GPU dispatch follows from
metadata resolution alone.

Immediately after the unchanged publisher loader returns, and before any forward,
the dumper compares the loaded head state with the published small head checkpoint
promoted to F32. Exact keys, shapes and finite F32 values are required.
`report.weights.head_tensors[name]` records `shape`, `checkpoint_dtype`,
`loaded_dtype`, `checkpoint_promoted_f32_sha256` and `loaded_f32_sha256`; any
intentional loader transform fails with partial evidence rather than being
accepted silently. BF16 checkpoint tensors and F32 scalars remain distinguishable.
This read-only proof does not copy or rehash the full backbone a second time.

The locked bootstrap targets worker
[`2133cfa12dbdd25653fd68b1a6054f69f80f7e9c`](https://github.com/CrispStrobe/CrispEmbed/commit/2133cfa12dbdd25653fd68b1a6054f69f80f7e9c)
and contract SHA256
`6fbe3efe8958b2d87f5fe42c52fa71857c2d644efe3ad363bb60d8dfa52885da`.
The bootstrap's own uploaded bytes remain a separately recorded identity.
It validates and preserves the complete lock before installation, checks isolated
prefixes, then verifies exact installed metadata and actual loaded module versions
before the unchanged CUDA/source barriers and original worker. This is prepared
source only; no fourth GPU attempt is submitted by this change.

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
