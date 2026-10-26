# SauerkrautLM-MoEdernColBERT source and reference contract

This is original-reference preparation, with no native parity or release claim.
The [requested collection](https://huggingface.co/collections/VAGOsolutions/sauerkrautlm-moederncolbert)
contains three parameter scales and several representations. The machine-readable
[source contract](moedern-colbert-source-contract.json) freezes all three original
repositories and their author GGUF repositories, source/configuration/license
hashes, published LFS weight SHA256 values and byte counts. Model payloads are
not distributed in this repository. CoreML tensor equivalence remains unaudited;
a common model name does not establish representation parity.

| Size | Original revision | Author GGUF revision | Hidden / factorized width | Blocks | Attention heads / KV | Expert width |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5M | `9d360b8b88ae11bea5784bf9012ff509fdd5c2da` | `fcc9c64025ba5d6cc0d8f88f846f55c5cd4e8bae` | 128 / 16 | 2 attention+MoE | 2 / 1 | 128 |
| 27M | `d2b7b1feaaac1bf87510ca9525208c3daeabcc5a` | `4ff61393a179480ed624e79606ea45353a40d52d` | 640 / 80 | 2 attention+MoE | 10 / 5 | 640 |
| 115M | `66380d2ade3b5027bba83bdcdba052d0bb86dcdb` | `d86747d5c597c7eb53ed793d5c29773f77f5d411` | 1024 / 192 | 6 convolution, 4 attention+MoE | 16 / 8 | 640 |

The three original repositories contain byte-identical
[`modeling_modernbert_hash_moe_v10.py`](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/blob/9d360b8b88ae11bea5784bf9012ff509fdd5c2da/modeling_modernbert_hash_moe_v10.py).
Their separate `1_Dense` heads project each token to 128 dimensions. All six
repositories publish the same complete Apache-2.0
[LICENSE](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/blob/9d360b8b88ae11bea5784bf9012ff509fdd5c2da/LICENSE);
the GGUF model cards explicitly cover model files and code. No additional NOTICE
file appears in the audited trees. The author C engine links MIT-licensed ggml,
with its README pinning
[`ggml-org/ggml@3612c0af`](https://github.com/ggml-org/ggml/tree/3612c0af93f2cfc8f048295300a61187723b12ae).
This lane independently implements the tensor schema; it does not vendor author
implementation code. Hosted source snapshots retain the original license files.
Any later copied/adapted engine must retain its applicable Apache/MIT notices.

## Actual inference blueprint

The embedding called “hash” is a factorized learned lookup `[vocab, P]`, then a
bias-free `[P, H]` projection and LayerNorm with weight, bias and epsilon `1e-5`.
It is not a hash function applied to input token IDs. Vocabulary size is 30,002.

Attention blocks apply pre-RMSNorm, bias-free Q/K/V projections, NeoX half-rotation
RoPE (`theta=160000`, head width 64), KV repetition by two, noncausal scaled dot
product attention and an output projection plus residual. The RMSNorm epsilon
is `1e-5`. The evaluation sliding window activates only above 513 tokens; this
first contract bounds sequences to 512.

After attention, RMSNorm feeds eight routed experts and one shared expert.
The router applies softmax, top-2 selection and selected-weight renormalization
with `1e-9`. Each expert computes `GELU_erf(gate) * up`, then `down`; add weighted
routed outputs, the shared expert and the residual. Original expert parameters
have shapes `[8,H,He]` and `[8,He,H]`; the GGUF mapping explicitly transposes the
last two dimensions. Torch top-k ties and the author's ascending-index host
selection require separate tie controls before claiming routing equivalence.

The 115M block order is `conv,conv,attn,conv,conv,attn,conv,attn,conv,attn`.
Its six convolution dilations are `1,1,2,1,1,2`, with kernel 3. A convolution block
uses RMSNorm, a three-way projection into B/C/X, independent RMSNorm on B/C,
`B*X` masked by the effective attention mask, depthwise convolution with symmetric
dilated padding, multiplication by C, output projection and residual. Final
RMSNorm precedes the separate bias-free 128-dimensional identity-activation
Dense head and per-token L2 normalization.

## Tokenization, roles and scoring

The pinned `tokenizer_config.json` declares **AlbertTokenizer**, lowercasing,
accent removal, prefix space and `[MASK]` as its padding token. The tokenizer
payload has not been inspected by local model/tokenizer execution. Its exact
Git blob is `e1fc54fd0e0456a6c76106be4a338bfc08f29861`, 1,956,542 bytes, identical
across the six repositories. Hosted preparation verifies that Git object,
freezes its SHA256 and reports actual JSON model/normalizer/pre-tokenizer,
post-processing and added-token flags before loading it. Large embedded
normalizer strings are represented by length and SHA256 in the small report.
Do not infer WordPiece, BPE or Unigram from the vocabulary count.

The author GGUF converter writes architecture metadata and marker IDs, **no
vocabulary, scores or merges**. The author's text wrapper therefore needs the
adjacent pinned tokenizer sidecar. This dependency also applies to the strict
F32 converter; the output is not a self-contained text model.

The original reference uses the pinned Sentence Transformers 6.1.0
[`MultiVectorEncoder`](https://github.com/huggingface/sentence-transformers/blob/b0bf39968ba461c34b29372144794eff1be06f6d/sentence_transformers/multi_vector_encoder/model.py)
legacy ColBERT loader. It remaps the saved PyLate Dense module and appends
MultiVectorMask and Normalize. It restores `[Q] ` / `[D] ` prefixes from the
saved role configuration. Queries truncate/expand to 32 IDs with MASK ID 4;
documents truncate to 512 and exclude configured punctuation token IDs from
scoring. CLS=2, SEP=3, query marker=30000, document marker=30001. Literal embedded
`[Q] [D] [MASK]` inputs are owned parity controls, since added-token semantics
must survive any future native tokenizer adapter.

Three masks have distinct meanings: the wrapper supplies attention zeros for
query expansion; the custom backbone turns the effective attention mask into
all ones when position 1 is query marker 30000; the final scoring mask includes
all 32 query vectors. Documents retain their attention mask in the backbone,
then remove configured punctuation IDs for scoring. Capture supplied, effective
and scoring masks separately. Do not copy the final scoring mask back into
backbone attention.

Outputs are ragged `[scoring_tokens,128]` normalized vectors. MaxSim sums, over
query rows, the maximum query/document dot product across document rows.
A proposed internal token-ID API takes an explicit role, IDs and supplied mask,
returns all per-position vectors, and leaves scoring-mask selection explicit.
Generic pooled encoding and generic ColBERT prefix handling are insufficient.
The parent integration owns eventual public adapters and context registration.

## Bounded original-first execution

The reusable [original workflow](../.github/workflows/moedern-colbert-original.yml)
accepts `variant` (default `1.5M`) and `convert_after_original` (default true).
It runs two-thread CPU F32 on hosted CI only, with a 60-minute bound. The original
remote Python code is executed only after every pinned source and payload hash
has passed, and the four relevant Sentence Transformers source hashes are
checked against the pinned upstream commit. Torch/Transformers/NumPy versions
and full installed package inventory are retained. The existing metadata dispatcher registers phase `moedern-original-1p5m`,
which calls this workflow with the smallest variant and conversion enabled.
Start with 1.5M; review the
actual original before scheduling larger configurations.

The [owned panel](../tests/fixtures/moedern-colbert/panel-v1.json) freezes ten
cases before candidate inference: German query/document, empty query, punctuation
and currency, literal markers, mixed scripts/diacritics/emoji, query/document
truncation, one character and whitespace. Two short cases capture every actual
backbone module input/output plus actual SDPA Q/K/V/mask/output through neutral
test-only hooks. Pipeline hooks preserve pre-normalized and normalized vectors. Router logits
are captured; softmax/top-2/renormalized weights and selection margins derived
from them are explicitly labeled `derived`, not observed internal arrays.
Other cases preserve final arrays and role/mask contracts. All snapshots clone
before later mutation; captures never enter production model files.
Pinned ST6.1 calls the backbone's bound `forward` method directly, bypassing
PyTorch root-module hooks. A temporary test-only wrapper observes that actual
method invocation and restores its original lookup in `finally`; ordinary layer
hooks still observe the called submodules. Dependency-free direct-call controls
cover input/output capture, unchanged results and restoration after failures.

Every case checks ordinary versus observed feature dictionaries bit-exact and
checks the public role output against the final scoring-mask slice. Original
MaxSim is recorded alongside an explicit query-row reduction. Array records
include row-major shape/type, byte count and SHA256. Full originals/checkpoints
and arrays stay in remote artifacts for 90 days; the small report contains
identities, stages, masks, scores, timings and peak RSS. Rejection reports and
partial stage arrays are retained even on failure.

The [F32 converter](../tools/moedern-colbert/convert_f32.py) requires a completed
original receipt and its exact SHA256, rehashes the original checkpoint and
rejects unknown/missing tensors, incorrect dimensions, non-F32 or nonfinite
values. The expected counts are 32 / 32 / 94, including the Dense head. It
preserves all constants and tensors, records explicit transpose mappings and
independently reads back every GGUF payload. It introduces no quantization.
Author GGUF files remain separate immutable artifacts; a later audit must compare
their actual payload inventory against the mapped original, not equate metadata
or whole-file hashes with tensor equivalence.

Before native execution, independently review the original receipt and freeze
its service artifact IDs, ZIP/member SHA256 values and actual tensor inventory.
Capture the author engine's actual intermediates with its pinned ggml separately;
its default repacking, grouped versus `mul_mat_id` expert path, flash-attention
precision and L2 epsilon semantics remain comparison boundaries, not proven
Python parity. Probes must retain actual graph allocations, including view owners,
and establish ordinary/dump neutrality before their arrays are trusted.

Future candidate gates are fixed at maximum absolute error `0.002` and minimum
per-row cosine `0.99999`, for stages and final vectors, with MaxSim absolute
error `0.002`, exact IDs/masks and ordinary/dump neutrality. Routing tie cases
need explicitly labeled handling rather than silent index changes. Use
`crispembed-diff` plus independent raw-array checks for actual native replay.
An owned panel is a tokenizer/numerical control, not representative retrieval
quality, speed qualification, default promotion or a release. No native engine,
public registry/API change, shared build change, quantization qualification or
CoreML equivalence is provided by this preparation.
