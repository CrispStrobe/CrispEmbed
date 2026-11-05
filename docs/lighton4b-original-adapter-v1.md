# LightOn4B original generated-position adapter

This is an executable observer implementation and source-bound capture adapter,
not an original-model run or document parity result. The capture entry is
**hard blocked** by `CAPTURE_TRANSPORT_ADMITTED=False`: a separately reviewed
transport protocol must admit the all-position cache shards before any original
forward. The existing [document-v1 protocol](lighton4b-document-parity-v1.md)
and its empty producer admissions remain unchanged. Its2GiB archive bound does
not cover this adapter's complete generated-cache transport.

`tools/lighton4b_original_document.py` verifies the immutable original dumper's
whole SHA, then adds exactly four observation sites through a separately hashed
AST: pre-model input validation, a first save callback, a top-level original
model output hook, and an independent observer-off generation replay. It does
not modify the old file or any upstream model implementation/arithmetic. The
original dumper already captures final normalization during prefill, cached and
full-prefix phases. Its generated-field filter drops generated final norms;
the new save callback streams those norms and positional arrays before that
filter. The new model hook streams each original-position full248320 logits and
all64 cache fields across32 layers. There are67 snapshots per generated position.

Linear caches must have exact `[1,8192,4]` convolution and `[1,32,128,128]`
recurrent geometry; full-attention keys/values must have four256-wide heads and
at most512 sequence positions. The actual processor must reproduce the frozen
452×317 RGB/template,560 patches with1536 features, grid1×20×28 and a batch-one
prompt of141–256 tokens before model construction. Each snapshot has finite
F32 values, a shape/byte/SHA record, an atomic file completion and explicit byte
limits. No reference cache values are supplied to an original or native forward.

The independent same-backend replay uses the same single loaded original model,
request, greedy parameters and fresh official generation cache. Every generated
ID and all248320 score values at every position must be exact, and streamed raw
model logits must equal the official generation scores. Greedy IDs, full
final-norm/position/cache coverage and EOS248046 are checked. Capped, nonfinite,
incomplete, duplicate and observer-changing captures fail while retaining their
partial manifests. Metadata/full tokenizer/checkpoint hashes, exact CPU package
versions, model/config modules and the separately pinned original Torch fallback
modules/functions must pass before the optional capture can construct a model.

The current conservative admission requires40GiB available RAM and48GiB free
output storage, after any existing snapshot occupies space. This intentionally
rejects the observed approximately32GB CPU worker. It is an enforced arithmetic
admission bound, **not a measured peak**:

| RAM reservation | Bytes |
|---|---:|
| 4,539,265,536 original parameters in F32 | 18,157,062,144 |
| Original BF16 load/mmap transient bound | 9,078,620,536 |
| Legacy retained trace arrays, checked before copying | 8,589,934,592 |
| Framework/workspace/copy headroom within40GiB | 7,124,055,688 |

At most256 input tokens plus256 generated positions bound the full KV context
to512. A conservative generated position reserves53,477,376 bytes for all
linear caches,33,554,432 for full-attention caches,2,621,440 for final norm,
4,096 for positions and993,280 for logits:90,650,624 bytes. At256 positions this
is23,206,559,744 bytes, below the enforced24GiB shard budget. Disk admission
reserves24GiB shards, two8GiB legacy NPZ/GGUF arrays and8GiB for writer padding,
casts, temporaries and headroom. Every write additionally checks actual space.

A separate sequential alternative could stream each original prefill/cached/
full-prefix phase, release phase outputs and remove the retained8GiB trace
dictionary while keeping one original model and exact greedy settings. It must
first match frozen original stage/logit/ID outputs and measure cold peak RAM on
a larger hosted worker; the F32 model/load transient alone is approximately
25.37GiB. No32GB execution is admitted by that arithmetic. This alternative is
not implemented here; no cap, dtype, strict budget or original request is lowered.

Hosted synthetic controls must verify the four AST injections,67 actual-width
snapshot events, unmodified input arrays, independent replay rejection, cleanup,
nonfinite/duplicate/byte-budget failures and transport rejection before model
access. They do not install Torch, load model weights or run original inference.
Pending work is independent source review, the larger-shard transport contract,
actual high-memory original execution and observer-neutrality evidence, packaging
into a separately admitted document protocol, and native format5 composition/
tokenizer/vocabulary/caches followed by real decoded and intermediate comparison.

Actual [hosted41976285821](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41976285821)
at `953800bde3bbe4251329bdc7e174c2fba82b3e0f` passes all5 synthetic controls,
with zero skips. Independent small-artifact review verifies all6 recorded source
hashes and the log digest, including the32-layer/67-field synthetic cache event,
read-only source arrays, independent replay rejection and transport rejection
before model access. [The receipt](../tests/results/qwen35-ocr/lightonocr3-4b-original-adapter-controls.json)
is explicitly model-forward negative. The [transport proposal](lighton4b-original-transport-proposal-v1.json)
defines at most512MiB parts, whole-file/range hashes, complete-field/EOS checks
and remote readback requirements; it is unreviewed, unimplemented and cannot
enable capture. The unchanged2GiB document-v1 envelope stays in force.
