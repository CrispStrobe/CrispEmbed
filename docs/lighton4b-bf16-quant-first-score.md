# Genuine original BF16 model with streamed Q8/Q4 weights: one-position protocol

This distinct diagnostic addresses whether the immutable quantized weights alter
the original model's first score independently of native runtime differences.
It creates no F32 model/reference or replacement gold. Exact source/weight/input
pins are in [the frozen protocol](lighton4b-bf16-quant-first-score-v1.json).

The worker loads the original authenticated9,078,620,536-byte checkpoint in BF16
using the unchanged original CPU/eager driver at8d3b2331. One AST substitution
replaces only the generation call with a bounded adapter. Metadata/tokenizer,
processor, prepared-input hashes, framework/source checks and checkpoint hash
remain the original guards. Before modifying any weight, one original first-score
forward must reproduce frozen score SHA a4bece2310 and ID27 exactly.

Each complete immutable Q8/Q4 GGUF is fully hashed before use. Two independent
header readers must agree on all723 names/shapes/types, including the flattened
vision patch mapping. All249 quantized text matrices (embedding included) are
independently dequantized in bounded chunks and converted to BF16 before in-place
copy into the existing original parameter. No F32 parameter/model is installed.
Every copied BF16 chunk and complete parameter is readback-hashed. All474 protected
tensors must match the original BF16 source exactly through their GGUF F32
representations; original protected hashes and buffers must stay unchanged.
Parameter identities/storage, complete BF16 dtype/shape census and tied
embedding/output head are checked before and after every arm. Q4 overwrites the
same complete249-matrix set after Q8, so no Q8-only matrix is left behind.

**Effective-precision limit:** dequantized matrix values are rounded to BF16 for
genuine BF16 original inference. This differs from the prior mixed-precision
activation oracle retaining F32 dequantized matrices. Per-tensor conversion error
and effective BF16 parameter hashes are recorded explicitly; neither result
silently substitutes for the other precision protocol.

Each arm executes exactly one fresh original image/decoder prefill with
`max_new_tokens=1`. Forward counters require one root, vision and text call.
Complete248,320-vocabulary score rows, first IDs/probabilities and decoded prefix
are retained for original/Q8/Q4. Prepared inputs are authenticated before every
arm. There is no EOS/document-completion or held-out quality claim. The original
driver's audit receives only a minimal copy of the baseline one-token result;
its one persisted score must still match original gold. Candidate results have
separate names and never overwrite the original177-position reference.

Initial effective available RAM must meet the existing22GiB original preflight;
use the previously authorized32GiB source CPU worker, never the shared host.
Streaming chunks are at most262,144 F32 elements (1MiB plus bounded BF16/dequant
scratch); no2.54GB F32 embedding is allocated. The model is resident once and
Q8/Q4 run sequentially. Before each quantized forward at least6GiB effective
headroom must remain, using actual cgroup hierarchy controls and reclaimable
clean file accounting. Require at least27GiB scratch initially; source checkpoint,
one candidate and all large payloads stay remote. Only task-owned candidate files
are unlinked after complete map/readback verification. Output is small reports
and three score rows; no caches or intermediate arrays are captured.

Model-free checks validate source723/249/474 census, patch geometry, both type
policies, missing/extra tensors, wrong protected/embedding/type/shape/recipe,
frozen driver binding and its single AST substitution. Actual execution remains
pending. A completed strict rejection exits nonzero while preserving all arms.
No native inference, conversion publication, default, alias, release or app-pin
change is authorized by this diagnostic.
