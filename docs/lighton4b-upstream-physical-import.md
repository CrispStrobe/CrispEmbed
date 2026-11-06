# Physical upstream Q4_K_M import: measured-layout audit plan

The goal is a native comparison using the same represented upstream text weights,
without requantization or a full F32 model. This is separate from the new weighted
HF-layout recipe: equal type schedules do not imply equal quantized values.
No inference result or runtime import is claimed yet.

The predeclared [audit protocol](lighton4b-upstream-physical-audit-v1.json) binds
publisher `8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01`, whole Q4_K_M file
SHA256 `b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26`,
and native protected-source file SHA256
`f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91`.
A previous bounded header read establishes426 physical text tensors:
177F32/216Q4_K/33Q6_K. The Q6 `token_embd.weight` is the tied output; no separate
output tensor exists. All426 physical→native names, shapes and types are frozen
under mapping SHA256 `cf3af1c7f7893a06cf1660358e3ec7084dd6352730e8bcfd384e749e4f11edb1`.

The header's general.name declares original revision
`b71010f095dc3735de7ee6b9969bff8f8f50b39a`. This refines the earlier independent
baseline limitation: the revision is declared in metadata, but source tensor
content identity is not independently established by a declaration or model card.

The [pinned converter](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/conversion/qwen.py)
changes grouped16-key/32-value heads into tiled order. At head dimension128,
its GDN out-projection column permutation crosses the256-value Q4/Q6 blocks with
shared scales. Copying/reordering whole256-value blocks cannot implement that
inverse permutation. Dequantizing then requantizing would change represented
weights and invalidate a same-weight experiment.

The feasible candidate is to retain out-projection physical blocks and reorder
the native grouped activation into upstream tiled order immediately before its
projection. QKV-V, Z, a and b can be inverse-permuted by whole rows without
altering quantization blocks. Conv and per-head small parameters require the
same coordinated permutation. Upstream pretransformed `−exp(A_log)` and shifted
RMS scales must be measured directly: inverse log/subtraction is not assumed to
recover original floating values or source arithmetic.

One authorized Kaggle CPU audit runs no model graph or decoder. Before transfers
it enforces6GiB available cgroup-aware RAM,12GiB scratch and1GiB output. Complete
immutable files total6,739,762,528 bytes; only existing quantized files are read.
The native EXCLUDE_FROM_ALL audit helper streams all249 actual Q4/Q6 matrices in
at most65,536 decoded values per chunk, emits packed/dequantized hashes and
magnitudes, and never writes dequantized tensors. Independently pinned upstream
NumPy decoders hash all represented values. Every177 small parameter is compared
against the appropriate original protected F32 value/transform, including inverse
loss measurements. Complete row permutations and168 predeclared real-valued
out-projection controls use deterministic geometry-wide inputs, with an
unpermuted negative control; no image output is used to fit arithmetic.

The native helper hashes its actual decode, while the independent path uses
upstream6c01da6278ab6c69ed67d9124806b50d2b299ba7. Exactness failures are retained;
execution completion does not waive them. Four build threads and a90-minute
whole-worker deadline bound the job. No GPU, original forward, model inference,
new quantized/F32 model, registry/default/release change, new PR or GitHub job.
After the physical operator result, any production activation/A/norm bridge and
same-input first-score/full-decoded comparison need separate gated integration.
Our source-exact protected vision and upstream F16 vision remain different
controls until actual upstream feature rows are captured and admitted.
