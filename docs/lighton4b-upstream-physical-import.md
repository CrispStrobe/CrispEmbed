# Physical upstream Q4_K_M import: completed layout/operator audit

The goal is a native comparison using the same represented upstream text weights,
without requantization or a full F32 model. This is separate from the new weighted
HF-layout recipe: equal type schedules do not imply equal quantized values.
No inference result or runtime import is claimed yet.


Authorized Kaggle CPU version1 completes at source
`5199609f4425b006bb78f488703b42ec4410f3ac`. All249 actual Q4/Q6 matrices decode
**bit-exact F32** in native ggml and independently pinned upstream NumPy code.
96 whole-row inverse permutations roundtrip without changing quant blocks.
All168 predeclared out-projection real-valued controls match exactly after the
activation permutation; leaving it out gives absolute errors0.00356691–2.58265.
This is represented-weight/layout proof, not model-inference or float-reduction
parity. No source graph, decoder or original model is executed.

| Actual small-parameter class | Tensors | Differences versus F32 transform | Inverse values lost |
|---|---:|---:|---:|
| Shifted RMS (+1) |81|0|4 (max2.98023224e−8)|
| Pretransformed A (−exp) |24|304 values versus NumPy exp (max3.81469727e−6)|106 (max8.94069672e−8)|
| V-channel conv inverse |24|0|0|
| Per-head dt inverse |24|0|0|
| Unshifted linear-attention norm |24|0|0|

The actual RMS/A inverse losses prohibit treating float log/subtraction as a
lossless conversion. The narrowly gated importer must consume source gamma and
pretransformed A directly, along with the out-activation bridge. The A exp
comparison is to NumPy F32 arithmetic, not a reproduced original PyTorch exp;
the physical source values remain authoritative and unchanged for import.

The [result](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-audit/physical-audit.json)
SHA256 is `a65e1652448a65538b06cbd1897773a3df802be1eca040e12b842db0a991fe2b`.
The [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-audit/independent-review.json)
rechecks all249 native TSV/independent dequant hashes, all426 frozen name/shape/type
mappings,168 dot controls/negative controls and aggregate inverse-loss counts.
All25 small remote outputs total739,734 bytes; no model/array payload is downloaded
locally. Exact submitted [credential-free bootstrap](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-audit/kaggle-bootstrap.py)
SHA256 `b97bb944009daca968a97eecdc84fba1319538856bdc809a0082c280b1cdab9f`
is independently authenticated through the service API; private service binding
stays outside Git. Reproduction source/helper/test are pinned at
[5199609f4](https://github.com/CrispStrobe/CrispEmbed/tree/5199609f4425b006bb78f488703b42ec4410f3ac).
Effective preflight RAM31,510,732,800 bytes, zero swap,1.037TB scratch and20.94GB
output satisfy the predeclared guards. Only65,536 values are decoded per chunk;
no full F32 model or requantization occurs. Runtime import and same-weight decoded
comparison remain the next separately verified step.

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

## Format6 runtime and no-checkpoint controls

Kaggle CPU controls version2 passes at source
`0610af26d9cb8972ecdfe55204e2306e9e740d5e`, based on stable PR15569abebb7e.
The [frozen protocol](lighton4b-upstream-physical-controls-v1.json) and
[complete result](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-controls/physical-controls.json)
cover557,857 physical dtype/gate/metadata/lane/scalar checks,1,147 HF-weighted
profile controls,713 oldQ6,84 tied-head,39 public-fence (four actual header-only
public rejections),9 old native,74 mixed,20 BF16-linear and16 BF16-weight checks.
Six unique Python transport/header/mapping cases pass. Actual native-document and
preflight binaries build. No checkpoint/model payload or learned-model inference
is used. The synthetic grouped core proves direct−1 equals legacy−exp(0) bitwise
for values/state, an incorrectly re-exponentiated physical A differs, and gate-off
legacy repeat remains bit-exact. Full16-token-chunk/4096-lane permutations pass.

The source now has a separate experimental format6/type-tag/domain: direct gamma
and A, HF recurrent state with tiled pre-out inputs, explicit incompatible BF16
cast rejection and cache-domain tags. It retains every format5 default. The
physical tokenizer metadata mirrors upstream CONTROL suppression and dual EOG
248044/248046; original format5 retains primary248046 only. Vocabulary audit
finds248,077 identical real strings/merges/template and243 upstream UNUSED labels
that decode empty like native empty rows. Six FIM tokens are additionally CONTROL
in the upstream profile, so only format6 changes their suppression. No output
fixture chooses those rules. Native Q6 is required and WASM remains unsupported.

The lossless importer copies297 protected vision tensors and transports426
physical text tensors using whole rows/blocks; it does not create a full F32
model or call a quantizer. Public loaders reject format6 even when experimental
gates are on, and no alias/payload/default/release is added. Current controls do
not yet prove a persisted full imported candidate or decoded quality. Next is a
coordinated Kaggle same-feature/source-provenance harness and actual composed
form/table/Arabic decodes against the retained upstream and BF16 references.

The [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-controls/independent-review.json)
rehashes all19 reported runtime source blobs against the exact Git tree and all30
small remote outputs (149,915 bytes). Submitted
[bootstrap](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-controls/kaggle-bootstrap-v2.py)
SHA256 is `990cee747ec07da9ecd341dd680dfb1b411c604e1ea79a08c0323df805867283`,
verified through the service API; private service binding stays outside Git.
Initial version1's missing protocol path is preserved: it stops after dependency
checks and before compilation/model access. The corrected protocol/path was
validated before version2. No GitHub job or duplicate worker is launched.

## Actual persisted physical import and initial decoder setup negative

The [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v1-setup-negative/independent-review.json)
preserves all 24 small outputs (646,619 bytes) from Kaggle CPU version1 at
`a6644247c48cb25e1b7e4b8da9e8ff91644eaca7`. Both native and pinned upstream
executables built. The actual lossless import verified every one of 723 persisted
tensor spans and produced 4,041,905,760 bytes, SHA-256
`843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`.
The census is 474 F32 protected/scalar tensors, 216 Q4_K and 33 Q6_K matrices;
96 whole quantized row permutations and 24 unchanged GDN out projections are
recorded. All 19 runtime/packer Git blobs remain exactly the passing component
control source.

The worker then stopped on missing PyYAML while importing pinned upstream
`gguf-py`, before any original, upstream or native forward. This is setup and
transport evidence, not a first-score or decoded-quality result. A separate
primary-source routing review also found that the pinned server bypasses the
initial CLI feature-capture site; the corrected hook targets the shared
`mtmd_helper_decode_image_chunk` encoded-pointer boundary. The bounded retry
pins the previously verified audit dependencies, imports them before transfers,
and requires the same complete candidate file hash. Its runtime behavior and
source-feature-only scope remain unchanged and default-off.

## Full same-feature result

The [complete three-case decoded comparison](lighton4b-upstream-physical-decoded-results.md)
now records actual source-controlled outputs: form174/EOS fails the character
edit floor (202 versus llama.cpp189), table256/cap remains incomplete despite
smaller edit counts, and Arabic6/EOS is exact. Exact physical text weights and
actual prefill bytes do not establish whole-runtime parity. Both experimental
gates remain off; component controls and future arithmetic ablations must not
be promoted from this negative result.
