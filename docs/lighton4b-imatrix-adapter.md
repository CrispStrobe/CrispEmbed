# LightOn4B external-text imatrix adapter

This tool adapts one authenticated public statistics file into CrispEmbed's input
format. It does not quantize weights or run a model. Python controls pass on the
real3.63MB file and synthetic negatives; an actual C++ consumer build/check is
pending on Kaggle CPU. No weighted model or quality result is admitted yet.

The source is
[richardyoung/LightOnOCR-3-4B-GGUF](https://huggingface.co/richardyoung/LightOnOCR-3-4B-GGUF/tree/8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01)
at8406c78c…, file`LightOnOCR-3-4B-imatrix.gguf`,3,626,496 bytes, SHA256
`10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7`.
Its496 F32 tensors are248 `.in_sum2`/`.counts` pairs: no embedding/output entry,
no vision entry, and one dense count per projection. Every count is66,048,
consistent with129 chunks of512. The publisher identifies calibration_datav3;
the corpus content hash and source F16 weights/quantizer command are not supplied.
The adapter therefore labels this external-text calibration, with corpus contents
and calibration source weights **not independently authenticated**. It is not
an OCR or heldout calibration claim.

Current pinned upstream semantics are explicit in
[common/imatrix-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/common/imatrix-loader.cpp),
[tools/quantize/quantize.cpp](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/tools/quantize/quantize.cpp)
and [the collector](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/tools/imatrix/imatrix.cpp):
`.in_sum2` is the raw F32 sum of squared projection inputs; `.counts` is stored
F32 and read as an integer count. The quantizer divides each dense vector by its
count. We reject zero/partial, fractional, nonfinite, negative and expert-count
inputs instead of filling missing data. Widths must exactly match all248 declared
LightOn4B projection input axes; padded or unknown domains reject.

CrispEmbed's older consumer expects a raw F32 sum vector at the exact weight name
and UINT64 metadata`count.<weight-name>`. Passing the upstream file directly used
to load zero vectors and fall back to unweighted quantization. An explicit
unusable`--imatrix` request now fails. Adapted files are further bound to a strict
profile/source/output hash, complete248-vector coverage and target header geometry;
missing or skipped application fails, and an adapted-profile output is published
only after all248 vectors were applied. Existing valid legacy imatrices retain
their partial-domain behavior. Protected vision stays unchanged in the adapted
LightOn profile. No weight quantization has been executed to qualify that future
path in this task.

## Input coordinates:24 inverse permutations

Upstream's
[Qwen3.5 converter](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/conversion/qwen.py)
reorders recurrent V heads from HF grouped order to tiled order. LightOn has16
key heads, two value heads per key and128 channels per value head. Its24
`ssm_out.weight` input vectors must be inverse-scattered:

```
native[(key*2 + sub)*128 + d] = upstream[(sub*16 + key)*128 + d]
```

This preserves the raw sum and count, changing only input coordinates. Other
projection input axes are unchanged: QKV/Z/alpha/beta output-row reorders do not
permute their2560 hidden input columns; FFN inputs remain in the same order.
No padding is accepted. Tests reconstruct grouped markers0…4095 from an
independently generated tiled vector and verify all248 normalized vector hashes.

Matching statistics does not make newly quantized payloads identical to the
publisher: block quantization is not invariant to input-column permutations.
A same-weight cross-runtime test must preserve effective weights by adapting
activation/layout coordinates, not dequantizing and requantizing column blocks.

## Reproduce the bounded checks

```sh
python tools/lighton4b_imatrix_adapter.py \
  --input LightOnOCR-3-4B-imatrix.gguf --output adapted.gguf --receipt adapter.json
python tests/test_lighton4b_imatrix_adapter.py --source LightOnOCR-3-4B-imatrix.gguf
```

The deterministic adapted file is3,621,184 bytes, SHA256
`105aba299aff48c63b43323bf6ec4f02d20f2fbb726b1ece0728e9e7fbb24663`.
Its profile is`lighton4b-text-inputs-hf-grouped-v1`. Output/receipt paths must be
new and distinct. Failed adaptation retains a failure receipt and removes only
its own temporary output.

One scoped hosted controller builds only`crispembed-quantize`, adapts the public
statistics, checks a723-tensor header without payloads, and runs
`--check-lighton4b-imatrix`. It compares the actual C++ consumer's248 normalized
F32 hashes to the Python means and checks F32/F64 division agreement. Wrong
width/name, tampered statistics and the real unadapted source must reject without
publishing a model. This route does not enter weight quantization, collect
activations, instantiate a reference model or read an OCR case. Artifacts stay
remote90days; no new model or PR is created by these checks.


The queued GitHub control42238407232 was cancelled at the user's request before
it started; it produced no artifacts or test result. The same checks are ported
to a CPU-only Kaggle bootstrap at `tools/kaggle/lighton4b-imatrix-adapter/kernel.py`.
Replace only its source token with the reviewed40-character commit and instantiate
the example metadata under the configured account outside Git. GPU/TPU and all
data sources stay disabled; internet is read-only source/statistics transport.
The bootstrap verifies its exact source and3a27 ggml pin, records RAM/swap/disk/load,
uses two build threads and a20-minute controller deadline within a30-minute kernel.
Only its fresh transient checkout/build directories are cleaned; all statistics,
receipts, logs and negative results remain in the kernel output. Account/access
references remain in private environment notes, never this public handoff.


The first Kaggle CPU attempt compiles the real consumer and passes all six Python
checks, but rejects a target-header fixture that mistakenly kept the original
Conv3D five-dimensional axes. Format5 stores that patch as `[1536,1024]`; only
this fixture geometry is corrected, with an added723-header rank/patch regression
control. The adapter output/mean data are unchanged. Initial failure receipts
and logs remain preserved. A bounded CPU retry uses the same source/statistics
guards and still executes no weight quantization or model forward.
