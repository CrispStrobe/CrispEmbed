# LightOn4B imatrix adapter: verified statistics, no weighted model yet

Kaggle CPU verification now passes the real public file, all248 consumer means
and the ordinary CLI rejection/legacy-load paths. Exactly24 recurrent output input
vectors are inverse-permuted. No weights were quantized, no model ran, and this
is not an image-trained imatrix or an OCR-quality result.

The tested source is
[`732942355`](https://github.com/CrispStrobe/CrispEmbed/tree/73294235567b93a1bda9290c9248805dc5309cf6).
The [evidence index](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-adapter/evidence.json)
preserves all four CPU attempt receipts/logs, plus the cancelled queued GitHub
check42238407232, which ran no tests and produced no artifacts. Kaggle versions
and sources are:

| Version | Source | Result |
|---|---|---|
| 1 | fe2ba10d0 | C++ builds and six Python controls pass; synthetic target incorrectly retains5D Conv3D source shape |
| 2 | 911d70798 | Correct format5 patch `[1536,1024]`; all248 actual consumer means and negatives pass |
| 3 | 1f0d16a23 | Production adapted coverage rejects correctly; tiny legacy fixture lacks required32-byte payload tail padding |
| 4 | 732942355 | Both fixture regressions fixed; eight Python controls and all production-path checks pass |

All four builds use the same consumer binary SHA256
`747ae3ffd0b0f49985d014f0857786987a4bff987ffee3fa36dfd945c307adec`.
The final bootstrap is
`c9ae260114fa1d2a1f11f24413500c4489434051c7cd5b0fc22f9423e63b9f56`.
The service slug is `lighton4b-imatrix-adapter-cpu-controls`; account/access details
stay in private environment notes. Source statistics and converted files remain
in versioned remote kernel output; small reports/logs alone are mirrored here.
There is no claimed service expiry. GPU and TPU were disabled.

The source
[publisher snapshot8406c78c](https://huggingface.co/richardyoung/LightOnOCR-3-4B-GGUF/tree/8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01)
has248 dense F32 sum/count pairs. Its3,626,496-byte imatrix SHA256 is
`10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7`.
Every count is66,048=129×512. The deterministic3,621,184-byte adapted result is
`105aba299aff48c63b43323bf6ec4f02d20f2fbb726b1ece0728e9e7fbb24663`.
All897,024 normalized F32 values match the actual C++ consumer, including the
98,304 coordinates in24 inverse-permuted4096-channel vectors.

The [upstream collector](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/tools/imatrix/imatrix.cpp)
writes raw squared-input sums and counts. Its
[quantizer loader](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/tools/quantize/quantize.cpp)
divides each dense vector by its count. The actual consumer checks F64/count→F32
against upstream F32/count byte-for-byte. We require the one complete positive
integer count per dense tensor, rejecting fractional/nonfinite/zero counts,
expert/padded/foreign domains, wrong widths and missing/extra vectors.

The [Qwen converter](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/conversion/qwen.py)
tiles V heads for16 key heads×2 values×128 channels. Only24 `ssm_out` importance
input axes need inverse scatter from `(sub*16+key)*128+d` to
`(key*2+sub)*128+d`. Other changed weight rows do not change their input columns.
The embedding/head has no statistic and remains an explicit unweighted exception.
No image/vision statistic is present. The corpus filename is publisher-declared
calibration_datav3; its content hash, underlying text license and calibration
source-weight revision are not independently verified. The publisher repository
declares Apache-2.0, which is not evidence of the underlying corpus's provenance.

The ordinary `--imatrix` path now errors on an unusable request rather than
silently using unweighted quantization. Valid old partial imatrices still load.
The adapted-profile opt-in requires its exact source/output identity, complete
248-vector coverage, unquantized target matrix shapes and protected F32 counts.
Wrong target coverage is rejected before output publication. The future weighted
output path additionally checks248 actual applications and uses an unpublished
partial file until success. That positive weight-quantization path has not run
in this verification.

## Why the publisher comparison is not our plainQ4 baseline

A bounded [header census](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-adapter/header-census.json)
reads no complete external model payload. Whole external-model SHAs are LFS
metadata declarations, not locally recomputed hashes.

| Model/container | Tensor types | Bytes |
|---|---|---:|
| Our combined plainQ4f873 | 474F32 +248Q4_K +1Q8_0 | 4,030,957,952 |
| Publisher Q4_K_M decoder | 177F32 +216Q4_K +33Q6_K | 2,708,804,576 |
| Publisher Q6_K decoder | 177F32 +249Q6_K | 3,464,055,776 |
| Publisher separate projector | 198F32 +100F16 | 672,423,200 |

The publisher's32 nonembedding Q6 names map exactly to our preserved32-Q6 rule,
while its tied head is Q6 rather than our Q8. It is not our24-output Q6 policy.
The projector has298 tensors versus our297 because it splits the temporal patch
weight into two4D convolution kernels; total vision parameter count agrees.
Most combined-size difference comes from vision F32 versus F16
(661,651,456 bytes). Publisher Q6 head saves153,958,400 bytes while32 mixed
promotions add164,904,960 bytes. Changing the file label cannot produce this policy.

Our immutable
[producer94ad7bda](https://github.com/CrispStrobe/CrispEmbed/blob/94ad7bda378ac285f03fda28dcb97d29cb496418/tools/ci-heavy/lighton4b_convert_quant.py)
uses plain `q4_k`, protected `v.*=f32` and no imatrix. The publisher reports
text-only F16-relative wikitext2 KL divergence. Its Q4_K_M and Q6_K values .0210
and .0042, and top-token matches92.8%/96.6%, do not measure our OCR cases, original
BF16 baseline or same-weight runtime error. The reported invoice check is not
broad OCR qualification.

## Bounded next experiment and fair runtime comparison

After source integration, one reviewed Kaggle-only experiment can generate a
weighted **plainQ4** candidate while preserving every tensor type, protected
vision and Q8 tied head. Use the authenticated original unquantized F16 carrier,
SHA757ad5d0…/9,757,493,600 bytes, or the original BF16 checkpoint, never quantized
Q4/Q8 donors. A fresh unweighted control must reproduce frozenf873 before any
importance-weighted result is admitted. Then require248 actual vector applications,
all723 shape/type hashes, unchanged474 protected/head payloads and explicit
external-text provenance. Retain the frozen table/Arabic calibration inputs,
precision gates, complete first-score rows and independent decode/EOS/cap limits.
The source corpus license/content limitation remains; no public model upload or
quality/default/release promotion follows. No such model job has started here.

A fair cross-runtime A/B needs the **same effective weights**, token IDs, positions,
causal prefix, EOS and greedy rule. It should first feed identical retained merged
vision embeddings to skip projector precision/preprocessing differences. Publisher
GDN V row permutations can be reversed as row reorders; output input-column
permutations cross256-weight quantization blocks and cannot generally be repaired
by reencoding without changing weights. Preserve raw quantized rows and adapt
activation coordinates at the projection boundary instead. A source original-BF16
same-weight oracle then distinguishes effective-weight BF16 rounding, dot/reduction
arithmetic and model quantization loss. Default llama quantized activation dot
arithmetic versus our F32-dequantized dot is a runtime difference to measure,
not automatically an implementation defect. Numerical comparisons must remain
same-prefix; complete independent decoded outputs are separate evidence.

The [source-revision correction receipt](../tests/results/qwen35-ocr/lightonocr3-4b-publisher-declared-source-revision.json)
records that the publisher Q4_K_M header `general.name` declares
`B71010F095Dc3735De7Ee6B9969Bff8F8F50B39A`, matching the official b710 reference.
This is a header declaration, not independent source tensor-content verification.
Producing converter/quantizer commit and corpus provenance remain unverified;
all earlier raw receipts and hashes are preserved.
