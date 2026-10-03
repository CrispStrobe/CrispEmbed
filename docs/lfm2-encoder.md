# LFM2.5-Encoder-230M

CrispEmbed runs the official
[LiquidAI GGUFs](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M-GGUF)
directly, using its bidirectional LFM2 backend. The 14-layer backbone combines
ShortConv and grouped-query attention, with 1024-dimensional final features
and a tied 65536-entry masked-LM head. No conversion is needed for these files.

## Usage

```sh
crispembed -m LFM2.5-Encoder-230M-F16.gguf --fill-mask \
  "The capital of France is [MASK]."
crispembed -m LFM2.5-Encoder-230M-F16.gguf --tokens-raw --json "hello world"
crispembed -m LFM2.5-Encoder-230M-F16.gguf --capabilities --json
```

Registry names are `lfm2-encoder-230m` (F16), `lfm2-encoder-230m-f16`,
`lfm2-encoder-230m-q8`, and `lfm2-encoder-230m-q4`. Downloads use the existing
LFM license acknowledgement and SHA-256 verification. Query/document prefixes
are absent because this checkpoint is a general-purpose masked encoder,
not the retrieval-trained LFM2.5-Embedding model.

```python
from crispembed import CrispEmbed

model = CrispEmbed("LFM2.5-Encoder-230M-F16.gguf")
ids, hidden = model.encode_tokens("hello world", normalize=False)
# hidden: [tokens, 1024], including tokenizer-added BOS
ids, normalized = model.encode_tokens("hello world")
positions, logits = model.masked_logits("The capital of France is <|mask|>.")
predictions = model.fill_mask("The capital of France is [MASK].", top_k=5)
# predictions[0]["predictions"][0]["token"] == " Paris"
```

The C ABI exposes `crispembed_encode_tokens_raw`, `crispembed_encode_tokens`,
`crispembed_last_token_ids`, `crispembed_has_masked_lm`,
`crispembed_masked_logits`, and `crispembed_token_bytes`. Returned C buffers
belong to the context; copy results before the next inference call. Python,
Rust, and Dart wrappers copy returned arrays. Rust provides
`encode_tokens_raw`, `masked_logits`, `fill_mask`, and `token_bytes`; Dart
provides `encodeTokens(normalize: false)`, `maskedLogits`, `fillMask`, and
`tokenBytes`.

`masked_logits` accepts the literal `<|mask|>` token. CLI `--fill-mask` and
wrapper `fill_mask` / `fillMask` also translate `[MASK]`. Multiple masks are
predicted jointly in one bidirectional forward, with positions including BOS.
The output is unnormalized logits at mask positions only; the head consumes
raw hidden states. Top-K probabilities use the full vocabulary softmax. A token
may represent incomplete UTF-8 bytes; display decoders replace invalid UTF-8
as Hugging Face does. The GGUF contains 1134 UNUSED padding vocabulary slots;
these decode to empty strings rather than invented `[PADnnn]` text.

Ordinary `encode` still returns the L2-normalized first-token vector. It is
available for experimentation, but this checkpoint has not been trained to
make those vectors suitable for semantic retrieval. It has no ColBERT head.

CPU encoder APIs default on for GGUFs declaring a non-causal LFM2 model and
mask token. `CRISPEMBED_LFM2_ENCODER=0` disables these APIs and restores the
historical ordinary-text tokenizer for bisection. Other LFM2 checkpoints and
GPU encoder graphs require `CRISPEMBED_LFM2_ENCODER=1`. GPU scheduling has not
yet passed the reference suite, so it remains opt-in. Existing pooled and
ColBERT paths remain available. The model metadata declares 128000 positions;
the validation suite reaches 649 tokens and does not establish quality at the
metadata limit. Native token/logit APIs reject sequences exceeding metadata
length rather than silently truncating.

## Reference validation

The reference uses the checkpoint's `Lfm2BidirectionalForMaskedLM` class,
FP32 weights, eager bidirectional attention, and the official symmetric
ShortConv patch. Hooks capture embedding lookup, all 14 layer outputs, final
RMSNorm, raw/normalized first-token vectors, and mask logits. Production API
comparisons also check exact token IDs, vocabulary decoding, normalized/pooled
consistency, repeated mixed calls, and cleared failure outputs.

The 15-case suite includes English, German, French, Japanese/Chinese, Arabic,
Unicode digits, whitespace, adjacent masks, literal special tokens, an empty
string, and a 649-token input. Quantized results are measured against the same
FP32 reference separately from the F16 port-correctness gate.

CPU results (2026-10-03), against Torch 2.11.0 / Transformers 5.13.0.dev0,
FP32 eager reference. Relative error is `||native-reference||₂ / ||reference||₂`,
taking the worst case; cosine is the worst token/mask row across the suite.

| GGUF | Hidden min cosine | Hidden max relative error | Logits min cosine | Logits max relative error | Mask top-1 matches |
|---|---:|---:|---:|---:|---:|
| F16 | 0.99998947 | 0.1813% | 0.99999957 | 0.0976% | 15/15 |
| Q8_0 | 0.94665218 | 7.2592% | 0.99934877 | 3.6796% | 13/15 |
| Q4_0 | 0.73603180 | 79.9290% | 0.93175417 | 56.8220% | 9/15 |
| Q4_K (CrispEmbed) | 0.55631144 | 42.5520% | 0.97438016 | 41.5711% | 10/15 |
| Q4_K + imatrix (CrispEmbed) | 0.75290679 | 28.2212% | 0.98749220 | 15.8073% | 11/15 |

F16 is the default because it passes the port-correctness gate and preserves all
15 decoded mask predictions. Q8_0 and Q4_0 execute correctly but their quantized
weights change predictions; Q4_0 also shows substantial hidden-feature drift.
All five match the reference token IDs and decode all 65536 vocabulary slots.
The F16 per-layer replay passes all 297 checks over 15 inputs, including the
production scheduler path. Live Rust and Dart examples exercise the same C ABI as Python.

Additional cosine metrics, each taking the worst of the same 15 inputs. CLS is
the first-token vector; its cosine is unchanged by L2 normalization. Whole-tensor
cosine compares all hidden elements together, while row mean weights each token
equally. These are reference-agreement measurements, not retrieval/task accuracy.

| GGUF | Size (MB) | CLS min cosine | Hidden min global cosine | Hidden min mean-row cosine |
|---|---:|---:|---:|---:|
| F16 | 461.9 | 0.99999924 | 0.99999843 | 0.99999934 |
| Q8_0 | 246.6 | 0.99353220 | 0.99737641 | 0.99823156 |
| Q4_0 | 149.1 | 0.85402948 | 0.87691055 | 0.86674437 |
| Q4_K (CrispEmbed) | 165.3 | 0.93585145 | 0.92115994 | 0.93970592 |
| Q4_K + imatrix (CrispEmbed) | 165.3 | 0.97074061 | 0.96062564 | 0.96542450 |

Q4_K is a footprint tradeoff, not a parity-preserving default. Importance-matrix
calibration improves CLS/global cosine and mask agreement versus plain Q4_K and
official Q4_0, but substantial per-token and norm errors remain. Keep F16 when
raw features or Python-reference parity matter. Q8_0 is a closer approximation
at a larger size. The mask suite includes stress inputs such as adjacent masks
and reserved tokens; its top-1 counts are agreement, not a labeled quality score.

`test-lfm2-diff` uses `crispembed_diff::Ref`. F16 passes all 297 checks over the
15 inputs at the 0.999 worst-row threshold; its lowest intermediate-layer cosine
is 0.999985 (layer 13, long input). Quantized probes on English, Japanese/Chinese
and the long input fail that strict port threshold. Their exact shapes/token IDs
still match, and the reference comparison shows drift accumulating through layers.
The 649-token probe is particularly informative:

| GGUF | Layer 0 min cosine | Layer 13 min cosine | Final norm min cosine | Final norm global cosine |
|---|---:|---:|---:|---:|
| F16 | 1.000000 | 0.999985 | 0.999989 | 0.99999946 |
| Q8_0 | 0.999947 | 0.946592 | 0.946652 | 0.99850363 |
| Q4_0 | 0.996677 | 0.666887 | 0.736032 | 0.90467989 |
| Q4_K (CrispEmbed) | 0.997529 | 0.432208 | 0.556311 | 0.93648052 |
| Q4_K + imatrix (CrispEmbed) | 0.998555 | 0.537307 | 0.752907 | 0.96062565 |

CrispEmbed's Q4_K artifacts use 82 Q4_K backbone matrices, one Q8_0 tied
embedding/LM matrix, and 49 unchanged F32 norm/ShortConv tensors. Official Q4_0
instead uses 82 Q4_0 matrices and a Q6_K tied embedding matrix. Both local Q4_K
files are 165331968 bytes. Calibration uses 134 separate EN/DE/code/structured
corpus sentences plus 17 eight-sentence groups (6711 tokens); none of the held-out
texts occurs in that corpus. All 82 backbone matrices receive importance vectors.
This small corpus does not establish a best possible multilingual quantization.

Full per-case metrics, decoded prediction changes, per-layer probe values and
artifact SHA-256 hashes are in
[the result manifest](../tests/results/lfm2-encoder/quantization.json).

To reproduce, keep weights/references outside the repository:

```sh
USE_TF=0 python tools/dump_lfm2_reference.py \
  --model "$LFM2_HF_DIR" --texts-file tests/lfm2_encoder_cases.json \
  --long-context --output "$LFM2_REF_DIR"
python tests/lfm2_encoder_parity.py \
  --model "$LFM2_F16" --refs "$LFM2_REF_DIR" \
  --lib build/libcrispembed.so --require-top1
build/test-lfm2-diff "$LFM2_F16" "$LFM2_REF_DIR/000.gguf" \
  "The capital of France is <|mask|>."
# Quantization quality: report drift and decoded changes without declaring parity.
python tests/lfm2_encoder_parity.py \
  --model "$LFM2_QUANT" --refs "$LFM2_REF_DIR" \
  --lib build/libcrispembed.so --measure-only
```

The converter `models/convert-lfm2-embed-to-gguf.py` also accepts the original
masked encoder checkpoint: it strips `lfm2.` weight prefixes, honors
`block_auto_adjust_ff_dim=false`, and writes special-token/mask metadata. Use
the official GGUFs when a fresh conversion is unnecessary. A fresh F16 conversion
also passes the complete 15-case reference suite and the 20-check first-case
per-layer replay.

## Build and assess your own Q4_K

Start from the F16 file, rather than requantizing Q4_0. Plain Q4_K:

```sh
build/crispembed-quantize "$LFM2_F16" "$LFM2_Q4K" q4_k
```

The calibrated candidate uses a fresh importance file (the native collector
merges existing statistics, so the calibration helper rejects an existing output):

```sh
python tools/calibrate_lfm2_encoder.py --model "$LFM2_F16" \
  --corpus tools/kaggle/crispembed-imatrix-quant/calib_corpus.jsonl \
  --output "$LFM2_IMATRIX" --lib build/libcrispembed.so
build/crispembed-quantize "$LFM2_F16" "$LFM2_Q4K_IMATRIX" q4_k \
  --imatrix "$LFM2_IMATRIX"
python tests/lfm2_encoder_parity.py --model "$LFM2_Q4K_IMATRIX" \
  --refs "$LFM2_REF_DIR" --lib build/libcrispembed.so --threads 4 \
  --measure-only --output "$LFM2_REPORT"
build/test-lfm2-diff "$LFM2_Q4K_IMATRIX" "$LFM2_REF_DIR/000.gguf" \
  "The capital of France is <|mask|>."
```

The quantized diff intentionally retains the strict 0.999 port gate: its failed
cosine/norm/prediction checks are expected measurements, not a successful parity
claim. Use the mean/global cosine, norm error and decoded agreement together.
Local Q4_K files load directly through the same APIs and CLI as official GGUFs;
no download alias or default model selection was changed by this experiment.


## Calibrated mixed precision

`crispembed-quantize` accepts repeated `--tensor-type GLOB=TYPE` overrides
(`f32`, `f16`, `q8_0`, `q6_k`, `q5_k`). Patterns use `*` and `?`; the last
matching override wins for eligible matrices. Norms, biases and ShortConv
kernels retain their preservation rules. A pattern matching no tensor is rejected
before opening the output file. Tensor dimensions can require a quantization
fallback, so inspect actual stored types rather than assuming the requested type.
The small GGUF integration test and CPU CI verify these behaviors.

For this experiment we converted the original checkpoint to FP32 GGUF, then
collected fresh statistics over 174 separate corpus records and 11 groups:
185 samples, 12065 tokens, longest sample 1149 tokens. The supplemental corpus
includes multilingual, masked, code, structured and Unicode inputs. All 82
backbone importance vectors match this source's tensor names and dimensions.
No exact evaluation input occurs in calibration; the reference suite was used
to select precision policies, so these scores are not an independent task benchmark.
The FP32 conversion uses `lfm.*` tensor names; importance vectors from the official
`blk.*` GGUF cannot be reused with it.

```sh
USE_TF=0 python models/convert-lfm2-embed-to-gguf.py \
  --model "$LFM2_HF_DIR" --output "$LFM2_F32" --dtype f32
python tools/calibrate_lfm2_encoder.py --model "$LFM2_F32" \
  --corpus tools/kaggle/crispembed-imatrix-quant/calib_corpus.jsonl \
  --extra-corpus tools/lfm2_encoder_calibration_extra.jsonl \
  --group-size 16 --threads 4 --lib build/libcrispembed.so \
  --output "$LFM2_EXPANDED_IMATRIX"
# Compact mixed Q4_K: upgrade attention, ShortConv projections and FFN down.
build/crispembed-quantize "$LFM2_F32" "$LFM2_MIXED_Q4" q4_k \
  --imatrix "$LFM2_EXPANDED_IMATRIX" \
  --tensor-type '*.attn*weight=q8_0' \
  --tensor-type '*conv.*_proj.weight=q8_0' \
  --tensor-type '*.ff.w2.weight=q8_0'
```

Attention was the strongest single-group F16 upgrade for masked prediction
agreement in the original Q4_K ablation. Combining attention, ShortConv input/output
projections and FFN down projections gives a stronger feature approximation.
Keeping the tied embedding/head in F16 costs approximately 63 MB over Q8_0 and
does not consistently improve token minima or decoded agreement.

The local CPU F16 matrix kernel uses F32 activations. K-quant matrix kernels
use Q8_K temporary activations, and Q8_0 kernels use Q8_0 temporaries.
Consequently, these upgrades change both weight storage precision and the
activation arithmetic. Quality is not monotonic as individual groups change.
Importance weighting applies to remaining K-quant weights; Q8_0 quantization
ignores importance vectors.

Across 26 screened profiles, the two useful size/precision choices are:

| Profile | Size (MB) | CLS min cosine | Token min cosine | Hidden min global cosine | Hidden max relative error | Mask agreement |
|---|---:|---:|---:|---:|---:|---:|
| Expanded-calibration Q4_K | 165.3 | 0.968517 | 0.756988 | 0.965374 | 29.95% | 11/15 |
| Mixed Q4_K, selected operators Q8_0 | 209.9 | 0.980123 | 0.881376 | 0.987888 | 15.64% | 13/15 |
| Mixed Q8_0, selected operators F16 | 330.2 | 0.998998 | 0.987746 | 0.999588 | 2.90% | 14/15 |
| Official F16 | 461.9 | 0.999999 | 0.999989 | 0.999998 | 0.18% | 15/15 |

The selected operators are all attention matrices, ShortConv input/output
projections and FFN down projections. The compact model stores 54 such matrices
in Q8_0, 28 FFN gate/up matrices in calibrated Q4_K, and the tied embedding/head
in Q8_0. The higher-fidelity model stores those 54 matrices in F16, the remaining
28 gate/up matrices and tied embedding/head in Q8_0. Both preserve all 49 F32
norm/kernel tensors byte-for-byte from the source.

Generate the 330 MB profile with:

```sh
build/crispembed-quantize "$LFM2_F32" "$LFM2_MIXED_Q8" q8_0 \
  --tensor-type '*.attn*weight=f16' \
  --tensor-type '*conv.*_proj.weight=f16' \
  --tensor-type '*.ff.w2.weight=f16'
python tests/lfm2_encoder_parity.py --model "$LFM2_MIXED_Q8" \
  --refs "$LFM2_REF_DIR" --lib build/libcrispembed.so --threads 4 \
  --measure-only
```

Upgrading gate/up matrices in layers 8, 9, 10 and 13 adds about 20 MB to the
330 MB profile without improving its worst token cosine or mask agreement.
The compact model changes the second adjacent-mask prediction and the second
Japanese/Chinese mask. The higher-fidelity model changes only the latter:
Python predicts `京都`, while this mixed model predicts `東京`. These are model
predictions on stress probes, not labeled factual answers.

All screened profiles, source/calibration provenance, selected artifact hashes,
full API contract measurements and layer probes are recorded in
[the mixed-precision manifest](../tests/results/lfm2-encoder/mixed_precision.json).
These mixed models improve approximation but do not pass the strict 0.999
worst-row port gate; F16 remains the default when Python parity is required.


The full 15-case replay passes exact IDs, all 65536 decoded vocabulary entries,
normalized/pooled consistency, repeated calls and cleared failure contracts for
both selected models. With the strict per-layer threshold unchanged, the 330 MB
model's English probe passes 20/20 checks; Japanese/Chinese passes all numerical
checks but fails decoded top-1 agreement (20 pass / 1 fail). The long probe reports
12 pass / 8 fail, with final token cosine 0.987746: gate/up drift first crosses
0.999 at layer 8 and grows further at layers 9 and 13. The compact model's strict
diff reports 6/17, 7/14 and 5/15 pass/fail respectively. Counts include norm and
decoded-output failures, not only cosine checks.
