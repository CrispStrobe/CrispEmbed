# TeleOCR numerical diagnostics

These reproduce the completed vision-layer-17 and decoder-layer-2 investigations
using one original block at a time. They distinguish error propagated from a
previous layer from local native arithmetic and compare the original F32 FFN
with a double-precision evaluation of the same equations. They do not replace
the frozen reference, change production arithmetic, or qualify the full model.

The fixed comparison budgets remain maximum absolute error `0.002` and minimum
row cosine `0.99999`. Every numerical failure remains in the output. Both Python
probes and the native FFN executable return `1` when their recorded comparisons
exceed those budgets. In particular, a diagnostic exit of `1` is expected for
the recorded text case; it does not prevent the output files from being written.

## Inputs

Use the checkpoint at Hugging Face revision
[`b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d`](https://huggingface.co/XingChen-AGI/TeleOCR/tree/b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d).
The probes require its `model.safetensors`, `config.json`, and
`modeling_naviocr.py`. They verify the original module/configuration hashes
before executing the module and record hashes of the selected F32 weights and
every input array. Only one block's weights are materialized.

Install the pinned reference dependencies from the repository root:

```sh
python -m pip install -r models/requirements-teleocr-reference.txt
```

The examples below use these caller-selected directories:

- `teleocr-hf/`: the original checkpoint files.
- `reference/`: frozen output from `tools/dump_teleocr_reference.py`, including
  `text-stages/*.npy`, `text.gguf`, `text.json`, and `text.png`.
- `native-f32/`: output from `tools/run_teleocr_validation.py` with
  `--dump-native-stages`, including `text-native-stages/*.f32`.

If native boundary dumps are missing, capture them with the existing full-model
validator. Its strict failures and exit code must be retained:

```sh
python tools/run_teleocr_validation.py \
  --binary build/test-teleocr-diff \
  --model teleocr-f32.gguf \
  --reference-dir reference \
  --out native-f32 \
  --cases text --threads 2 --dump-native-stages
```

## Replay the two investigated blocks

Use new or empty output directories. Existing diagnostic results are never
overwritten. These commands use the same two CPU threads as the recorded probes.
Run each command separately so an expected failing numerical comparison does
not skip the next diagnostic.

```sh
python tools/teleocr_diagnostics/vision_layer17.py \
  --model-dir teleocr-hf \
  --reference-dir reference/text-stages \
  --native-dir native-f32/text-native-stages \
  --out diagnostics/vision17 --threads 2
```

```sh
python tools/teleocr_diagnostics/decoder_layer2.py \
  --model-dir teleocr-hf \
  --reference-dir reference/text-stages \
  --native-dir native-f32/text-native-stages \
  --out diagnostics/decoder2 --threads 2
```

Each writes `propagation.json`, `provenance.json`, and `probe.gguf`. The vision
probe also writes `original-f32-vs-f64.json`. Arrays retain the source inputs,
selected weights, F32 intermediates, and computed double-precision FFN output.
The double-precision array is diagnostic data; it is not substituted into the
frozen full-model reference. The two scripts replay the original block on both
frozen and native previous-layer outputs, preserving the distinction between
local differences and propagated differences.

## Replay the FFN with identical source input

Build the small native probe against the same pinned ggml used by the runtime:

```sh
cmake --build build --target test-teleocr-ffn -j2
```

Then run each source-generated FFN probe:

```sh
build/test-teleocr-ffn diagnostics/vision17/probe.gguf diagnostics/vision17/native
```

```sh
build/test-teleocr-ffn diagnostics/decoder2/probe.gguf diagnostics/decoder2/native
```

The executable uses F32 CPU matmuls, RMSNorm, SiLU, and the same FFN expression
as the runtime. It reports each intermediate's maximum error, worst coordinate,
reference/native values, row cosine, and count above `0.002`, then writes all
native intermediates as raw F32 files. Its fixed 256 MiB graph scratch supports
the recorded bounded text probes, with at most 1,024 input rows. It does not
load a full model. Exit codes are `0` for passing comparisons, `1` for numerical
failures, and `2` for invalid input or execution/I/O errors.

For the recorded CPU run, source-fed FFN maximum errors were `0.00244140625`
(vision layer 17) and `0.03515625` (decoder layer 2). The larger composed vision
gap was reproduced by the original block when given the native previous-layer
output. Hardware/kernel differences can change exact low-order bits; retain
the measured outputs and provenance rather than treating these numbers as new
tolerances. Full token IDs, EOS, decoded text, and every frozen stage remain
the separate responsibilities of `test-teleocr-diff`.

## Attribution

The original model module and the adapted FFN equations are Apache-2.0:
Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
The upstream source is
[`modeling_naviocr.py` at the pinned revision](https://huggingface.co/XingChen-AGI/TeleOCR/blob/b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d/modeling_naviocr.py).
See [NOTICE.teleocr](../../NOTICE.teleocr) and the
[complete Apache license](../../models/licenses/apache-2.0/LICENSE).

These tools are portable adaptations of the executed diagnostic scripts. The
original executed scripts and their initial results remain in the validation
evidence; the new entry points do not replace those historical records.
