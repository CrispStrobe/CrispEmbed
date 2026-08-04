# Qwen2.5-VL-7B reference transcripts

One `.txt` per fixture, produced by the reference PyTorch stack under the exact
contract the native VL lane (`src/qwen2vl_ocr.cpp`) implements. These are the
gold a CER gate for that lane should read.

`manifest.json` pins what produced them — model id, resolved revision, prompt,
the applied chat template, decoding params, dtype and hardware. Per-corpus
manifests sit alongside the transcripts. If any of those change, the transcripts
are stale and the gate is measuring the wrong thing.

## What makes these gold rather than "some model output"

A VL model is only an OCR engine because of its prompt, so the prompt string
here is byte-for-byte the one the native lane uses, the chat wrapper comes from
the checkpoint's own template (not hand-rolled), decoding is greedy, and the
preprocessor's pixel bounds are untouched — those bounds set the effective
resolution the model reads at, so overriding them would make the numbers
unquotable against the published checkpoint.

## Coverage

| corpus | fixtures | transcripts | mean CER vs ground truth |
|---|--:|--:|--:|
| `synth/` | 20 | 20 | 0.0 |
| `cc0/` | 5 | 3 | 0.043 (over the 3) |

Two CC0 fixtures have no transcript: `commons_test_ocr_document.jpg` and
`german_official_print.jpg`. Both are ~4.8 Mpix scans, which the preprocessor
turns into ~6k vision patches. On the reference hardware (2x Tesla T4, compute
capability 7.5) SDPA has no memory-efficient kernel for the vision tower's mask
and falls back to the math path, which materialises the score matrix — a single
3.98/4.07 GiB allocation. Five weight placements and three host-memory retries
were measured; the OOM only ever moved to whichever device held the late vision
blocks. See `tests/results/ocr_parity_qwen25vl_2026-08-04.json` for the
per-device residency and peak figures.

**TODO:** re-run those two fixtures on a single Ampere-or-newer card with >=24 GB
(L4/A100), where the vision tower is not split across devices and SDPA can use a
memory-efficient kernel. Until then a CER gate must skip them rather than treat
a missing transcript as a failure.

## Regenerating

See `tools/kaggle/qwen-vl-parity/README.md`. The reference does not run on the
dev Mac: 15.45 GiB of 16-bit weights against 16 GiB of total unified memory.
