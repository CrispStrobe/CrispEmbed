# Qwen2.5-VL-7B reference transcripts

One `.txt` per fixture, produced by the reference PyTorch stack under the exact
contract the native VL lane (`src/qwen2vl_ocr.cpp`) implements. These are the
gold a CER gate for that lane should read.

These were produced in **float16**, not the checkpoint's native bfloat16: the
reference cards are compute capability 7.5, where torch emulates bf16 rather than
running it. That is an open follow-up — spot-check on an Ampere-or-newer card in
bf16 before treating byte-level differences against this gold as lane bugs.

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
| `cc0/` | 5 | 5 | 0.02902 |

All 25 fixtures have a transcript. Two of them cost real effort to get, and that
shows up in the timings rather than the text: `commons_test_ocr_document.jpg`
and `german_official_print.jpg` are ~4.8 Mpix scans, which the preprocessor
turns into ~6k vision patches. On the reference hardware (2x Tesla T4, compute
capability 7.5) SDPA has no memory-efficient kernel for the vision tower's mask
and falls back to the math path, which materialises the score matrix — one
3.98/4.07 GiB allocation that no weight split on 2x15 GiB can make room for.
Seven placements and three host-memory retries were measured before a
configuration with ~11.5 GiB of weights in host memory read both pages, at
2512 s and 1414 s, for CER 0.00637 and 0.00991.

Those two rows carry `cpu_offloaded` and `timing_comparable: false` in the
results JSON. **Read the cc0 timing column for nothing at all** — the same
constraint also spilled part of the model to disk on the GPU-resident pages.
`synth` is the clean latency measurement. See
`tests/results/ocr_parity_qwen25vl_2026-08-04.json` -> `vram_investigation` for
the per-attempt residency and outcome table.

**TODO:** re-run on a single Ampere-or-newer card with >=24 GB (L4/A100), where
the vision tower is not split across devices and SDPA can use a memory-efficient
kernel. That would give a usable cc0 latency column and settle the bf16 question
below in the same run.

## Regenerating

See `tools/kaggle/qwen-vl-parity/README.md`. The reference does not run on the
dev Mac: 15.45 GiB of 16-bit weights against 16 GiB of total unified memory.
