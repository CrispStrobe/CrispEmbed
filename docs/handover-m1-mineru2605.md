# M1 fallback: MinerU2605 provider wrapper

The Linux panel is already dispatched. Check its exact run IDs in
[the execution receipt](../tests/results/ocr-benchmark-expansion/mineru2605-aligned-panel-20261104.json)
before starting work. Do not duplicate those jobs or substitute Apple measurements
into a receipt labelled Linux. This handoff supplies a separate Apple route if
hosted capacity remains unavailable; it does not claim an Apple result.

## Immutable starting point

Use branch `audit/mineru2605-aligned-q4-20261104`, commit
`1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6`. Work in an isolated feature worktree;
fetch and merge current main before edits, preserve the producer commit and record
any resulting executable-source difference. Source model and helper identities
are in `models/mineru25-pro-2605-source.json`; the eight immutable inputs and bars
are in `docs/mineru25-pro-2605-protocol.json`.

The Linux receipt case passes all 208 final-logit positions at all four precisions,
with exact decoded output and helper JSON/Markdown. Actual Q4 mix is 384 F32,
273 Q8_0 and 24 Q4_K tensors: 1,278,646,496 bytes, SHA256
`90d7684fa3350351eae8f37be4a055a7d5763ed9c5a18c307171a22d237232cd`.
Do not broaden the storage parser to admit the initial undeclared Q4_0 fallback.
See [the reviewed receipt](../tests/results/ocr-benchmark-expansion/mineru2605-aligned-review-41871894063.json).

## First stage: bounded build only

Inspect load, available RAM, swapouts and free disk before compiling. Keep build
parallelism at 2 on the 16 GB machine. Configure a separate scratch build with
`CMAKE_BUILD_TYPE=Release`, `GGML_NATIVE=OFF`, `CRISPEMBED_NATIVE=OFF`,
`GGML_METAL=OFF`, `GGML_CUDA=OFF`, `GGML_BLAS=OFF` and `GGML_ACCELERATE=OFF`.
Build `test-mineru25-profile`, `test-mineru25-run`, `test-mineru25-diff`,
`crispembed-quantize` and `crispembed-shared`; run the profile controls first.
Retain the exact source, compiler, CMake cache, binary SHA256, wall time, peak RSS
and swapout delta. No model is needed for this first stage.

## Provider wrapper required before inference

The existing `tools/ci-heavy/mineru2605_qualify.sh` and
`tools/mineru2605_resources.py` are Linux-specific (`lscpu`, `/proc/meminfo`, CPU
wheel index). They cannot run unchanged on macOS. Add a separate Darwin wrapper
and source-reviewed provider contract; preserve the Linux scripts and frozen
producer. Reuse their converter, quantizer, original, native and document-helper
commands without changing model arithmetic, prompts, token caps, repeats or bars.

- Measure genuinely available RAM on Darwin; retain the 12 GiB inference
  headroom requirement. Stop if unavailable. Retain the combined scratch/output
  disk calculation. The frozen planner's roughly24 GiB shared bound assumes one
  temporary GGUF, while its shell retains all four. Use the conservative outer
  reservation in `tools/kaggle/mineru2605-cpu/kernel.py` (about33 GiB combined)
  until a separately reviewed cleanup repair exists, plus checkout/live reserve. Do not count one free
  filesystem twice or treat swap as RAM.
- Use Python 3.12 and the same direct package versions in
  `tools/mineru2605-reference-requirements.txt`, with torch 2.9.0 and torchvision
  0.24.0 from their actual Apple wheels. Record resolver output, full freeze and
  `pip check`; verify original framework sources before any forward. Do not
  spoof Linux, GitHub or Kaggle environment fields to pass a guard.
- Keep all CPU backend flags above off and original/native thread count at 4.
  Record actual architecture, OS and package identities. A changed package or
  original source needs an explicit reviewed protocol, not a version-guard bypass.
- The recipe uses `crispembed-quantize` with the exact Q8 overrides for decoder
  attention, gate and up matrices; down matrices remain genuine Q4_K. Use the
  exact commands in the frozen Linux wrapper. Independently inspect all tensor
  roles and hashes. Standard-header export/readback remains separate from these
  legacy-header hash-pinned measurements.

Start with one still-needed immutable case. Freeze all original arrays and every
151,936-wide generated-position logit before native execution. Then run F32,
F16, Q8_0 and the aligned Q4 mix, preserving each intermediate error/magnitude,
free generation IDs/text/EOS/cap, resident-repeat stability, same-host timing,
peak RSS and helper parity. All original prefixes and positions must be covered;
component cosine, a successful build or a top-k score list cannot qualify an arm.

Accept only the unchanged exclusive final-output bars (.99/.98/.97/.94). Keep
strict .002/.99999 failures and any decoded differences in the report. One Apple
case is not an eight-case panel, a Linux result or publication qualification.
Scope the remaining Apple cases only after that first result and resource review.
Return small JSON/log receipts and source commits; keep weights/raw arrays outside
Git and preserve them remotely if continued analysis requires them. Public files
must contain no machine locations or access details. No alias, default, app pin,
release tag or model upload is part of this fallback.
