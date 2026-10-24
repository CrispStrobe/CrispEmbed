# M1 OvisOCR2 published-alias consumer audit

This is the operator-authorized 16GB M1 fallback while Linux CI is queued.
Use immutable source `d4c30424af99f1e9970c43449011b290e3721ac6`.
It contains the real `ovisocr2-f16` alias and fixes Darwin child-RSS reporting;
native code, ggml and CMake are unchanged from the qualified F16 source.
Linux [40080458492](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40080458492)
continues independently at5296e955. Do not cancel it or submit another Linux run.

The [qualified public model](https://huggingface.co/cstr/OvisOCR2-GGUF/tree/b7229e46f31bcf342d0999966bae8a03bf02bb07)
is1717222848 bytes, SHA-256
`d3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db`.
All three smaller quantized attempts are rejected; use only this F16 artifact.
No publication/write credential or original framework/checkpoint is needed.

## Resource gate before build or model access

Inspect `uptime`, `memory_pressure`, `vm_stat`, `sysctl vm.swapusage hw.memsize`
and `df -h .`. Proceed only with green memory pressure, roughly6GiB available/
reclaimable RAM, at least10GiB free disk and background load allowing two build/
inference threads. Record initial swapout counters. Keep `-j2`/two inference
threads; stop if memory pressure or new swapouts make the run unsuitable.
Build and inference are sequential, with one loaded model process at a time.
Two separate task caches download about3.44GB total; keep them outside Git.

## Exact isolated source and execution

From an existing CrispEmbed clone, preserve unrelated work and create a detached
worktree. These task-specific temporary directories contain no owner credentials:

```sh
git fetch origin d4c30424af99f1e9970c43449011b290e3721ac6
M1_OVIS_WORK="$(mktemp -d)"
M1_OVIS_OUTPUT="$(mktemp -d)"
git worktree add --detach "$M1_OVIS_WORK" d4c30424af99f1e9970c43449011b290e3721ac6
cd "$M1_OVIS_WORK"
git submodule update --init --recursive
export GITHUB_SHA="$(git rev-parse HEAD)"
export HEAVY_OUT="$M1_OVIS_OUTPUT/reports"
export HEAVY_SCRATCH="$M1_OVIS_OUTPUT/scratch"
mkdir -p "$HEAVY_OUT" "$HEAVY_SCRATCH"
python3 tests/test_ovisocr2_published_consumer.py -v
python3 tools/ci-heavy/ovisocr2_published_consumer.py --preflight --precision f16
python3 tools/ci-heavy/ovisocr2_published_consumer.py --precision f16
```

Use Python3.12 or newer, CMake and an Apple C++17 toolchain. The helper imports
only the standard library; do not install Torch, Transformers, ONNX or NumPy.
It builds only CLI/server targets with the explicit portable CPU configuration,
verifies the25 native source hashes against full quality, retrieves the immutable
small provenance/PNG fixture, and exercises real model downloads/inference.
The model and PNG URLs/hashes are already in the reviewed receipts.

The helper requires genuinely absent CLI/HTTP caches. Preserve a failed output;
for any retry use a new output/scratch directory, not a reused cache presented as
fresh. Do not modify source, inject outputs or relax comparisons to obtain a pass.

## Acceptance and return evidence

Require status `completed published consumer alias audit`, both portable targets,
two fresh-cache full GGUF SHA/size checks, online and offline CLI results plus
HTTP result, exact `Hello OCR\n\n12345`/one region, and unchanged confidence
budgets0.0006 CLI/0.00006 HTTP against the selected F16 mean0.995388269.
The helper sets the three entry gates and audited `both` vision preset itself,
uses two threads and a12-token cap. Public APIs expose text/confidence; the nine
IDs/EOS proof is separately frozen, so do not claim new observed M1 IDs/stops.

Return only the small report JSON/logs, SHA manifest, source/toolchain/CPU/thread
record, memory-pressure snapshots and swapout delta. RSS fields are normalized
to KiB and remain cumulative child high-water upper bounds, not fresh per-call
peaks. Keep full original logs and actual local paths in ignored private notes;
audit public review/Markdown for private paths before committing it. Do not pack
model caches, build trees or credentials into the returned archive or push main.

This validates an Apple alias/API case, not full Apple numerical/eleven-case or
maximum-resolution OCR quality. Linux frozen references and all raw strict flags
remain unchanged. A failure must retain actual output, magnitudes/logs and the
first failing stage; it is evidence, not permission to weaken a budget.

The earlier build-only handoff remains available at immutable
[8b13e839](https://github.com/CrispStrobe/CrispEmbed/blob/8b13e839/docs/handover-m1-ovisocr2.md).
The app remains on immutable0.17.12; this source audit creates no runtime release.
