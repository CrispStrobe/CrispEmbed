# LightOn4B bounded output projection component

The pinned 4B tied head has 248,320 rows and width 2,560 (635,699,200
weights). The existing `project_cpu()` rejects more than 536,870,912 total
materialized/input/output elements before allocation. Raising that limit would
materialize over 2.5 GB of F32 head weights plus the raw source copy.

`project_rows_cpu()` in `src/qwen35_linear.cpp` adds a separate component behind
both `CRISPEMBED_QWEN35_OCR=1` and default-off
`CRISPEMBED_QWEN35_STREAM_PROJECTION=1`. It reads contiguous F32/F16 source rows
through the ggml backend API, converts one tile, and uses the existing F32
projection arithmetic. Complete logits are returned in token-major order;
there is no top-k approximation. The original projector, caches and decoder
call sites remain unchanged. No format-5/model admission is added.

Bounds are width 32,768, vocabulary 300,000, 16 token positions, 1–1,024 tile
rows, and at most 1,048,576 weights per tile. For width 2,560 the maximum tile
is 409 rows; callers can use 256. Output storage is at most 4,800,000 floats,
source tile storage at most 4 MiB, and converted tile storage at most 4 MiB,
plus the separately bounded existing projector's graph/materialization. These
are allocation bounds, not a measured RSS claim. Each tile repeats graph setup
and input copy; no speed improvement is claimed. Quantized source tensors are
accepted only for Q8_0 or Q4_K with `CRISPEMBED_QWEN35_QUANT=1` and a
block-aligned row width. Each quantized tile is dequantized and projected with
the same bounded F32 path. This support is a source change awaiting hosted
controls and actual model logits; it is not a quantized quality result.

Calibration observes the original named matrix once with the original hidden
input; internal tiles suppress callbacks. Output is committed only after every
tile succeeds, so a late nonfinite weight exposes no partial logits.

The model-free `test-qwen35-stream-projection` target is registered in ordinary
hosted Build controls. It checks F32/F16 against independent double dot products
and the original projector at four tile sizes, partial final tiles, token-major
layout, unset/empty/zero and OCR gates, late nonfinite failure, and observer
restoration. No builds or suites were run on the shared host. `git diff --check`
and environment-inventory generation are the only local verification so far.
[Hosted Build41924115399](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41924115399)
passes at exact runtime source `45043bcfe`: static guards, all seven platform
builds, sibling shared-library checks and Flutter bundle. Linux actually runs the
new control and reports `STREAM_PROJECTION_CONTROLS pass=1` for F32/F16, seams,
independent dots, legacy agreement, gates, atomic failure and observer handling.
Earlier41923876329 is preserved separately; superseded41923837804 was cancelled
before this final source. An unallocated-backend guard, callback exception
containment and32-bit source-span bound are included in the final source.
Actual full-size checkpoint/backend readback,
complete original/native logits and decoded/EOS quality remain required.
Historical frozen source hashes must retain their original closure; this runtime
delta requires a separately reviewed source closure before model qualification.

This is an unreleased experimental addition to the existing development
0.17.15 candidate. It does not create a release, publish assets, change an app
pin, or qualify the larger model.
