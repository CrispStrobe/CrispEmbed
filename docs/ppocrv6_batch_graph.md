# PP-OCRv6 batching investigation

Status 2026-08-04: **the fused lane runs on CPU AND Metal with scalar-identical
output** — the historical blocker was never a Metal limitation. The fused
caller passed a zeroed value into `pp_graph_run_batch`'s in-out width
parameter, so every batch graph was built at width 0; the stem shapes
collapsed until `ggml_pool_2d` asserted `ne[0] > 0` on *any* backend, and the
failure was mis-filed as a "Metal fourth-dimension pooling" issue. With the
group width seeded: CPU two-crop contract passes byte-identical
(`parity=PASS`), and the Metal probe runs `width=320 batch=3 action=fused`
with all items scalar-identical at **54.9 ms vs 118.6 ms scalar (2.2x)**.
The lane is the DEFAULT since the large-stem extension and 26/26 sweep (non-CPU
backends can be re-gated with `CRISPEMBED_PPOCRV6_BATCH_GRAPH_CPU_ONLY`);
promotion needs the large-stem (small/medium) extension plus a multi-fixture
sweep. Historical investigation below, kept for the record — its Metal
conclusions are superseded.

## What was implemented

- `ppocrv6_ocr_recognize_raw_batch()` accepts multiple caller-owned RGB/gray crops,
  preserves caller order, copies each result into caller-owned buffers, and isolates
  invalid/empty items.
- The orchestrator now batches the detector crop/orientation handoff through that C
  API. It groups by the dynamic recognizer width while retaining original page order.
- `CRISPEMBED_PPOCRV6_BATCH_MAX` bounds group size; the default is 4.
- The tiny logits graph has an experimental fourth-dimension batch shape behind
  `CRISPEMBED_PPOCRV6_BATCH_GRAPH`.
- Metal is explicitly excluded from that experimental fused path. When the flag is
  requested on Metal, benchmark mode emits a scalar-fallback record and the normal
  grouped scalar route runs.

## Evidence

- The two-crop scalar/batch contract passes with byte-identical outputs and both
  items completed. The first reported CPU timing (`52.5 ms` scalar / `35.1 ms`
  batch) was not a fused-graph measurement: `CRISPEMBED_PPOCRV6_FORCE_CPU` disables
  graphs, so it measured grouped scalar execution only.
- A German CC0 full-page live route completed `33/33` regions and `1,146` chars in
  `68.75 s`; recognition was `46.45 s`. The page had 22 distinct dynamic widths,
  so same-width grouping alone cannot create a large page batch.
- A real Metal fused probe reached GGML's fourth-dimension pooling path and aborted
  at `GGML_ASSERT(ne[0] > 0)` inside `ggml_pool_2d`. This is why the Metal gate is
  conservative and why no GPU speed claim is made.
- The graph debug taps showed the scalar tiny reference shapes as backbone `3x20`,
  pooled `1x10`, and head outputs `1x10`. The experimental batch graph currently
  flattens/loses the per-item spatial dimension around pooling and CTC reshaping;
  the output shape cannot yet be treated as independent per-item logits.

## Correct test modes

Normal production/reference CPU route:

```sh
CRISPEMBED_PPOCRV6_FORCE_CPU=1 build-metal/test-ppocrv6-rec MODEL IMAGE IMAGE
```

Experimental CPU graph probe (requires a rebuilt test binary):

```sh
CRISPEMBED_PPOCRV6_FORCE_CPU=1 \
CRISPEMBED_PPOCRV6_GRAPH=1 \
CRISPEMBED_PPOCRV6_GRAPH_ACCEPT=1 \
CRISPEMBED_PPOCRV6_BATCH_GRAPH=1 \
build-metal/test-ppocrv6-rec MODEL IMAGE IMAGE
```

Do not use `CRISPEMBED_PPOCRV6_BATCH_GRAPH=1` as a Metal promotion switch. It is a
debug probe, and the runtime gate keeps Metal on the safe scalar path.

## Remaining work

1. Build an explicit per-item branch/sequence representation that survives
   `ggml_pool_2d`, `ggml_permute`, and CTC flattening instead of relying on a
   fourth dimension being implicitly preserved.
2. Add two-crop gold logits and decoded-text comparisons for each item, with a
   cosine threshold before accepting fused output.
3. Re-test CPU fused execution with a graph-enabled CPU probe; the earlier CPU
   timing must not be reused as fused evidence.
4. Only after CPU parity passes, test Metal. Extend to small/medium large-stem
   SVTR only after the tiny logits lane is stable.
