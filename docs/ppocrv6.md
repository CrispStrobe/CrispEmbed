# PP-OCRv6

CrispEmbed supports the official PP-OCRv6 model family through one GGUF
converter and one runtime family:

| tier | detector | recognizer |
|---|---|---|
| tiny | `PP-OCRv6_tiny_det` | `PP-OCRv6_tiny_rec` |
| small | `PP-OCRv6_small_det` | `PP-OCRv6_small_rec` |
| medium | `PP-OCRv6_medium_det` | `PP-OCRv6_medium_rec` |

The source repositories are the official
`PaddlePaddle/PP-OCRv6_*_safetensors` repositories. Keep source checkpoints,
F16 GGUFs, quantized GGUFs, and parity fixtures on the external model volume:

```text
$CRISPEMBED_GGUF_DIR/
```

Convert one model with:

```bash
python models/convert-ppocrv6-to-gguf.py \
  --model-dir "$CRISPEMBED_GGUF_DIR/source/PP-OCRv6_small_rec_safetensors" \
  --output "$CRISPEMBED_GGUF_DIR/PP-OCRv6_small_rec-f16.gguf"
```

The converter folds inference BatchNorm into convolution weights. PP-OCRv6's
policy-q4 deployment files intentionally retain the complete detector or
recognizer graph in F16: quantizing intermediate CNN/SVTR weights compounds
error through the CTC path and fails parity. The policy therefore prioritizes
quality over file-size reduction for these compact models. In the published
F16 artifacts, the detector/recognizer output head is additionally retained
in F32 because it is the most sensitive part of the DB/CTC decision boundary.

F32 small/medium conversions match the native reference through logits, while
the published F16 artifacts accumulate measurable drift through repeated
layers. A first true Q8 experiment (pointwise CNN/SVTR weights quantized,
sensitive tensors retained) degraded small-rec logit cosine to about 0.59, so
Q8 is not enabled by default for the full graph. The supported compromise is
head-only Q8 from an F32 source:

```bash
build/crispembed-quantize PP-OCRv6_small_rec-f32.gguf \
  PP-OCRv6_small_rec-q8-head.gguf q8_0 --ppocrv6-q8-head
```

This keeps the CNN/SVTR backbone in F32 and quantizes only the final head;
current logits cosine is 0.999987 (small) and 0.999934 (medium). Q4 remains
an explicit debug-only policy variant.

The official v6 preprocessing is also load-bearing: recognizers use a 48-pixel
height, aspect-ratio-preserving width with padding up to 320 pixels, RGB
conversion, rescaling by `1/255`, and the model's declared normalization. Text
detection uses the v6 736-pixel minimum-side policy and ImageNet channel
normalization. These values must remain part of the parity fixtures.

## Backend status and GPU roadmap

**Status as of 2026-08-04.** The graph paths described below are no longer
staged experiments — read this section against `src/ppocrv6_det.cpp` and
`src/ppocrv6_ocr.cpp`, not against the older staged-rollout wording it replaces.

The **detector full graph is the default for tiny, small and medium** since
2026-08-04 (`ppocrv6_det.cpp::graph_build`). It covers the stem, backbone,
neck — including medium's RepLKFPN neck — and head.
`CRISPEMBED_PPOCRV6_DET_SCALAR=1` restores the scalar reference, which is the
bisection lever. The historical `CRISPEMBED_PPOCRV6_DET_GRAPH` /
`CRISPEMBED_PPOCRV6_DET_GRAPH_ACCEPT` opt-in pair no longer exists.

The detector graph deliberately runs on the **CPU backend**: Metal ran the same
graph 9x slower (1693 ms vs 187 ms on `synth_00_clean`, 2026-08-04) because conv
at these spatial sizes does not pay for the dispatch.
`CRISPEMBED_PPOCRV6_DET_GPU_LOAD=1` is the explicit GPU opt-in and
`CRISPEMBED_PPOCRV6_FORCE_CPU` pins CPU.

Promotion evidence (2026-08-04), after the insert-SE double-scale fix that was
the actual cause of the old geometry mismatch: probability cosine ~1e-8 versus
the scalar path, 25-fixture labelled CER net-better at 0.06394 vs 0.06410, and
1.5-1.7x faster (`synth_00_clean` 175 ms vs 316 ms; a 1920x2518 page 1363 ms vs
2056 ms). Medium was validated the same day: every `med_*` tap at cosine
0.99999998-1.0, probability 0.99999999 with equal norms, identical box counts,
detector time 6.9 s -> 1.0 s on `synth_00_clean` and 41.4 s -> 8.7 s on
`german_official_print`, German CER 0.04856 graph vs 0.04955 scalar.

This supersedes the earlier probe that reported detector probability-map cosine
0.99113, head pre-sigmoid 0.99898 and one extra box (31 vs 30) on the German CC0
fixture; that measurement was taken on 2026-08-01, before the double-scale fix,
and is kept here only so the number is not re-derived as a live defect.

Use `CRISPEMBED_PPOCRV6_DET_BENCH=1` to print the detector timing split, and
`CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE=1` to run the scalar reference alongside
the graph. On the German CC0 page (2026-08-01, CPU build) normalization was
about 2.5 ms while the scalar detector work was about 626 ms; preprocessing
stays on CPU unless a backend-specific measurement proves the transfer
worthwhile.

Use `CRISPEMBED_PPOCRV6_BENCH=1` for the routed end-to-end split, including
quad crop geometry, orientation, and recognition. On the German CC0 page the
Metal build measured (2026-08-01, before the detector graph was promoted)
detector 6.9 s, crop 3.4 ms, orientation 358.6 ms, and recognition 455.2 ms;
geometry is therefore not a useful GPU-offload target, while orientation is a
separate graph optimization target. The detector number is superseded: the
medium graph took the same fixture family from 6.9 s to 1.0 s on 2026-08-04.

**The small/medium recognizer graph is the default since 2026-08-02**
(`ppocrv6_ocr.cpp::pp_graph_enabled`). It was promoted on evidence: decoded text
identical to the CPU reference on all 26 fixtures tried (20 synthetic plus 6 CC0
scans, the largest 71 regions), and ~1.9x faster end-to-end on a quiet box
(`synth_00_clean` 1230 -> 646 ms; the 1920x2518 `german_official_print` scan
9369 -> 4964 ms). `CRISPEMBED_PPOCRV6_NO_GRAPH=1` restores the CPU reference
everywhere.

The **fused batch graph is also the default since 2026-08-04**: same-width crops
are grouped (`CRISPEMBED_PPOCRV6_BATCH_MAX`, default 8) and run as one graph.
26/26 fixtures decode byte-identical to the scalar-per-crop path and a 47-crop
receipt runs recognize 3743 -> 2563 ms on Metal.
`CRISPEMBED_PPOCRV6_BATCH_GRAPH=0` (or `CRISPEMBED_PPOCRV6_NO_BATCH_GRAPH`)
disables it; `CRISPEMBED_PPOCRV6_BATCH_GRAPH_CPU_ONLY=1` restores the old
backend gate. The recorded "Metal fourth-dimension pooling" failure that kept
this lane CPU-only was a zeroed in-out width parameter, not a backend
limitation — every batch graph was built at width 0 and asserted on any backend.

Recognizer input width is bucketed up to a multiple of 64 by default since
2026-08-04, so nearby widths share one graph shape and land in the same fused
group (`CRISPEMBED_PPOCRV6_WIDTH_BUCKET=<n>` overrides the step, `0` disables).
Rounding up only adds gray padding, which decodes to CTC blanks. 25-fixture CER
gate: 0.06408 vs 0.06410, jitter in both directions.

What is still gated: the **tiny** variant's single-crop full-logits lane remains
diagnostic-only unless `CRISPEMBED_PPOCRV6_GRAPH_ACCEPT=1` is set, and the CPU
recognizer remains the fallback result for it. The fused batch lane (groups of
two or more equal-width crops) accepts graph logits without that switch.
`CRISPEMBED_PPOCRV6_SVTR_GRAPH=1` and
`CRISPEMBED_PPOCRV6_SVTR_DECODER_GRAPH=1` remain available for bisecting the
SVTR tokenization and attention/MLP seams, and
`CRISPEMBED_PPOCRV6_GRAPH_STOP=backbone` stops the graph at the backbone
boundary. Use `CRISPEMBED_PPOCRV6_GRAPH_BENCH=1` for per-line recognizer graph
latency and the selected backend.

The tiny/small static-shape graph allocation is retained across line crops.
Only the input staging tensor is refreshed per crop; backend buffer planning is
not repeated for every line. This optimization does not alter the accept gate
or the CPU reference fallback. A future dynamic-shape recognizer must clear
the allocation before rebuilding its scheduler graph.

The graph-output gate is stricter than activation-reference parity: on the
`HI` line fixture, tiny accepted-output parity has been validated on CPU and
Metal. Small/medium full-graph gold validation passes Arabic, receipt, and
German fixtures via `tests/test_ppocrv6_graph_gold.py --require`: CPU logits
cosine is `0.999995–0.999996`, Metal is `0.999956–0.999982`, with unchanged
decoded text. Only the tiny single-crop lane still needs the explicit accept
switch; small and medium are accepted by default (see above).
Regenerated references are backed up under
`/Volumes/backups/ai/crispembed-gguf/` and belong in the corresponding
`cstr/PP-OCRv6_*_rec-GGUF` repositories.

Non-CPU detector graphs use F16 resident convolution weights; the CPU graph
stays F32 (`ppocrv6_det.cpp::graph_conv`). Set
`CRISPEMBED_PPOCRV6_DET_F32_WEIGHTS=1` when running a high-precision backend
comparison. On the Apple M1 probe (2026-08-01) this reduced the detector graph
from roughly 3.6 s to 3.3 s without changing the reported parity cosines — note
that the detector now defaults to the CPU backend, so this only applies under
`CRISPEMBED_PPOCRV6_DET_GPU_LOAD=1`.

The PP-LCNet orientation graph is also opt-in with
`PPLCNET_ORIENTATION_GRAPH=1` and `PPLCNET_ORIENTATION_GRAPH_PIPELINE=1`. It uses a backend scheduler with CPU fallback
for ggml operations not implemented by the selected GPU backend. Its output is
diagnostic-only unless `PPLCNET_ORIENTATION_GRAPH_ACCEPT=1` is set; the current
Metal probe executes safely and matches the CPU reference within 0.026 logit
absolute error on the German line fixture. Run
`PPLCNET_ORIENTATION_GRAPH_PARITY=1` with the orientation test to enforce that
gate. The current expanded probe passes 9/10 German/Arabic/derived fixtures;
the uneven-illumination Arabic fixture has a Metal delta of 1.07/3.22 while
the CPU graph passes with 0.0046/0.0139. Tap diagnostics show the Metal drift
starts around SE block 4 and accumulates through the later depthwise/SE blocks;
it is therefore a backend numerical issue rather than preprocessing or graph
topology. Production still requires the explicit accept switch until that
backend case is resolved. The explicit pipeline graph now
passes the full orchestrator smoke safely; on the German CC0 page it takes
about 1.15 s for 30 crops versus 0.36 s for the CPU path because each crop
currently reallocates the mixed scheduler. Without the pipeline switch, the
orchestrator keeps the faster CPU orientation path.

The current CPU parity probe reports detector probability-map cosine 0.99113
and head pre-sigmoid cosine 0.99898 on the German CC0 fixture. The graph still
produces one extra box (31 vs 30), so the explicit accept switch remains a
debug gate rather than a production setting.

Dump the current torch reference fixture and enable native comparisons with:

```bash
python tools/dump_ppocrv6_reference.py \
  --model-dir "$CRISPEMBED_GGUF_DIR/source/PP-OCRv6_tiny_rec_safetensors" \
  --image /path/to/line.png \
  --output "$CRISPEMBED_GGUF_DIR/PP-OCRv6_tiny_rec-ref.gguf"
PPOCRV6_REF="$CRISPEMBED_GGUF_DIR/PP-OCRv6_tiny_rec-ref.gguf" \
  ./build/crispembed -m "$CRISPEMBED_GGUF_DIR/PP-OCRv6_tiny_rec-f16.gguf" \
  --ocr /path/to/line.png
```

The same comparison is available as a standalone regression binary:

```bash
PPOCRV6_REF="$CRISPEMBED_GGUF_DIR/PP-OCRv6_tiny_rec-ref.gguf" \
  ./build/test-ppocrv6-rec \
  "$CRISPEMBED_GGUF_DIR/PP-OCRv6_tiny_rec-f16.gguf" \
  /path/to/line.png
```
