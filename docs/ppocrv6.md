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

The default PP-OCRv6 detector, recognizer, and PP-LCNet orientation runtime
remain correctness-first CPU implementations. An experimental persistent GGML
full recognizer graph is available for tiny/small recognizers with
`CRISPEMBED_PPOCRV6_GRAPH=1`; it covers the stem, depthwise/pointwise blocks,
SE gates, activations, residuals, all four backbone stages, pooled head,
folded batch norms, and projection logits. The detector/orientation paths
remain on the CPU reference path until their graph parity gates are complete.

The graph port is staged: first reproduce detector, SVTR recognizer, and
PP-LCNet logits with persistent ggml graphs and CPU parity taps; then enable
Metal/CUDA scheduling with residency checks; finally batch line crops and cache
static-shape graphs and dequantized critical tensors. Every stage must pass
cosine/logit parity and live CER gates on the CC0/German corpus.

Until those gates pass, PP-OCRv6 remains explicitly CPU-first; enabling the
experimental graph does not silently change the default execution path.

The detector has a separate staged graph switch,
`CRISPEMBED_PPOCRV6_DET_GRAPH=1`. For tiny/small models it constructs the
detector stem, backbone, neck, and head graph. Its probability map remains
diagnostic-only until tensor/box parity is complete; setting
`CRISPEMBED_PPOCRV6_DET_GRAPH_ACCEPT=1` explicitly accepts non-empty graph
detections. Without that second switch, the pipeline uses the CPU reference
path, preventing graph geometry or score drift from changing OCR results.

The current CPU parity probe reports detector probability-map cosine 0.99113
and head pre-sigmoid cosine 0.99898 on the German CC0 fixture. The graph still
produces one extra box (31 vs 30), so the explicit accept switch remains a
debug gate rather than a production setting.

Use `CRISPEMBED_PPOCRV6_DET_BENCH=1` to print the detector timing split. On
the German CC0 page, CPU normalization is about 2.5 ms while the scalar
detector work is about 626 ms; the graph diagnostic reports about 590 ms on
this CPU build before its CPU fallback. This keeps preprocessing on CPU unless
a backend-specific measurement proves the transfer worthwhile.

Use `CRISPEMBED_PPOCRV6_GRAPH_BENCH=1` to print per-line recognizer graph
latency and the selected backend. On the Metal build this is a graph execution
measurement, not a claim that the detector graph is production-accepted.

Non-CPU detector graphs use F16 resident convolution weights by default for
backend throughput; set `CRISPEMBED_PPOCRV6_DET_F32_WEIGHTS=1` when running a
high-precision backend comparison. On the Apple M1 probe this reduced the
diagnostic detector graph from roughly 3.6 s to 3.3 s without changing the
reported parity cosines.

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
