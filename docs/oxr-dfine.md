# OXR-1.0 D-FINE layout detector: native GGML

OXR's document pipeline needs a page-layout detector. Upstream ships it as a
separate PyTorch HTTP service (`services/layout_analyze`, a D-FINE model with
weights in `layout_analysis/last.pth`). CrispEmbed now runs that detector
natively as the **`dfine` architecture of `layout_detect`**, next to the
RT-DETRv2 heron model, so neither the Python service nor PyTorch is needed.

**Status: experimental.** Parity evidence is in
[`tests/results/oxr-dfine/`](../tests/results/oxr-dfine/) (see
[Qualification](#qualification)). The model is gated by `CRISPEMBED_OXR=1` and
`CRISPEMBED_ACCEPT_OXR_LICENSE=1`. Nothing is published, aliased or enabled by
default.

## Sources, lineage and license

| Item | Pin |
| --- | --- |
| Checkpoint | `SoMarkAI/OXR-1.0` @ `c708407f9d2b4df658f2543d9132b62bb7221df4`, `layout_analysis/last.pth` (78,791,307 bytes, sha256 `dff78ccf…dc17`), `layout_analysis/metadata.json` (718 bytes, sha256 `658be516…c756`) |
| Blueprint | `SoMarkAI/OXR` @ `c1a1dcb312ee9f691404f127785510453c0d9350`, `services/layout_analyze/dfine_la` |
| Upstream architecture | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) (Apache-2.0), "Stage-2 B2" = D-FINE-M layout (HGNetv2-B2) |

The checkpoint is part of OXR-1.0 and is distributed under the **OXR License**
(Apache-2.0 with additional terms; [`licenses/oxr-LICENSE`](../licenses/oxr-LICENSE)).
The service code retains D-FINE's Apache-2.0 notice. Every surface that loads
the GGUF refuses without both gates, exactly like `src/oxr_ocr.cpp`.

Weight lineage is measured, not assumed: `tools/oxr_dfine_checks.py lineage`
compares every checkpoint tensor with the public HGNetv2-B2 stage-1 and
D-FINE-M Objects365/COCO checkpoints. The result is in
`tests/results/oxr-dfine/lineage.json` and summarized under
[Qualification](#qualification). The training config freezes the whole
backbone (`freeze_at=3`, `freeze_stem_only=False`), so an exact backbone match
would identify its initialization.

## Architecture (as converted and run)

| Stage | Blueprint module | Native |
| --- | --- | --- |
| Preprocess | `preprocess.py` | RGB uint8 → OpenCV `INTER_LANCZOS4` resize to fit 960 (Python `int()` of `w * 960 / max(w, h)`), centered on a 960×960 canvas filled with 114, CHW `/ 255`. No mean/std. `src/core/opencv_lanczos4_u8.h` is a bit-exact port of OpenCV's generic Lanczos4 kernel |
| Backbone | `HGNetv2('B2', use_lab=True)` | Stem (3×3 s2, 2×2 pair with right/bottom padding, 2×2 max pool, concat), four HG stages (stages 3 and 4 light blocks with 5×5 depthwise convs), squeeze/excitation aggregation, LAB affine; FrozenBatchNorm kept as scale/shift |
| Encoder | `HybridEncoder` | 1×1 input projections + BN, one AIFI layer (8 heads, GELU-erf, 2-D sin-cos positions) on stride 32, CCFM top-down/bottom-up with `RepNCSPELAN4` (CSP + RepVGG blocks, fused as in `convert_to_deploy`) and `SCDown` |
| Query selection | `DFINETransformer._get_decoder_input` | valid-mask, `enc_output` + LayerNorm, class logits, top-300 tokens by max logit (ties by index), `enc_bbox_head` + anchors |
| Decoder | `TransformerDecoder` (4 layers) | query position MLP (clamped ±10), self-attention, multi-scale deformable attention with 3/6/3 points and no value/output projection, gated fusion, FFN; FDR: per-layer distribution head over 33 bins × 4 sides, integral with the deploy weighting function, `distance2bbox` against the layer-0 refined boxes; LQE on the last layer |
| Postprocess | `DFINEPostProcessor(letterbox, clip)` | sigmoid, top-300 over (query, class), cxcywh → xyxy × 960, clip to [0, 959], letterbox inverse in float32 exactly as torch (`960 / max` is `reciprocal * 960`) |

Shared with heron inside `src/layout_detect.cpp`: the AIFI layer, the CCFM
topology, the decoder self-attention and FFN blocks and the deformable sampling
core (`aifi_encoder_layer`, `ccfm_forward`, `decoder_self_attn`, `decoder_ffn`,
`deformable_sample`). Heron's op sequence is unchanged (its sampling keeps its
original coordinate arithmetic; D-FINE uses torch `grid_sample` arithmetic);
byte identity is checked on hosted CI against the base branch.

D-FINE runs on the CPU backend. `CRISPEMBED_OXR_LAYOUT_GPU=1` opts into the
GPU backend; that path is unqualified. `CRISPEMBED_OXR_LAYOUT_TRACE=1` prints
per-phase timings.

## Convert

```bash
python models/convert-oxr-dfine-to-gguf.py --model SoMarkAI/OXR-1.0 --dtype f32 --output oxr-dfine-f32.gguf
python models/convert-oxr-dfine-to-gguf.py --model SoMarkAI/OXR-1.0 --dtype f16 --output oxr-dfine-f16.gguf
```

The converter downloads the pinned files, checks size and SHA-256 of both the
checkpoint and `metadata.json` (and the metadata's own contract) **before**
deserializing, loads with `torch.load(weights_only=True)`, and accounts for every
checkpoint tensor (19 deploy-discarded training tensors are listed in the
receipt). 533 tensors; every name ≤ 63 bytes (`tests/test_oxr_dfine_conversion.py`).
GGUF metadata: `layout.arch=dfine`, `layout.license_gate=oxr`, source pins,
labels, hyperparameters. F16 stores conv/linear matrices in F16 and keeps
vectors, norms, tables and the LQE head in F32.

## Use

Anywhere a layout GGUF is accepted (`layout_detect::load`,
`crispembed_layout_init`, the orchestrator's `layout_model`), an OXR D-FINE
GGUF now loads by its metadata. `detect_file`/`detect_rgb` then apply OXR's
own client-side selection (`parse_layout_batch`: score ≥ 0.5 by default, clip,
class-wise NMS IoU 0.6) in service order.

`layout_detect::region` keeps `label`/`label_name` (17 canonical classes) and
adds `raw_class`, `raw_label` and `canonical_exact`:

| OXR id | OXR label | canonical | exact |
| ---: | --- | --- | --- |
| 0 | Caption | caption | yes |
| 1 | Footnote | footnote | yes |
| 2 | Formula | formula | yes |
| 3 | Text | text | yes |
| 4 | Page-footer | page_footer | yes |
| 5 | Page-header | page_header | yes |
| 6 | Picture | picture | yes |
| 7 | Title | title | yes |
| 8 | Table | table | yes |
| 9 | Code Block | code | yes |
| 10 | Stamp | picture | **no** |
| 11 | Chemical Structure | picture | **no** |

`crispembed_layout_region` (the older C struct) is unchanged for ABI reasons
and carries only the canonical label.

### Service-format C API

```c
crispembed_oxr_layout_context * crispembed_oxr_layout_init(const char * gguf, int n_threads);
const char * crispembed_oxr_layout_detect(crispembed_oxr_layout_context *, const uint8_t * pixels,
                                          int width, int height, int channels, float confidence);
const char * crispembed_oxr_layout_timings(crispembed_oxr_layout_context *);
const char * crispembed_oxr_layout_last_error(const crispembed_oxr_layout_context *);
void crispembed_oxr_layout_free(crispembed_oxr_layout_context *);
```

`detect` returns the `/inference` body for a one-page batch, as Starlette
serializes it (compact separators, CPython float repr of the float32 values):

```json
{"status":"success","params":{},"data":{"labels":[[8,7,...]],"boxes":[[[22.3,24.1,179.0,75.2],...]],"scores":[[0.964,...]]}}
```

All 300 candidates are returned, in service order. A finite `confidence` in
[0, 1] reproduces the service form field (keep `score >= confidence`;
`params` = `{"confidence":c,"class_confidences":{}}`); NaN or negative means
none. Per-class overrides are not implemented. Pixels are gray/RGB/RGBA
(alpha discarded, as `cv2.IMREAD_COLOR`).

Wrappers: Python `crispembed.CrispOXRLayout` (`detect`, `analyze` for a
batch, `regions` for OXR-selected region dicts), `CrispOXR(..., layout_model=)`
for the full document pipeline without a layout service; Rust `OxrLayout`;
Dart `CrispOXRLayout`.

## Qualification

Hosted CI lane: `tools/ci-heavy/oxr_dfine_qualify.sh`. First complete run:
[41250159815](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41250159815)
(Linux x86-64, AMD EPYC 9V74, 4 threads, torch 2.11.0 CPU, OpenCV 4.11.0.86),
receipts in [`tests/results/oxr-dfine/hosted-41250159815/`](../tests/results/oxr-dfine/hosted-41250159815/).
Budgets fixed before the run (`tests/oxr_dfine_diff.cpp`): every stage
`max_abs / max|ref| <= 5e-3`; OXR-selected regions (score >= 0.5, clip,
class-wise NMS 0.6) with identical labels in order, boxes within 0.5 px and
scores within 1e-3. Five pages, JPEGs decoded once with PIL to PNG for both sides.

| Page | OXR regions native/ref | max box err (px) | max score err | earliest stage over 5e-3 (rel) | candidates ≥0.05 matched (IoU ≥ 0.99) |
| --- | ---: | ---: | ---: | --- | --- |
| receipt_example | 21 / 21 | 3.1e-4 | 1.8e-4 | dec.2.cross (worst dec.3.cross 0.065) | 219/220, 0.54 px |
| german_official_print | 17 / 17 | 4.4e-4 | 1.5e-5 | dec.2.cross (worst dec.3.gate 0.023) | 90/90, 0.42 px |
| scan_page_pd | 9 / 9 | 6.1e-5 | 3.6e-7 | none (worst dec.logits 0.0022) | 58/58, 0.012 px |
| simple_table | 2 / 2 | 1.9e-5 | 4.2e-6 | none (worst dec.2.cross 0.0018) | 7/7, 2e-4 px |
| japanese_print | 1 / 1 | 1.7e-4 | 7.1e-6 | none (worst dec.2.gate 0.0048) | 58/58, 0.074 px |

* **Preprocessing is exact**: the native 960×960 input equals the service's
  input bit for bit on all pages (max_abs 0), on x86 with the shipped OpenCV
  (IPP enabled; its Lanczos4 output equalled the generic kernel on all five
  pages) and on macOS arm64.
* **Backbone and encoder**: all stages within rel 7e-5; the top-300 query
  selection picks the same token set on every page (near-tied scores may be
  ordered differently, so per-query stages are compared after aligning queries
  by token).
* **Decoder**: errors grow in layers 2–3 for a few low-score queries (4 of 300
  on receipt) whose distribution-head logits (|x| ~ 200) are near ties. The stage
  budget fails there on two pages; this failure is retained. A float64 run of
  the same original model is the yardstick: the native result is closer to
  float64 than PyTorch FP32 is on 56/56 stages for four pages and 55/56 for
  german (`yardstick-f64.json`). The remaining stage difference is therefore
  PyTorch-FP32 rounding, not a port error; it does not reach the selected regions.
* PyTorch with 1 vs 4 threads changes only `post.*` ordering (near ties).
* The service JSON contract passes on all pages (`api.json`): compact
  Starlette serialization and float repr reproduce byte-for-byte when
  re-serialized, keys/counts/types match, `confidence` handling matches, the
  Python wrapper returns the same object, and OXR's own `parse_layout_batch`
  yields equivalent blocks for native and reference responses.
* Heron (`rt_detr_v2`) regions are byte-identical to the base branch
  (`9b718fd7`) on the five pages at threshold 0 (300 regions each, sha256
  `252f60b1…`), so the shared-helper refactor did not change heron.
* **F16 is rejected**: on receipt one selected box differs by 147 px; stage
  budgets fail from the encoder (`enc.pan1`).

### Speed (same host, 4 threads, CPU, FP32)

| | median per page |
| --- | ---: |
| PyTorch service model forward (`DfineWithPostProcessor`) | 0.86–0.92 s |
| PyTorch preprocessing (decode + Lanczos + tensor) | 6–93 ms |
| Native total (preprocess + network + postprocess) | 2.30–2.33 s |
| — native encoder graph (backbone + hybrid encoder + query head) | 2.22–2.26 s |
| — native decoder (4 layers, CPU deformable sampling) | 53–56 ms |

The native detector is **about 2.6× slower than PyTorch CPU**. The backbone/encoder
graph dominates; the decoder is fast. Opt-in A/B arms
(`CRISPEMBED_OXR_LAYOUT_DW_DIRECT=1`, `LAYOUT_CONV_DIRECT=1`) are measured in the
next lane run; the defaults stay unchanged until an A/B shows equal regions and
better speed.

### Weight lineage

No checkpoint tensor is bit-identical to the public HGNetv2-B2 stage-1 weights or
to D-FINE-M COCO/Objects365 (`lineage.json`). Mean cosine similarity per group is
highest for `dfine_m_obj365.pth` (backbone 0.952, encoder 0.932, decoder 0.939;
6 decoder buffers bit-equal), then COCO (0.888 / 0.483 / 0.260) and the stage-1
backbone (0.876). This is consistent with fine-tuning from the public D-FINE-M
Objects365 checkpoint (Apache-2.0) with the backbone not actually frozen, but it
does not prove it; the checkpoint's own terms are the OXR License either way. The
`dfine_m_obj2coco` URL returned HTTP 404 and is recorded as such.

## Not done / open

* Speed parity (native 2.6× slower than PyTorch CPU).
* F16 or quantized recipes (F16 rejected).
* GPU backends (`CRISPEMBED_OXR_LAYOUT_GPU=1` is unqualified).
* Per-class `class_confidences` overrides of the service.
* End-to-end OXR document output with `CrispOXR(layout_model=)` against the
  service-backed pipeline (the layout boundary itself is verified above).
* Rust is type-checked (`cargo check`); Rust and Dart wrappers are not runtime-tested.
