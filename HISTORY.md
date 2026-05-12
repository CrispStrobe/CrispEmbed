# CrispEmbed — History

Completed milestones and work log. See PLAN.md for current roadmap.

---

## May 12, 2026 — Face Pipeline Complete

Full detect → align → encode pipeline for face recognition.

### RAG parity: prompt prefixes + new models

- Added auto-prefix system: BGE, E5, Nomic, Jina models get query/passage
  prefixes auto-applied. `crispembed_query_prefix()`/`crispembed_passage_prefix()`
  C API. CLI auto-applies query prefix with `--prefix ""` override.
- Converted 3 new models: SPLADE-PP-en-v1 (cos=1.0), granite-embedding-278m
  (cos=1.0), granite-embedding-107m (cos=1.0). All quantized Q8_0+Q4_K.
- Python wrapper already had sparse/colbert/rerank (Phase 1 complete).
- Model registry: 47 models total (25 encoder + 7 decoder + 2 multimodal + 12 reranker + 1 SPLADE).
- 95+ GGUF files in cache (F32 + Q8_0 + Q4_K variants).

### SCRFD preprocessing + anchor decode fixes (3 bugs)

1. **RGB→BGR channel swap**: SCRFD is trained with OpenCV (BGR). stb_image loads
   RGB. Added R↔B swap in preprocessing.
2. **Anchor center offset**: used `(col+0.5)*stride` but InsightFace uses `col*stride`
   (integer grid positions, no 0.5 offset).
3. **Top-left placement**: InsightFace puts resized image at (0,0) with right/bottom
   padding. We were using centered letterbox with padding on all sides.

After these fixes: landmarks match InsightFace within 1-10px, IoU 0.91-0.98.
Face matching cosine scores match ground truth (0.68 vs 0.68, 0.74 vs 0.74).

### SCRFD anchor decode fix (data layout mismatch)

The anchor decode in `detect()` used interleaved indexing
`(row * grid_w + col) * n_anchors + a` but the data is in ggml's
channel-last layout `col + row * grid_w + a * grid_w * grid_h`.
Similarly, bbox channels were read interleaved but stored channel-last.

After fix: detection counts match InsightFace exactly on all test images
(1/1/4/4/8 faces). Previous false positive count was 2x-4x too high.

Root cause: the ggml graph replayer's Transpose/Reshape are passthrough
(not implemented for 3D+ tensors), so data stays in ggml's native
[W, H, C] layout rather than the ONNX output's interleaved layout.

### Face alignment fix (root cause: 4 sign errors)

`face_align.cpp:estimate_affine()` had 4 sign errors in the normal equations
for the similarity transform. The design matrix rows `[sx, -sy, 1, 0]` and
`[sy, sx, 0, 1]` were incorrectly cross-accumulated:
- `A[0][3]`: `-sy` → `+sy`
- `A[1][2]`: `+sy` → `-sy`
- `A[2][1]`: `+sy` → `-sy`
- `A[3][0]`: `-sy` → `+sy`

After fix: alignment matches InsightFace `norm_crop` with MAE=0.00.
Per-face embedding cos=0.994-0.999 vs InsightFace ArcFace.

### Pipeline implementation

- `cnn_embed::detect_file()` — letterbox resize to 640x640, coordinate scaling back
- `cnn_embed::encode_aligned()` — 5-point landmark similarity transform + encode
- `cnn_embed::face_pipeline()` — detect → align → encode in one call
- `cnn_embed::encode_face_file()` — encode face from file path + landmarks

### CLI

- `--face-pipeline` mode with `--det` detection model
- `--conf` configurable confidence threshold
- Cross-image face matching with cosine similarity

### C API (crispembed.h)

- `crispembed_face_context` opaque handle
- `crispembed_face_init/free/dim/type`
- `crispembed_detect_faces()` — letterbox + coordinate scaling
- `crispembed_encode_face()` — aligned face encoding
- `crispembed_face_pipeline()` — full pipeline

### Server API (crispembed-server)

- `POST /detect` — face detection from image path
- `POST /face` — full pipeline (detect + align + encode)
- `--det` and `--rec` arguments for face model loading
- Server can run face-only (no `-m` text model needed)

### Wrappers

- **Python**: `CrispFace`, `CrispFacePipeline` classes
- **Rust**: `CrispFace`, `CrispFacePipeline` structs with safe wrappers
- **Dart/Flutter**: `CrispFace`, `CrispFacePipeline` classes with FFI

### Models converted

- SCRFD-10GF (det_10g.onnx → scrfd_10g.gguf, 16.1 MB)
- w600k_r50 ArcFace (buffalo_l → w600k_r50.gguf, 166 MB)
- AuraFace-v1 (248.6 MB)
- SFace (36.8 MB)

---

## May 11-12, 2026 — Vision Models & Parity Fixes

### SigLIP image embedding
- Converter: `models/convert-siglip-to-gguf.py`
- Forward path: `src/vit_embed.cpp` — cos=0.996 vs HF mean-pool
- Native `--image` flag with stb_image preprocessing
- Uploaded: cstr/siglip-base-GGUF

### Face detection (SCRFD)
- Generic ONNX graph replayer in `src/cnn_embed.cpp`
- FPN backbone + multi-scale detection heads
- Anchor decode + NMS at strides 8/16/32
- Semicolon delimiter for ONNX tensor names with commas

### Face recognition (SFace + AuraFace)
- SFace MobileFaceNet: cos=0.9999 vs ONNX, 128-D
- AuraFace ResNet-100: cos=0.9999 vs ONNX, 512-D
- BN folding/precomputation at converter time
- PReLU: relu(x) + slope * (x - relu(x))
- Conv F32→F16 auto-cast for ggml_conv_2d

### Text model parity fixes (35 models)
- GTE v1.5: post-LN + GeGLU half swap + NTK RoPE
- Jina reranker v2: post-LN + position offset
- NomicBERT: SwiGLU fc11/fc12 swap
- Ollama format: auto-strip ▁ prefix, dual metadata keys, pooling type mapping

---

## Apr 12, 2026 — v0.1.0 Release

30-commit session: FastConformer extraction, granite 3.x support,
NeMo FC-CTC, omniASR, Silero LID, CI, Windows, Vulkan, benchmarks.
Tagged v0.1.0 release with multi-platform binaries.
