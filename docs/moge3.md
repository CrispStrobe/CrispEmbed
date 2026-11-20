# MoGe-3 native geometry runtime

## Decision and status — 20 November 2026

Keep **MoGe-3 ViT-L** as an active candidate for a native CrispEmbed geometry
runtime used by CrispMesh. Target ggml's CPU and Metal backends, with a small
C interface and Rust integration following the standalone SAM 2.1 library
pattern. Evaluate complete, refined geometry on the existing reconstruction
inputs before committing to performance tuning.

**Status: design and implementation plan.** CrispEmbed does not yet implement
MoGe-3. The milestones below are pending; this document does not establish
native inference, Apple numerical parity, runtime performance or improved
meshes. The first target is FP32 inference using existing trained weights.
Reproducing inference does not itself require retraining.

The main work is the dense forward graph, sparse 3D refinement, numerical
agreement and camera conversion. Packaging a checkpoint as GGUF supplies
weights and metadata; it does not supply those operations.

## Why Apple execution is a reasonable target

The [official README][moge-readme] attributes its macOS limitation to FlexGEMM's
Triton dependency and the absence of macOS Triton wheels. The [sparse
U-Net][moge-sparse] imports FlexGEMM neighbor-cache and sparse building blocks.
Changing a PyTorch device string to `mps` does not replace that implementation.

PyTorch documents [`PYTORCH_ENABLE_MPS_FALLBACK`][pytorch-mps] as a fallback for
unsupported MPS operations. Given the external FlexGEMM dependency, our
conclusion is that this flag alone cannot install or replace its Triton
kernels. A native CrispEmbed port would use ggml's Metal backend directly.

| Reference implementation | Documented capability | Use and evidence limit |
| --- | --- | --- |
| [LibreYOLO MoGe-3][libreyolo-docs] | Plain-PyTorch sparse refiner, CPU/CUDA/MPS, FP32, three refinement steps | Quick complete-model evaluation on Apple hardware. Published upstream comparisons are on CUDA; they do not establish MPS parity. Use the depth task: normal-only export is not a complete refined-depth evaluation. |
| [MLXcel MoGe3][mlxcel-docs] | Rust/MLX dense decoder, sparse 3D U-Net, F32 checkpoints and configurable refinement | Independent implementation and small operation fixtures. Its [technical report][mlxcel-report] validates ROCm/CPU and explicitly leaves Metal/CUDA execution unvalidated for that report. |
| [mtlgemm][mtlgemm] | Metal sparse-convolution work derived from FlexGEMM | Kernel implementation material to investigate after the CPU reference exists. It is not evidence of a complete MoGe-3 backend or a drop-in ggml integration. |

These are implementation references, not new production dependencies. No port
was executed as part of this planning change.

## Source baseline

The design review used the following source revisions. They identify the
reviewed code, not a frozen benchmark or an independently validated model.

| Source | Reviewed revision | Relevant material |
| --- | --- | --- |
| Microsoft MoGe | [`74fbce054ebed49800de42d0ad0e83495065719a`][moge-root] | v3 forward/infer, inherited v2 loading, DINOv2 encoder, dense decoder, sparse U-Net, camera recovery |
| LibreYOLO | [`359fc39f9189086c395753df47017a75187b1e48`][libreyolo-root] | Plain-PyTorch sparse semantics and checkpoint provenance |
| MLXcel | [`e427b7a02aba041fdbaa847257ccc01fb6100fb0`][mlxcel-root] | `src/vision/dense/moge3/`, geometry/sparse fixtures and validation limits |
| CrispEmbed | [`06af98b175efe4eb15f12ebce638421972c03a78`][crispembed-base] | Existing SAM standalone library, GGUF loader, bindings and comparison utilities |
| CrispMesh | [`64d7d8f55d1551974e60cae97dd82ea865dc3ea8`][crispmesh-base] | Dynamic SAM C-ABI loading, camera conventions, depth reframing and native replay |
| CrispEmbed's ggml submodule | [`3a27b6c665b191cb230be367db5c52d81dd3e2ac`][ggml-base] | Check actual operation support, layouts, types and backend placement at implementation time |

Start with [Ruicheng/moge-3-vitl][moge-weights]. LibreYOLO's [loader][libreyolo-model]
records checkpoint revision `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be` and
`model.pt` SHA-256
`9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`.
These are reported provenance values; this review did not download or rehash
the checkpoint. Verify the actual bytes before freezing a reference capture.

The checkpoint's embedded `model_config` is authoritative. The [training
configuration][moge-config] is useful for understanding the model but must not
silently override checkpoint metadata.

## Proposed runtime division

| Component | Initial implementation | Main acceptance concern |
| --- | --- | --- |
| Image preprocessing, encoder and dense decoder | ggml CPU graph, then the same supported operations on Metal | Dynamic rectangular grids, weights/layouts, resize semantics and intermediate activations |
| Voxel coordinates and neighbor maps | Native CPU indexing with FP32 geometry | Exact integer coordinates, ordering, occupied-site semantics and overflow handling |
| Sparse 3D refinement | Clear CPU reference; bounded gather/matmul/accumulation graph; specialized Metal kernels after profiling | Every sparse block, occupied-only pooling, saved-coordinate upsampling and geometry update |
| Camera recovery and output conversion | Native CPU implementation | Focal/shift solve, scale, masks, normalized intrinsics and pixel conventions |
| Application interface | Proposed standalone `crispembed-moge3` library plus C/Rust surfaces | Ownership, errors, backend reporting and CrispMesh depth replay |

Use [SAM 2.1](sam2.md) as the packaging precedent. Current
[`CMakeLists.txt`](../CMakeLists.txt) builds `crispembed-sam2` from
[`sam2.cpp`](../src/sam2.cpp) and [`sam2_capi.cpp`](../src/sam2_capi.cpp), with an
optional shared library. This permits a small geometry library without pulling
every CrispEmbed engine into CrispMesh. The MoGe target and symbols above are
proposed names, not existing build targets or an available API.

Reuse the typed metadata/weight ownership in
[`src/core/gguf_loader.h`](../src/core/gguf_loader.h) and graph/backend patterns
from SAM and [`src/inpaint_graph.h`](../src/inpaint_graph.h). Check each actual
ggml shape/type/backend combination. API-level availability of an operation
does not prove that the required Metal variant works.

### Dense stage and GGUF conversion

The reviewed ViT-L configuration uses DINOv2-L/14 with intermediate layers
`[5, 11, 17, 23]`. Four projected feature maps are summed and joined with two
view-plane channels. The dense neck widths are `[1024, 256, 128, 64, 32]`, with
point, mask, normal and metric-scale heads. The token range is `[1200, 3600]`.
The sparse refiner uses widths `[32, 64, 128, 256, 512]` and four factor-two
downsampling steps. See the [configuration][moge-config], [encoder][moge-encoder]
and [dense decoder][moge-dense] for the complete definitions.

The existing [`vit_embed.cpp`](../src/vit_embed.cpp) provides useful ggml
patterns, but its fixed image/pooled-embedding API does not implement MoGe's
dynamic positions, LayerScale or intermediate feature taps. Build the actual
DINOv2 graph rather than substituting a pooled image embedding.

The converter should preserve source tensor names or publish a complete,
reversible name map; record source/checkpoint revisions, hashes, configuration,
layout and schema version. Validate every mandatory tensor and shape, and fail
on an unsupported configuration. Handle ordinary and transposed convolutions
separately. Flatten a five-dimensional sparse kernel into a documented
ggml-compatible layout without changing tap order. A successful archive load
must never substitute randomly initialized missing weights.

Check preprocessing and decoder behavior explicitly: antialiased bilinear
encoder resizing, `align_corners=False`, positional interpolation, replicate
padding, transposed-convolution layout and final non-antialiased bilinear
resizing. Preserve the factorized point representation while resizing; apply
the log-depth exponential at the same stage as upstream. These details come
from the [encoder][moge-encoder], [decoder][moge-dense] and [v3 forward][moge-v3].

### Sparse topology and arithmetic

The [official sparse blocks][moge-blocks] and [LibreYOLO reference][libreyolo-sparse]
make the sparse contract concrete:

| Operation | Required semantics |
| --- | --- |
| Voxelization | FP32 factorized points `(x/z, y/z, logz)`; quantize log-depth with ties-to-even rounding at the configured resolution (256 in the reviewed configuration), then subtract the per-image minimum bin. Coordinates are `(batch, row, column, depth_bin)`. |
| Active sites | Initial sites correspond to pixels on the decoder grid. Convolution preserves that site's coordinate set; absent neighbors contribute zero. |
| Kernel ordering | Original weights are `(C_out, 3, 3, 3, C_in)`. Preserve row/column/depth tap order when flattening to 27 offsets. |
| Pooling | Floor-divide spatial coordinates by the downsampling factor, group occupied children and average only those children; spatial extents use ceil division. |
| Upsampling | Apply the learned channel projection, then gather through the saved child-to-parent inverse. Restore the exact saved coordinate order before adding skip features. |
| Residual blocks | Match affine LayerNorm with epsilon `1e-6`, SiLU, both sparse convolutions and the residual addition. |

For one sparse convolution, look up each occupied site's neighbors, gather
their feature rows, multiply by the learned offset weights and accumulate at
the existing output sites. Process offsets and/or rows in bounded chunks so
temporary storage does not scale as one fully materialized neighborhood
tensor. Benchmark accumulation order and temporary memory before replacing
this reference arithmetic with a fused kernel.

Build and cache neighbor maps while a level's coordinates remain unchanged;
retain pooling inverses for its decoder. Rebuild topology after every geometry
refinement update. The [v3 implementation][moge-v3] explicitly forces FP32 in
voxelization because depth-bin changes alter subsequent neighborhoods. C++
`std::round` is not a substitute for PyTorch's ties-to-even rounding; include
positive and negative half-bin tests.

A dense 3D-convolution replacement must not introduce values at inactive sites.
It would also waste memory: as an illustration, a `256^3` volume with 64 FP16
channels takes exactly 2 GiB for one activation tensor. This is arithmetic,
not MoGe-3's measured grid size. Keep the occupied representation.

CPU-built maps introduce synchronization with GPU features. Measure map
construction, upload/readback, graph execution and end-to-end latency
separately before moving indexing to Metal. Unified memory does not remove
the dependency between a new geometry prediction and its next neighbor map.

### Camera and output contract

The [camera recovery implementation][moge-camera] assumes a centered optical
center, an undistorted view and equal scale in the two image directions. A
horizontal FOV fixes the focal value but still requires depth-shift recovery.
Its solver inputs use the matching 64-by-64 nearest-neighbor samples of points,
view-plane coordinates and mask. Preserve the sparse-validity fallback behavior
as well as normal inputs.

The native result must retain upstream meaning: OpenCV camera axes, FP32
metric depth/points and normalized intrinsics with principal point `(0.5, 0.5)`.
Specify the normalized-to-pixel and half-pixel convention explicitly. Recover
camera parameters separately for each requested refinement-step output; apply
shift, optional reprojection and metric scale in the upstream order. With
masking enabled, invalid depth/points use positive infinity and invalid normals
use zero. The final mask includes positive-depth validity. See [v3 infer][moge-v3]
and LibreYOLO's [pixel-intrinsics conversion][libreyolo-utils].

### CrispMesh integration seams

CrispMesh's [`sam-ggml` provider][crispmesh-sam] already dynamically loads
`libcrispembed-sam2` and resolves its C ABI. Follow this pattern for the proposed
MoGe provider. Its [experimental learned-depth adapter][crispmesh-depth]
separates `infer` and `reframe`, and the [native stereo runner][crispmesh-replay]
accepts per-view FP32 `[height, width]` arrays through `--reuse-depths`. The
adapter currently runs DA3-BASE; it is an integration seam, not existing MoGe
support.

Keep view order, masks, world-to-camera transforms and the final level's
intrinsics/dimensions fixed. CrispMesh's [projection/unprojection][crispmesh-level]
uses camera-space Z depth and rays `(x + 0.5 - cx) / fx`,
`(y + 0.5 - cy) / fy`. Convert normalized intrinsics into this convention;
do not silently substitute Euclidean ray distance or an integer-centered
wrapper's principal point. Align predicted metric depth with the units of the
recorded camera translations before fusion; any scale fitting needs an explicit
protocol and does not independently validate reconstruction quality.

Treat the model camera and the calibrated/cropped camera as separate coordinate
systems. Horizontal FOV alone does not correct an off-center crop. The existing
`reframe_depth` handles **intrinsics changes within the same camera
orientation**. A virtual-camera rotation also requires the corresponding ray,
Z-depth and extrinsic transformations; simple intrinsics resampling is
insufficient. The [adapter tests][crispmesh-depth-tests] already cover useful
off-center, identity, invalid-hole and virtual-camera cases.

Preserve upstream infinity/mask semantics in native/reference outputs. At the
CrispMesh adapter boundary, explicitly map invalid/nonpositive depth to its
zero-invalid representation before reframing: the existing command rejects
nonfinite source planes. Write the ordered `depth_NNN.npy` members to the NPZ
archive and check dimensions and target-camera reprojection before replay.
Keep input images, calibration, crops, masks and reconstruction settings fixed
during comparisons.

### Native interface and ownership

Follow the existing C ABI and Rust wrapper conventions, with explicit model
load, inference, backend inspection and destruction. The planned request needs
image dimensions/stride, token or resolution selection, refinement count and
optional horizontal FOV. The result needs dimensions, depth, validity mask,
intrinsics and optional points/normals/per-step diagnostics.

Choose caller-owned buffers or a documented native result/free pair before
publishing the ABI. Validate dimensions, multiplication overflow, lengths,
finite input and FOV bounds before allocation. Make allocation/model/backend
errors distinguishable from a successful empty mask. State context
thread-safety and result lifetime in C, Rust, Python and Dart.

Keep new inference and optimized paths behind documented default-off gates
using `core_env::on()`. Reference observers belong in test targets, not shipped
libraries. A zero-refinement result is a useful dense-stage milestone and must
be labeled as such; complete MoGe-3 support includes the sparse refiner.

## Implementation sequence and acceptance

All runtime milestones are pending. Record the chosen fixtures and numerical
budgets before evaluating a candidate; report absolute/relative error and
magnitude alongside cosine rather than adopting unrelated models' thresholds.

| Milestone | Deliverable | Completion evidence |
| --- | --- | --- |
| M0 — Evaluate and freeze references | ViT-L evaluation with a complete Apple-capable implementation; immutable source/checkpoint/input/options manifest | Actual device and fallback report; depth/points/intrinsics/mask for the initial, one-step and three-step results; initial reconstruction comparison. Third-party execution remains distinct from Microsoft-reference parity. |
| M1 — Convert and port dense inference | FP32 GGUF converter, preprocessing and DINOv2/dense ggml graph | Metadata/tensor round-trip checks; intermediate encoder taps and dense heads compared with the pinned original; correctly labeled zero-refinement output. |
| M2 — Complete CPU refinement and camera recovery | Sparse indexing, CPU arithmetic, full U-Net, geometry updates and output conversion | Exact synthetic topology/pooling/inverse maps; all 27 distinct kernel taps; stage features and full outputs after one and three steps; fixed/recovered-FOV and invalid-support cases. |
| M3 — Accelerate with Metal | Supported dense and sparse ggml operations, then measured kernel optimization | CPU/original comparisons retained; observed device placement, actual Apple timings, peak memory and CPU/GPU synchronization breakdown. |
| M4 — Integrate and qualify CrispMesh | Standalone library, C/Rust integration, external-depth replay and camera reframing | Real native API result, owned-buffer/error tests, unchanged camera/mask reconstruction comparisons and measured mesh-quality benefit. |
| M5 — Evaluate reduced precision | Selective F16 or quantization candidates after complete FP32 agreement | Per-stage/step numerical and topology comparison plus final mesh-quality and memory/latency results; no automatic promotion based on file size. |

Before numerical comparison, require compatible keys, shapes, dtypes, exact
mask/support classes and finite/NaN/positive-infinity/negative-infinity
locations. Then compare shared finite values. Preserve a failure rather than
dropping invalid entries until a metric passes. Include bin boundaries,
missing neighbors, occupied-only pooling, odd extents, coordinate-order
restoration and topology changes between iterations in model-free controls.
Use independent small fixtures such as the [MLXcel oracles][mlxcel-fixtures]
with their provenance and notices intact; they supplement a direct pinned
Microsoft model reference.

For application quality, assess silhouettes, thin structures, holes, surface
noise, completeness and scale consistency against the same reconstruction
inputs. Report the selected measurements and cases; a plausible depth image
or improved tensor cosine alone does not establish a better mesh. Collect
wall time, stage timing, peak memory and backend fallback on the actual Mac.

Follow [the contribution checklist](contributing.md): isolated worktrees,
PLAN ownership, hosted builds/model runs, source-bound evidence and exact-head
review. Qualified model publication, SHA-256 registry pins, release assets and
consumer version updates are later explicit gates. This design introduces no
model alias, new runtime switch or release version.

## Licensing and open decisions

The upstream [MoGe license][moge-license] includes MIT code and the DINOv2
Apache-2.0 notices; the [ViT-L model card][moge-weights] labels its weights MIT.
Check the exact artifacts and any adapted third-party source/fixtures before
importing them, and preserve their notices. Conversion does not change rights.

Resolve during M0/M1: initial supported image/token range and memory budget,
the reference-producing Apple implementation, output ownership and ABI schema,
fixed versus recovered-camera evaluation cases, and the actual reconstruction
quality metrics. Resolve after profiling: which sparse arithmetic needs custom
Metal kernels and whether CPU topology construction is a material bottleneck.
Speed, memory efficiency and improved CrispMesh geometry remain measurements
to obtain, not assumptions about Apple hardware.

[moge-root]: https://github.com/microsoft/MoGe/tree/74fbce054ebed49800de42d0ad0e83495065719a
[moge-readme]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/README.md
[moge-v3]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py
[moge-config]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/configs/train/v3.json
[moge-encoder]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/dinov2_encoder.py
[moge-dense]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/conv_stack.py
[moge-sparse]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/sparse_unet.py
[moge-blocks]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/flex_sparse_blocks.py
[moge-camera]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/geometry_torch.py
[moge-license]: https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/LICENSE
[moge-weights]: https://huggingface.co/Ruicheng/moge-3-vitl/tree/184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be
[pytorch-mps]: https://docs.pytorch.org/docs/stable/mps_environment_variables.html
[libreyolo-root]: https://github.com/LibreYOLO/libreyolo/tree/359fc39f9189086c395753df47017a75187b1e48
[libreyolo-docs]: https://www.libreyolo.com/docs/models/moge-3
[libreyolo-model]: https://github.com/LibreYOLO/libreyolo/blob/359fc39f9189086c395753df47017a75187b1e48/libreyolo/models/moge3/model.py
[libreyolo-sparse]: https://github.com/LibreYOLO/libreyolo/blob/359fc39f9189086c395753df47017a75187b1e48/libreyolo/models/moge3/sparse.py
[libreyolo-utils]: https://github.com/LibreYOLO/libreyolo/blob/359fc39f9189086c395753df47017a75187b1e48/libreyolo/models/moge3/utils.py
[mlxcel-root]: https://github.com/lablup/mlxcel/tree/e427b7a02aba041fdbaa847257ccc01fb6100fb0
[mlxcel-docs]: https://github.com/lablup/mlxcel/blob/e427b7a02aba041fdbaa847257ccc01fb6100fb0/docs/dense-prediction.md#moge3-geometry
[mlxcel-report]: https://github.com/lablup/mlxcel/blob/e427b7a02aba041fdbaa847257ccc01fb6100fb0/TECHNICAL_REPORTS/2577-moge3-dense-geometry-20261023.en.md
[mlxcel-fixtures]: https://github.com/lablup/mlxcel/tree/e427b7a02aba041fdbaa847257ccc01fb6100fb0/tests/fixtures/moge3
[mtlgemm]: https://github.com/pedronaugusto/mtlgemm
[crispembed-base]: https://github.com/CrispStrobe/CrispEmbed/tree/06af98b175efe4eb15f12ebce638421972c03a78
[crispmesh-base]: https://github.com/CrispStrobe/CrispMesh/tree/64d7d8f55d1551974e60cae97dd82ea865dc3ea8
[crispmesh-sam]: https://github.com/CrispStrobe/CrispMesh/blob/64d7d8f55d1551974e60cae97dd82ea865dc3ea8/crates/dense/src/photos/sam/ggml.rs
[crispmesh-depth]: https://github.com/CrispStrobe/CrispMesh/blob/64d7d8f55d1551974e60cae97dd82ea865dc3ea8/scripts/photo_regression/windowed_depth.py
[crispmesh-replay]: https://github.com/CrispStrobe/CrispMesh/blob/64d7d8f55d1551974e60cae97dd82ea865dc3ea8/crates/dense/src/stereo/run.rs
[crispmesh-level]: https://github.com/CrispStrobe/CrispMesh/blob/64d7d8f55d1551974e60cae97dd82ea865dc3ea8/crates/dense/src/stereo/level.rs
[crispmesh-depth-tests]: https://github.com/CrispStrobe/CrispMesh/blob/64d7d8f55d1551974e60cae97dd82ea865dc3ea8/scripts/photo_regression/test_windowed_depth.py
[ggml-base]: https://github.com/CrispStrobe/ggml/tree/3a27b6c665b191cb230be367db5c52d81dd3e2ac
