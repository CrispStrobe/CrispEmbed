# Encoder Q8/Q4 evidence and remaining gates

This read-only audit distinguishes existing precision artifacts from numerical
parity, task quality and current-source qualification. It adds no model,
quantization, inference, publication or default change. The
[machine-readable audit](../tests/results/encoder-precision/evidence-audit-20261029.json)
records exact sources, artifact hashes, seven bounded Hub metadata responses,
report identities and executable next steps.

**LFM2.5-Encoder230M already has Q8 and genuine calibrated Q4_K evidence. Typed
EmbeddingGemma already has Q8 evidence; its Q4_K recipe and qualification remain
missing.** Neither family's small reference suite is an independent task benchmark.
Do not recapture the existing original references to fill a precision/client gap.

## LFM2.5 masked Encoder230M

This is the bidirectional masked encoder, not the separately trained
LFM2.5-Embedding350M retrieval model. Its15 cases contain15 total mask predictions
and reach649 tokens. The original FP32/eager reference uses Torch2.11.0 and
Transformers5.13.0.dev0. Existing measurements are in
[quantization](../tests/results/lfm2-encoder/quantization.json),
[mixed precision](../tests/results/lfm2-encoder/mixed_precision.json) and
[arithmetic attribution](../tests/results/lfm2-encoder/arithmetic_audit.json).

| Stored artifact / CPU arithmetic | Actual stored types | Minimum token cosine to original | Mask agreement |
| --- | --- | ---: | ---: |
| Official F16 |83 F16 +49 F32|0.999989475|15/15|
| Official Q8_0, ordinary|83 Q8_0 +49 F32|0.946652176|13/15|
| Official Q8_0, FP32 row-dot|same stored Q8|0.995878021|14/15|
| Calibrated Q4_K, ordinary|82 Q4_K +1 Q8_0 tied table +49 F32|0.752906787|11/15|
| Same calibrated Q4_K, FP32 casts|same stored Q4_K|0.889689598|12/15|
| Mixed210, ordinary|28 Q4_K +55 Q8_0 +49 F32|0.881376436|13/15|
| Mixed210, FP32 row-dot|same stored mixed types|0.928912007|14/15|
| Mixed330, ordinary|54 F16 +29 Q8_0 +49 F32|0.987746257|14/15|
| Mixed330, FP32 row-dot|same stored mixed types|0.999817563|14/15|

Official Q4_0 is a separate149MB artifact:82 Q4_0 matrices, one Q6_K tied table,
and49 F32 tensors; it reaches0.736031801 and9/15. It is not Q4_K. Plain local
Q4_K reaches0.556311435 and10/15; it must not be substituted for the calibrated
published Q4_K hash. The [publication receipt](../tests/results/lfm2-encoder/publication.json)
identifies the165,331,968/209,893,888/330,152,448-byte published artifacts and their
actual tensor-type counts.

The critical mixed-profile groups are attention, ShortConv input/output
projections and FFN down:54 matrices. Mixed330 keeps those in F16, the other28
gate/up matrices and tied table in Q8_0, and49 norm/kernel tensors byte-identical
F32. Mixed210 keeps those54 matrices and tied table in Q8_0, with28 calibrated
Q4_K gate/up matrices and the same49 F32 tensors. All26 screened profiles remain
recorded. Upgrading additional tensors was not uniformly better; the remaining
mixed330 mismatch predicts `東京` where the original predicts `京都` on a
multilingual stress probe. This is reference disagreement, not a labeled factual
accuracy score.

Initial Q4_K importance calibration used134 sentences/151 samples/6,711 tokens;
expanded calibration used174 records/185 samples/12,065 tokens, maximum1,149.
Both collected82 importance vectors with zero exact evaluation-text overlap.
Precision policies were nevertheless selected using the15 probes, so those
probes are development evidence. Q8_0 encoding ignores imatrix values. Original
corpora, source/calibration hashes and49 preserved tensor names are retained.

The same-weight control separates activation arithmetic from storage loss.
With exactly dequantized official Q8 weights in original Python, both opt-in
`CRISPEMBED_LFM2_F32_MATMUL=1` and `CRISPEMBED_LFM2_F32_DOT=1` pass297 layer checks
and15/15 same-weight masks; final minimum token cosine is approximately
0.999999707. That does not restore full original-checkpoint parity: original
comparison remains14/15. These flags change arithmetic, not stored weights;
an alias does not enable them automatically.

The retained ordinary649-token layer traces expose early errors and scale:

| Profile | First stage below legacy0.999 cosine | Stage cosine / max absolute error | Final-norm cosine / max absolute error |
| --- | --- | --- | --- |
| Q8_0 |layer5|0.998871 /2.82|0.946652 /5.21|
| Plain Q4_K |layer0|0.997529 /0.321|0.556311 /22.8|
| Calibrated Q4_K |layer0|0.998555 /0.219|0.752907 /14.4|
| Mixed210 |layer4|0.998923 /0.569|see complete retained report|
| Mixed330 |layer8|0.998635 /0.668|0.987746, decoded mismatch retained|

These are reported captured-stage values, not a new causal experiment. Historical
LFM API gates are minimum-row cosine0.999 and relative L2 at most0.02; direct
layer checks also enforce a5% norm-ratio bound and compare decoded masks. Absolute
error is reported, without a universal0.002 gate. For example, the passing
same-Q8-weight297-check replays record maximum absolute0.747 at layer9 against
reference norm924.318. Do not relabel these as passing a newer all-stage0.002
budget or infer exact equality from rounded printed cosines.

Official F16 CPU/TeslaT4 evidence is stronger than its precision neighbors:
[CUDA audit](../tests/results/lfm2-encoder/cuda-f16.json) passes594 direct checks
and both15-case ordinary API panels; the
[paired performance review](../tests/results/lfm2-encoder/cuda-f16-performance.json)
checks420 timing samples. Both are F16-only and session/device scoped. They do not
qualify Q8/Q4 MMQ, other GPUs,128,000-token metadata capacity or independent task
quality. Preserve the [initial performance timeout](../tests/results/lfm2-encoder/initial-cuda-performance-timeout.json)
and the incomplete calibrated-Q4 row-dot optional screen, exit137; the latter
supplies no quality conclusion.

## Typed EmbeddingGemma2

The complete fine-tune is pinned to
[`fcc165b7`](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision/tree/fcc165b7d7d97827e241e09752e83fd11165a74b).
Its trained schema-conditioned head, original prompt, mean pooling, renderer,
option order and fitted temperatures are part of inference. This is not the
separate embeddinggemma300M retrieval model.

| Existing artifact | Storage/preservation policy | Measured reference scope |
| --- | --- | --- |
| FP32,1,102,182,464 bytes|original floating weights|145 checks: five probes×29 stages, including PLE and549-token local-window case; eight typed answers|
| Q8_0,307,492,928 bytes|2D `encoder.*` matrices excluding embedding projection quantized; head, output projection, norms/biases/scalars/temperatures remain F32|30 decision-bank texts across three scenarios;8/8 typed answers, explicit approximation mode|
| Q4_K |no typed recipe/export/result found|unqualified; do not borrow other Gemma Q4 reports|

The FP32 stage gate is minimum cosine0.99999 and relative L2 at most0.003. The
[ARM stage report](../tests/results/gemma-decision/arm-validation.json) records
maximum relative error1.481e-5 and maximum absolute0.0164185 at projected tokens
(reference norm19,613.900). Again, no universal absolute0.002 gate is established.
Q8 decision-bank minimum cosine0.999956429, relative error0.00736837 and maximum
absolute0.00115769 do not pass that FP32 embedding gate; Q8 was deliberately run
with `--measure-only`. Its typed labels still match8/8, with maximum probability
error0.002824783 on ARM and0.002825319 on macOS. Preserve both facts. FP32 errors
are4.82762e-6/3.87479e-6 respectively.

The head-on-original-embedding test remains strict and separate. The publisher's
probability temperatures are retained without refitting; this is not an
importance-matrix quantization calibration. Quantized results require
`CRISPEMBED_GEMMA_DECISION_F32_DOT=1`. The exporter currently accepts only `f32`
and `q8_0`; its FP32 head-preservation policy is visible in
[the converter](../models/convert-gemma-decision-to-gguf.py). The guides attribute
FP16 overflow to the publisher; this audit establishes no new FP16 experiment.
A future critical-tensor recipe should preserve F32 until separately justified.
Actual per-tensor published Q8 counts are not in the retained summaries; do not
invent a census from the filename or exporter policy.

[Publication](../tests/results/gemma-decision/publication.json) and
[fresh alias readback](../tests/results/gemma-decision/registry.json) are retained.
Actual C/Python/CLI/HTTP/schema/head/error/repeat contracts cover FP32 and Q8;
fresh-cache CLI readback checks both aliases/eight answers each, with declared
probability bounds3e-4 FP32 and3e-3 Q8. The
[real Rust/Dart helper](../tools/ci-heavy/gemma_decision_bindings.py) hardcodes
`gemma-decision-f32.gguf`: its three-scenario binding receipt proves FP32,
not Q8. That helper is byte-identical to the publication source. Q8 Rust/Dart
therefore remain precision-specific work. The dedicated Gemma context explicitly initializes
`ggml_backend_cpu_init()` and a one-backend scheduler; GPU support needs an
explicit backend path before qualification, not merely a GPU machine or generic
CLI flag. Likewise, the LFM Rust live test names
F16, and its Dart example accepts a path without a retained per-quantization
receipt. `tests/test_lfm2_bindings.py` targets the different350M retrieval model.

## Immutable metadata and present-source limits

Seven unauthenticated public metadata requests total approximately33KB; no
model, reference array, tokenizer or package payload was fetched. All eight
published GGUF size/LFS-SHA declarations match tracked receipts and current
registry pins at the recorded immutable revisions. Both fixture trees remain
listed: LFM original-FP32 and same-Q8-FP32 at`e3599a94`, typed Gemma original-FP32
at`508188c3`. The JSON preserves every tree member identity for hosted reuse.
The three LFM audit reports and five typed-Gemma scalar reports match exact Hub
Git blob identities; the later pinned registry report matches too. Local Gemma
publication/validated-artifact wrappers are not byte mirrors of the older remote
reports; their bodies were not downloaded to infer stronger correspondence.

This metadata check is not a fresh full-file readback. Historical payload
readback receipts remain the source for that claim. Actual GGUF
`general.file_type`/quantization-version headers were not read; tensor storage
Q4_K enum12 must not be confused with standard file-type enums14/15. Verify
headers and actual inventories on hosted resources before any new publication
or metadata bridge; do not mutate existing published hashes from this audit.

The JSON records current and historical Git blobs. Current LFM arithmetic and
Gemma-specific source leaves match later archived sources, but shared loader,
C ABI/glue, root CMake and GGML differ: historical backend890 versus currenta06.
Unchanged model-specific files do not prove current whole-library parity.
Original LFM HF checkpoint revision is missing from the old quantization
manifests; immutable reference fixtures and model hashes remain available.
Never replace a missing provenance field with a latest-checkpoint fallback.

## Executable remaining lanes

1. **Replay existing references on a new exact source closure.** Use pinned Hub
   models and fixture trees, verify every file before loading, and build only on
   hosted CPU. Run existing `test-lfm2-diff`/`lfm2_encoder_parity.py` and typed
   `test-gemma-decision-diff`/`gemma_decision_parity.py` through a separately
   versioned replay-only caller. Do not schedule the old heavy validator unchanged:
   it downloads the checkpoint and dumps originals again. Keep the historical
   numeric gates, IDs/vocabulary/schema/head, ordinary/repeat and decoded checks;
   report current compiler/backend/binary identities and quantized drift separately.
2. **Prepare typed-Gemma's first protected Q4_K profile.** Start from the pinned
   complete FP32 GGUF, preserve head/projection/temperatures/norms/scalars in F32,
   validate every eligible shape/block width and actual stored type, reject Q4_0
   fallback. Freeze any calibration/development corpus before outputs. Reuse five
   layer fixtures and eight typed answers; add explicit Q8/Q4 Rust/Dart actual
   clients. A development result is not independent task accuracy or publication.
3. **Qualify existing LFM precision choices on a declared heldout task.** Reuse
   published Q8/Q4/mixed files and original/same-weight fixtures. Preserve the
   low-margin mismatches and all26 profile records. Do not fit another precision
   selection to these same15 probes and call it heldout quality. Extend precision-
   specific public consumers where a receipt is absent.
4. **Then test low-precision GPU/MMQ and paired performance.** Pin actual device,
   model/arithmetic/storage, compiler and placement. Require matched F16/FP32
   controls, per-layer Ref/independent magnitudes and decoded outputs before
   latency/RSS claims. No CPU fallback, GPU lottery or inference from F16 T4
   performance to Q8/Q4. Typed Gemma requires an explicit GPU implementation first.
   Long-context and representative task gates remain separate.

No lane above is submitted by this audit, and no current default, alias, app pin
or publication state changes.
