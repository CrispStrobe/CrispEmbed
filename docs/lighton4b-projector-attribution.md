# Pinned projector attribution: setup evidence

The first projector-only worker at source `0f92f3b6a1e6f46a81df15ea5287e2b377a3bb53` stopped during executable linking after250.86seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v1-link-negative/projector-audit.json) SHA256 `5f6845f8d836e4dea130e1c0a52a42947645c677c5216f4e52222d7009fb8c34`; [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v1-link-negative/independent-review.json).

The standalone C++ object and pinned libmtmd compiled, but the reference target requested obsolete library `common`, producing `cannot find -lcommon`. Compile/load-before-transfer admission worked: no parameter transfer, canonical comparison or projector forward occurred. There are no valid stage metrics from this attempt.

The scoped proposed correction `c01008696` removes the unneeded dependency and links only mtmd, which publicly links ggml and llama. Source arithmetic, parameters, CPU profiles, observer controls and resource bounds are unchanged. A retry remains a new result; this failure is retained. No product/runtime code, default gate or public loader changes are included.


## V2: canonical and preprocessing controls valid; intermediates rejected

The corrected-link worker completed318.87seconds at source `c01008696`. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v2-qualified/projector-audit.json) SHA256 `64eb77727c39265cc1cc32cd5290ed501c1e29629f4aba3f44a5b7b2308c9db2`; [qualified independent review](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v2-qualified/independent-review.json). The raw receipt's measured status does **not** validate its intermediate-stage metrics.

All297 native protected vision tensors and298 publisher tensors were authenticated and canonically mapped, covering333,514,240 native F32 values. All native values are exactly representable BF16. The198 F32 source tensors match bit-exactly;99 F16 matrices differ from native F32 values by at most2.9802322387695312e-8. Native-to-F16 rounding matches287 tensors bit-exactly;10 require signed-zero/tie inspection. No full canonical bit-equivalence or original-checkpoint identity is asserted. The temporal patch Conv3D was explicitly joined from the two source Conv2D slices. Total parameter transfer2,006,480,160bytes remained remote; no text checkpoint or full F32 model was instantiated.

Both standalone source projector OFF/ON merged outputs exactly match the retained actual upstream bytes (`b9a171…240d1b`). Valid normalized/preprocessed patch comparison gives RMS .000221816890/max .0156862736 over860,160 values. Six selected vector hashes and the RMS/max were independently recomputed; source OFF/ON merged bytes also independently checked. This is an actual earliest input difference, not decoded-quality proof. Resize pixel differences and normalization arithmetic still require separate capture.

**Reject every postcompute intermediate metric.** Patch+bias, position-added and many odd block outputs have identical hashes; another group contains the even block outputs and a normalization result. Backend memory was reused despite requested output flags. The nonzero check and unchanged final output were insufficient to prove these individual observations. The reported giant patch/layer differences are invalid and must not be used to blame model math; learned-position magnitude is unsupported too. The raw evidence and explicit duplicate groups are preserved, never replaced.

The bounded correction uses live scheduler callbacks to copy a node immediately after execution, per-node callback counts and exact patch+position/FFN-residual identities. It separately captures actual input/resized uint8 RGB and normalized HWC/CHW, reconstructs native resized pixels with exact normalization inversion, and isolates resize from normalization on identical pixels. F16-round numeric/signed-zero samples are added. Seven model-free controls and exact three-source patch anchors pass locally; hosted callback results remain pending. Experimental product gates/defaults are unchanged.

## V3: live stages valid; source precision policy closed

The live-callback worker at source `c1ed8b153` completed350.42seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v3-live/projector-audit.json) SHA256 `427b38fd03a9d4aec47c9292f15f8771e99e263b2258f415149f0f20dd612f0c`; [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v3-live/independent-review.json). All32 selected callbacks occurred once. Live patch+position and first FFN residual additions close byte-exactly; source OFF/ON merged bytes reproduce actual upstream `b9a171…240d1b`. This validates V3 observations, while V2's reused intermediate buffers remain rejected.

The precision-policy audit resolves V2's10 misleading F16-round mismatches: those are **source F32 parameters**, which must remain F32. All198 F32 parameters match exactly. All99 native matrices, corresponding to100 source F16 tensors after the temporal split, match after the required native-to-F16 rounding. Thus all297/298 source-declared transforms close over333,514,240 native scalars. Casting every parameter to F16 would incorrectly change small biases/norms/position values. This is canonical parameter/precision equivalence for these authenticated files, not proof of an original full-checkpoint identity.

Actual resized RGB differs in341 of430,080 channels, maximum2 uint8 levels. On identical source RGB, native versus source normalization contributes RMS5.78718e-9/max5.91390e-8; resize alone contributes normalized RMS.000221816792/max.0156862736. Seventeen selected raw files were independently rehashed, including exact HWC→CHW, normalization, resize counts and both live arithmetic identities. Other source/native stage metrics below remain report-derived rather than an independent complete vector download.

| Actual comparable stage | RMS | Maximum absolute error | Source maximum magnitude |
| --- | ---: | ---: | ---: |
| Packed normalized pixels | .000221817 | .0156863 | 1 |
| Patch+bias | .000821497 | .0123301 | 4.30017 |
| Learned position | .000415226 | .0263162 | 6.15296 |
| Block0 output | .00300514 | .0482702 | See raw receipt |
| Block11 output | .0398324 | 21.2116 | 151.212 |
| Block23 output | 7.98955 | 1794.81 | 5710.86 |
| Merged features | .0112370 | 1.68147 | 5.43132 |

These are the same observed form and authenticated parameters, with different computation semantics: native Torch/BF16 vision versus upstream F16 matrices/F32 states and upstream resize. The live-stage evidence justifies separately gated corrections and source-fed checks; it does not show decoded improvement or held-out generalization. No text model or original BF16 forward ran in this projector-only worker. Experimental defaults remain unchanged and the composed candidate remains unqualified.

## Separately gated implementation: model-free controls pass

The source-compatible draft at `0eca8d13f` adds two default-off paths: Pillow22
resize/source normalization, and F16-matrix/F32-state vision under verified
physical ownership. Existing Torch/BF16 arithmetic and defaults remain. The
vision path explicitly constructs F16 im2col and paired SGEMM, avoiding the
preserved fork's F32 convolution promotion. It uses source precision for every
parameter and F32 states, residuals and GGML operators. The Linux fixed-AVX2
private ABI/four-thread profile is closed; the main-library implementation stays
an unsupported stub. No vendor trait is changed.

The model-free worker completed690.72seconds, [raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-vision-profile-controls/v1/vision-controls.json)
SHA256 `31e6d6e54ec5a26c464e0e0f7a151e2408f04c4ec517b59648997a76a3f5ee09`.
[Independent review](../tests/results/qwen35-ocr/lightonocr3-4b-vision-profile-controls/v1/independent-review.json)
rehashed all58 owned output files. Eight predeclared synthetic resize shapes
match the actual pinned upstream22-bit implementation byte-for-byte. The actual
form's430,080 resized RGB channels and860,160 packed F32 values also match live
V3 source bytes exactly. These directly close the previously observed341-channel
resize and normalization discrepancy.

The supported unit executes the complete24-block vision graph on a minimum2×2
synthetic grid with shared64MiB aligned zero matrices and unit norm scales:
37 checks verify execution, shapes, finite zeros and gate clearing. Main-library
unsupported guards pass7 checks; Pillow/gate and main-link units pass17 each.
These are synthetic component controls, **not actual-weight vision-stage or
decoded-quality proof**. No checkpoint was acquired and no original BF16/text
forward ran. Timing includes native/source builds and is not a runtime speed
comparison. The initial title/slug400 submission rejection is retained separately;
it created no accepted worker.

Next is one bounded projector-only actual-weight comparison: authenticate all297
tensors of the1.334GB protected vision suffix, reproduce retained native legacy
stages first, then compare actual source stages and meaningful magnitudes. This
requires the passing controls receipt, source-declared precision guard and exact
RGB/packed-pixel admission before interpretation. Full composed decode follows
only if component evidence supports it. The candidate remains unqualified;
no release/default change or held-out claim is made.

## Actual-component V1: adapter gate admission failed

The first actual-stage worker at `d338c577c` stopped after131.69seconds in the
reference adapter's metadata mapping, before legacy or F16 actual-weight vision
execution. All297 protected tensor SHAs and the source-declared precision policy
passed;1,334,056,960 vision-only bytes were acquired remotely. Synthetic graph
controls passed again. There are no new actual-weight stage metrics.

[Public redacted receipt and logs](../tests/results/qwen35-ocr/lightonocr3-4b-vision-profile-component/v1-map-gate-negative/index.json)
bind the private original receipt SHA256
`760db939e0a783812d4350dc517f32e1663809e15d7f4c19c1a5b8b06c3dfef3`.
Private mounted-evidence locators are redacted in the public copies; the index
records both original and public hashes. Status and measurements are preserved.

The adapter enabled its OCR gate after calling `map_lighton4b_vision`; the
controller deliberately clears inherited gates. Production code is unaffected.
Reviewed correction `b59e5d3a9` moves gate setup before mapping and adds a real
metadata-only CLI test using the same297-name schema and a sparse zero-data
mapping. Positive/OCR-off rejection/restored admission now execute before any
parameter transfer. The metadata schema retains SHA256 `1de40b987ba017a8c583e96ada32f8155a63383ed82be0f1702d7ff43fb9b6db`.
The sparse test is removed before real acquisition; it reads no weights and runs
no forward. Source/GGML/CMake/numerical-unit bytes stay strictly equal to the
passing controls; the reference-only adapter exception has its own reviewed
hash. Seven local controls pass. Retry results remain separate and unqualified.

## Actual-component V2: all28 vision stages byte-exact

Corrected worker source `b59e5d3a9` completed280.59seconds. [Raw receipt and index](../tests/results/qwen35-ocr/lightonocr3-4b-vision-profile-component/v2-exact/index.json)
SHA256 `039b024f41b499cfd8344cb4e89835bbfe559a1f298225866c691000d4e6c8c0`.
The new metadata-only admission passed before transfer. All297 native vision
parameters were authenticated, with198 sourceF32 parameters retained asF32 and
99 matrices rounded to their100 sourceF16 tensors. No text checkpoint, original
Torch/BF16 reference or new upstream forward was instantiated.

The retained legacy path reproduces all28 visual arrays plus processor patches
exactly. The new F16-matrix/F32-state path then matches **all28 actual live source
arrays byte-for-byte**: patch+bias, learned position, embedded input, every24
block output and final merged features. Every measured RMS and maximum error is
zero. Actual resized RGB and normalized packed patches also match source bytes.
This is a real-weight/source-stage result on the observed form; it does not imply
all inputs, text-decoder parity or OCR quality qualification.

[Independent review](../tests/results/qwen35-ocr/lightonocr3-4b-vision-profile-component/v2-exact/independent-review.json)
verified nine selected actual files totaling18,206,720bytes, including raw bytes
for initial stages, blocks0/23, merged features and preprocessing. All28 reported
new hashes cross-close against both the output manifest and retained source
manifest; all29 legacy hashes cross-close against the old composed receipt.
The full73-owned-file and28-source-file remote rehash remains an explicit
pre-model gate in the composed-form follow-up, avoiding local bulk arrays.

| Ordered vision-only arm | Elapsed | Sampled process peak RSS |
| --- | ---: | ---: |
| Legacy Torch/BF16 path | 138.00s | 1,504,272,384bytes |
| Source F16-matrix/F32-state path | 15.47s | 1,508,159,488bytes |

These single ordered component timings include their diagnostic snapshots and
are not a balanced speed benchmark or full decode measurement. Both use the
same weights, image and four threads. Gate/owner setup is explicit in the
reference adapter; its recorded launch environment is not the complete effective
in-process gate tuple. Production numerical code is unchanged from the passing
controls. Defaults remain off. Complete independently decoded composed-form
comparison is next; the candidate remains unqualified until broader evidence and
reference limitations are resolved.
