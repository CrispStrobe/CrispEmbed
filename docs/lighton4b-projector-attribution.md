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
