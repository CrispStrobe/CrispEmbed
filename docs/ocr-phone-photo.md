# Phone-photo geometry and cleanup

The shared OCR crop has an opt-in true projective warp. The scan deskew detector
has an independent opt-in bounded analysis path for large camera images. Existing
defaults and recognition models remain unchanged.

## Geometry contract

- `CRISPEMBED_OCR_CROP_PROJECTIVE=1`: `ocr_crop::extract_quad` computes an inverse
  homography and bilinear pixel samples. Four distinct convex corners may arrive
  in any order. Centroid ordering avoids duplicate-corner selection for diamond
  shapes. NaN, concave/degenerate quads, unsupported channel counts, excessive
  padding and output sizes fail with an empty buffer and zero dimensions. Gray,
  RGB and RGBA retain their channels. Padding replicates the quad edges.
- `CRISPEMBED_SCAN_BOUNDED_DESKEW=1`: Hough and the independent differential skew
  detector analyze the same area-averaged image, at most 1200 pixels on its long
  side. The RGB entry point creates only that bounded float analysis buffer.
  Final rotation samples the original full-resolution pixels and uses white fill.
  Valid input has 3–32768 pixels per axis, at most 64 million pixels, and a finite
  maximum angle in `(0,45]`. Invalid RGB calls clear their output pointers and sizes.
- Unset, empty, `0`, `false`, `off` and `no` leave the corresponding experiment
  disabled. This is not automatic page-corner detection or a model qualification.

The old quad path bilinearly interpolates corner positions and uses nearest-neighbor
sampling. That map is not a homography; a trapezoidal camera view can retain
uneven scaling after its crop. The independent synthetic coordinate-ramp test
measures mean absolute error **0.283081/255** for the projective path, compared
with **11.1119/255** for the legacy map. All 24 corner permutations yield identical
projective bytes. Disabled-gate output stays byte-identical to the prior path.

## A phone preparation recipe needs illumination correction

Correcting angle alone can make a shadowed page harder to recognize: newly white
rotation margins meet its dark photographic background. The retained
[initial deskew-only receipt](../tests/results/ocr-phone-photo/initial-deskew-only.json)
shows this concrete failure on both the existing and bounded detectors. The angles
were correct, but CER rose from 0.02075 to 0.71162 and from 0.12448 to 0.87759.

The application phone profile should combine geometry with the existing
`scan_cleanup_whiten` illumination correction (kernel 51 in this panel), while
offering an original/processed preview. The tested recipe is RGB deskew followed
by the existing grayscale whitening primitive for shadowed fixtures. It does not
claim coverage of learned denoising, automatic border finding or a complete model
orchestration run. The final report retains the deskew-only outputs separately;
adding the required illumination stage does not conceal that negative result.

## Measured text quality

The [quality receipt](../tests/results/ocr-phone-photo/quality-v1.json) records
exact synthetic ground truth, source hashes, font hash, binary hash, actual reader
version, texts, CER, WER, character/word counts, angles, dimensions and timings.
The fixtures use printed service records, ±rotation, a lighting gradient,
deterministic noise, JPEG compression and an independently constructed camera
projection. Tesseract is an independent downstream reader; these measurements do
not qualify a CrispEmbed recognition engine.

| Synthetic case | Raw CER | Prepared CER | Preparation |
| --- | ---: | ---: | --- |
| Already aligned | 0 | 0 | No resampling |
| 4.2° clockwise | 0.02697 | 0 | Deskew |
| 6.7° counterclockwise with shadow | 0.02075 | 0 | Deskew and background whitening |
| 10° clockwise with shadow | 0.12448 | 0 | Deskew and background whitening |
| Perspective and shadow | 0.14523 | 0.00415 | Projective crop |

Both skew paths estimate the three nonzero fixture angles to the reported 0.1°
precision. Their corrected text is exact after the declared recipe. The legacy
quad map gives CER 0.00830 on the last case. Projective output retains two spurious
boundary characters; that residual remains visible in the report. These are
bounded synthetic results, not representative real-camera or physical-device
coverage. Both new geometry paths remain opt-in.

The preset acceptance gate is `prepared CER <= max(0.025, raw CER + 0.01)`;
that is a non-regression check with a 2.5% floor, not an unconditional clean-text
guarantee. The perspective gate separately requires CER at most 0.025 and no
increase above legacy CER plus 0.005. Angle error is limited to 0.55°.

## Repeat the bounded checks

The standard CMake target is `test-ocr-photo`. A small production-source-only
build avoids downloading weights or compiling the full inference runtime:

```sh
git submodule update --init ggml
mkdir -p build-ocr-photo/artifacts
c++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Isrc -Iggml/include \
  tests/test_ocr_photo.cpp src/ocr_crop.cpp src/scan_cleanup.cpp \
  src/classical_preproc.cpp src/morph_fast.cpp -Wl,--gc-sections \
  -o build-ocr-photo/test-ocr-photo
build-ocr-photo/test-ocr-photo self
python tests/test_ocr_photo_quality.py \
  --binary build-ocr-photo/test-ocr-photo --artifacts build-ocr-photo/artifacts
```

The Python runner needs Pillow, NumPy, the DejaVu Sans font and an English
Tesseract installation. [The hosted workflow](../.github/workflows/ocr-photo-quality.yml)
pins Python dependencies and preserves original/processed images plus its report.
The geometry control also runs in the ordinary model-free native build. No
physical-phone run, broad native-model replay, release or default promotion is
implied by these checks.
