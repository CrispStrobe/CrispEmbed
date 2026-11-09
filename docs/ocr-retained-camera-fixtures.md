# Retained camera fixture qualification

This bounded lane extends the synthetic phone-photo geometry checks with two
existing photographic inputs. It changes tests only; the production geometry,
recognition defaults and paired native writer pin remain unchanged.

The public-domain [historical document photograph](https://commons.wikimedia.org/wiki/File:Bernhard_Freudenberg_(1823-1877)_citizenship_document_from_1848.jpg)
already stored at `tests/regression/images/cc0/german_official_document.jpg`
contains EXIF identifying Apple / iPhone 7. Its exact 312,092 bytes and 960×1280
decoded dimensions are checked before use. Four manually selected page-border
pixel centres exercise the public perspective API against independent
[OpenCV perspective mapping](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
The torn, curled historical sheet is not a perfectly planar page: matching this
user-selected quadrilateral is not evidence of recovering its physical shape.
The maximum allowed mean pixel difference is 1/255; the 99th-percentile limit is
4/255, accounting for the reference reader's quantized linear interpolation.
An identity correction must preserve every decoded RGB byte.

Both this photograph and the existing `commons_test_ocr_document.jpg` printed
page are checked unchanged and with independently added −4.2° and +6.7°
rotations. The estimated correction must change by that known rotation within
0.8° of the original estimate. This measures recovery of added tilt; it does
not turn the original estimate into a measured physical camera angle.

Only the modern printed page has independent full-text ground truth. Official
Tesseract runs English plus French with automatic page segmentation on raw,
deskew-only and deskew-plus-whitening arms. The declared prepared arm is always
deskew plus whitening. Its Unicode-NFC/whitespace-normalized CER must remain at
most 8% and may not exceed the raw arm by more than 1.5 percentage points. Case,
accents and punctuation remain in the score. Text, character/word/line counts,
CER, WER, mean word confidence and elapsed time are retained for every arm;
the historical Fraktur/manuscript page has no invented text score.

The [hosted workflow](../.github/workflows/ocr-camera-fixtures.yml) compiles the
actual production photo translation units and the existing C driver. The
report binds the source commit, source files, binary, original hashes, fixture
licenses, selected EXIF fields and reader versions. PNGs preserve the pixels
from every arm; native transport PPMs are reproducible and removed after PNG
preservation. An initial failure remains evidence and must not be replaced by
an after-the-fact successful-arm selection or unexplained threshold increase.

This is qualification of retained camera input, not a newly observed capture,
mobile application workflow, native recognition model, broad document corpus,
dewarping system or physical-device memory/thermal behavior. Hosted execution
and its exact receipt are required before claiming these checks passed.

## Executed source and measured results

[Run 42635475023](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42635475023/job/129222355005)
passes at immutable source `8021ca2603ba84f948057b30bb3adf88e1afd04a`.
The six production translation-unit/header hashes match the application's
qualified `9c44a69` runtime exactly. The [artifact receipt](../tests/results/ocr-phone-photo/camera-8021/artifact-identity.json)
and [complete report](../tests/results/ocr-phone-photo/camera-8021/report.json)
retain the exact fixture, ground truth, binary and reader identities.

The selected iPhone quadrilateral produces 755×1259 pixels. Its mean absolute
channel difference from OpenCV is 0.104874 on the 0–255 scale; the 99th
percentile is 1 and maximum 4. The full-frame identity assertion passes. The
initial artifact omitted that identity PNG; the `fe0e279` follow-up preserves
its geometry and PNG on both success and failure. This omission is explicit
rather than treated as independently retained identity pixels.

Both photographs recover the independently added −4.2° and +6.7° tilt at the
reported 0.1° precision relative to each original estimate. The printed original
has an estimated 0.9° correction, which is not physical-angle ground truth.
No prepared output is downscaled.

| Added rotation | Raw CER | Deskew-only CER | Deskew + whitening CER |
| --- | ---: | ---: | ---: |
| Original | 0.1677% | 0.4696% | 0.3690% |
| −4.2° | 12.2442% | 0.2348% | 0.4025% |
| +6.7° | 61.4894% | 0.3355% | 0.5032% |

All scores retain case, accents and punctuation after Unicode NFC and whitespace
normalization. WER, confidence, line counts and all actual text/TSV files remain
in the receipt directory. The prepared arm was fixed before execution, not
chosen after comparing the results. These inputs show why preparation remains
reviewable: tilt correction recovers the heavily rotated text, while whitening
slightly worsens an already readable original. Earlier shadowed synthetic cases
benefit from whitening; neither result supports making it unconditional.

The archive digest, size, ZIP CRCs, original decoded phone pixels, perspective
error metrics, production source hashes and all reader-output hashes were
checked again after download. Image arms remain available in the pinned hosted
artifact until its recorded expiry; text and numerical evidence are also saved
in git. Physical-device capture and native model qualification remain separate.
