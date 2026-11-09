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
