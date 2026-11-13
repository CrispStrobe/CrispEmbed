# Model-free image drift windows

Artifact-only CPU execution144616841/version1 passes the authenticated source/
native book image window comparison. The [safe selected receipt](../tests/results/qwen35-ocr/lightonocr3-4b-book-image-window-review.json)
contains every16-row window's RMS/max errors and operand magnitudes, fixed
log-scale threshold crossings and selected row metrics. Full226,071-byte report
SHA256 is `0ea2f24770e04a5d3f24b83f86eca59d3f2e5bc3699af5973053db8787349073`;
complete raw rows remain remote. No inference or model download occurs, and no
text or token sequences are published.

The reader reuses original receipt, exact266-row input/position, physical-layout,
shape and output-crop admission. Source graph2 supplies252 image rows; native
full prefill layers0/1 supply the matching global rows4–255. Historical native
chunk sizes are authenticated separately. All indices below are zero-based.

Layer0 has only three image rows whose maximum error exceeds1e-6:23,144 and164.
These are descriptive thresholds, not qualification budgets. The first spike
(image23/prompt27) lies inside the second16-token image chunk:

| Image row | Layer0 RMS / max error | Layer1 RMS / max error |
|---|---|---|
|22|5.47e-9 /5.96e-8|2.84e-8 /9.54e-7|
|23|1.687e-5 /8.720e-5|0.000737949 /0.00258777|
|24|1.71e-9 /1.49e-8|0.000201700 /0.000981227|
|144|1.100e-5 /3.958e-5|0.000578929 /0.00566673|

The first image16 window remains RMS5.508e-9/max1.192e-7. The second window
rises to4.217e-6/8.720e-5; many later layer0 windows return to roughly1e-8 RMS.
Layer1 shows broader propagation after the first spike, with213 rows exceeding
1e-6 maximum error. Its largest maximum occurs at image144.

This isolated layer0 spike followed by later-layer spread is consistent with
pointwise FFN/quantized-projection sensitivity, but hidden outputs alone cannot
prove that cause. Existing operator/cache observation covers only the first
image16 rows. The single next diagnostic captures the second image chunk with
the same complete prefill/cache/hash closures to locate the first spike's actual
operator boundary. No runtime rule, prompt, cap, recipe, scorer or default is
changed, and all prior quality negatives remain binding.
