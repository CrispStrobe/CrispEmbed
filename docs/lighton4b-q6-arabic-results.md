# Corrected physical Arabic diagnostic

The corrected configuration preserves the frozen Arabic line exactly: both arms independently generate all six original/upstream IDs and raw text, ending in EOS, with 0/12 character edits and 0/2 word edits. This completes the three **observed** calibration floors alongside [table/form](lighton4b-q6-tiled-full-ab-results.md); the frozen held-out panel is still pending. Gates remain default-off and unqualified, with no release or alias promotion.

[Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q6-arabic/v1/q6-arabic.json) `5da65561b9aec475eb24c1481e4cad2cce27a60dd72fc386bacd26476bef50b3` comes from `lighton4b-q6-arabic-v1`, version1, controller `991fa04b0278c58919e3dc9c064fceca40b6db1e`. It reuses the exact successful table/form binary and libraries from source `8fcdf1f739fedf9ef3a068ad8028bc108ad5bc8a`, without rebuilding. Candidate weights remain `843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`.

This is an explicit **three-mechanism** comparison: GDN source operators, Q4 repack and Q6 tiled change from 0 to 1 together. Corrected Pillow/F16 vision, F16 attention, native projections/RMS and text/head precision remain fixed. OFF first closes every old Arabic score row, 38 text stages, teacher0, vision/processor bytes, complete IDs/text and EOS against the unchanged earlier calibration. Both merged feature arrays are byte-identical to actual upstream; ON also preserves the complete decoder input and positions.

| Measurement | Old OFF | Corrected ON |
|---|---:|---:|
| Generated IDs / completion | 6 / EOS | 6 / EOS |
| Raw character edits / reference | 0/12 | 0/12 |
| Raw word edits / reference | 0/2 | 0/2 |
| First-score RMS vs actual upstream | .109235580 | .127732760 |
| First-score maximum absolute error | .691092014 | .725925922 |
| Arm wall seconds | 24.70 | 33.09 |

Numerical error and runtime worsen despite identical decoded output. There is no numerical-parity or speed claim. Both arms use full independent greedy generation (cap512) after one teacher row. Every vocabulary row is finite and its argmax matches the generated token; prefill/teacher0/greedy0 bytes agree. OFF has no Q4 repacked entries; ON has 216 bounded quantized entries and both have zero converted-F32 weight entries. The first differing paired native hidden layer is 0; actual upstream hidden parity is not asserted. Total worker time is247.22s including authenticated model staging.

The [independent audit](../tests/results/qwen35-ocr/lightonocr3-4b-q6-arabic/v1/independent-review.json) verifies nine selected owned hashes, exact raw text/IDs/EOS against both authenticated references, cache and changed-gate records, and recomputes first-score magnitudes/errors from three rehashed arrays. Large score/stage arrays remain remote and producer-guarded. This short two-word observation is not a general Arabic quality claim, and the BF16 transcription is a reference rather than expert-adjudicated ground truth.
