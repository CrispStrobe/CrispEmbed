# Unqualified BF16 vision candidate

Actual source891d07da39e9c361a206f2ec4b40fe8339712e0f preserves source BF16
boundaries behind default-off`CRISPEMBED_LIGHTON4B_VISION_BF16`; legacy shared
vision remains unchanged. Corrected Q8/Q4 smokes42005430424/42005434552 complete
but retain capped prefixes`<figure`/`<table>` respectively. Q4's initial decoded
regression is unrepaired. Both initial768-only normalization-guard failures are
preserved in`lightonocr3-4b-vision-bf16-initial.json`.

[Actual stage comparison42006413571](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42006413571)
authenticates all original/native stage bytes. Position interpolation becomes
bit-exact and patch discrepancies fall to20/573440 values (max0.00390625,
RMS8.26115607e-6), versus legacy max0.0126757622/RMS0.000805617467. Only seven
embedded values differ. This component improvement does not establish decoded
quality: layer0 max0.03125/RMS0.000781619591 grows to layer23 max3128 and merged
max2.46484375/RMS0.0223860911, worse than legacy merged RMS0.0112389284.

Q8 first-score RMS worsens0.130495035→0.200536952 (max1.09219897,
cosine0.996400623). Q4 RMS changes0.571912783→0.545669756 but maximum worsens
2.49422884→2.61007118 and its decoded selection stays negative. No default,
quality, speed or release promotion follows. The
[complete numerical report](../tests/results/qwen35-ocr/lightonocr3-4b-vision-bf16-stages.json)
and [review](../tests/results/qwen35-ocr/lightonocr3-4b-vision-bf16-review.json)
preserve magnitudes/errors and exact sources. Raw archives remain remote.

The next unexecuted attribution would compare source-fed patch accumulation and
first-layer norm/attention/MLP boundaries using exact retained inputs/weights,
with an original-operation control before another arithmetic change. User-requested
CI/PR consolidation pauses new dispatches. The experimental branch and gated
alternative remain retained; closing its draft PR does not erase this evidence.
