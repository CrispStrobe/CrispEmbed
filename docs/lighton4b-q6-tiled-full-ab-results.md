# Physical Q6 tiled: observed full decode

The default-off physical Q6 tiled option improves both observed cases against the frozen upstream Q4_K_M output under the unchanged BF16-transcription denominator. This is **not qualification**: Arabic and the frozen held-out panel have not been replayed with this configuration. The earlier held-out receipt failure and provisional-label limits remain; no release or alias change is authorized.

[Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q6-tiled-full-ab/v1/q6-tiled-full-ab.json) `f5d0077052b907e2d674caef9e7c4ae09b08456ff48a79f327cbdfd030b2dcc9` comes from `lighton4b-q6-tiled-full-ab-v1`, version1, source `8fcdf1f739fedf9ef3a068ad8028bc108ad5bc8a`. The model remains physical candidate `843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`. Only `CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED` changes. GDN source operators and Q4 repack stay ON in both arms; corrected vision and F16 attention remain fixed.

| Case | Upstream character/word edits | Q6 OFF character/word edits | Q6 ON character/word edits | IDs/EOS OFF → ON |
|---|---|---|---|---|
| Table | 348/607; 97/106 | 348/607; 97/106 | **337/607; 95/106** | 282/EOS → 282/EOS |
| Form | 189/656; 46/90 | 194/656; 40/90 | **175/656; 45/90** | 167/EOS → 167/EOS |

The form's word error increases relative to OFF, while remaining below upstream's 46 edits. These are raw transcription distances against the frozen BF16 reference, not expert-adjudicated ground truth. Actual upstream output has 282 table IDs and 170 form IDs; token or numerical identity is not claimed.

Table first-score RMS against actual upstream improves `.188863288 → .178908697`, but form RMS worsens `.123101444 → .152798624`. Both first differing paired native hidden stages are layer0; these are native OFF/ON comparisons, not actual upstream hidden-state parity. Exact same-batch operator proof does not remove the native16 versus upstream `[4,140,6,4]` form-batching limitation.

Every OFF greedy row (282/167), 38 text stages, teacher row0, vision/processor bytes, IDs, text and EOS matches the prior combined baseline. ON retains identical vision and decoder inputs. All full vocabulary rows are finite with matching argmax; prefill/teacher0/greedy0 bytes agree. Both arms retain216 quantized Q4 buffers within2GiB and zero converted-F32 weight entries. The new model-free unit passes38 controls, including widths256/768/9216, actual eligible production dispatch and immutable cache-mode transitions; existing GDN, attention and repack controls pass before weights are acquired.

Table wall time is191.38s OFF /196.75s ON; form142.07s /150.65s. This worker is slower with the option enabled; no speed claim. The worker completes in1058.84s including build, authenticated staging and all four model arms.

[Independent review](../tests/results/qwen35-ocr/lightonocr3-4b-q6-tiled-full-ab/v1/independent-review.json) rehashes19 selected owned files plus immutable references, recomputes raw text/IDs/EOS/cache checks and CER/WER with a separate dynamic program, and reproduces all four first-score metrics from six authenticated arrays. Full greedy/stage arrays remain remote and producer-guarded; they were not independently replayed in full. Preserve all earlier negative alternatives and keep every experimental gate default-off until the remaining fixed panel passes.
