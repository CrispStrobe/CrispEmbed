# Critical tensor precision: measured evidence and next ablations

These are development results, not qualified quantized model recommendations.
Read the immutable reports below; preserve each model's original gates. Cosine,
absolute/RMS error, source magnitude and final task outputs are separate evidence.

| Model | What the actual comparison supports | Next bounded ablation |
|---|---|---|
| Josev |[Final-pair Q8](josev2b-last-projection-v1-result.md) first layer23 cosine0.999991033/max0.0559692 versus original magnitude16.7336; typed probability maximum0.0000433922 passes its limited screen.75 strict stages/full serialization remain negative.292 critical tensors already F32, other projections F16. Single gate/up Q4 fails probability gates. |Keep head, norms, vocabulary, convolution and state/gate tensors protected. Compare actual-input calibrated versus uncalibrated final-pair Q4, with fixed old-four controls and no NEW20 selection. More F32 down weights previously offered negligible benefit on these inputs. |
| LFM2.5 masked230M |[Mixed330](encoder-precision-evidence-20261029.md) keeps54 attention/ShortConv input-output/FFN-down matrices F16,28 gate-up matrices plus tied table Q8,49 norm/kernel tensors F32. F32-row-dot cosine0.999817563 still leaves one wrong masked prediction,京都 becoming東京. |Preserve this evidence and test remaining sensitive group through same-input Python/native replay before another profile. Existing calibration screened on development inputs is not independent held-out quality. |
| Typed Gemma |[Frozen audit](encoder-precision-evidence-20261029.md) retains head, embedding projection, norms, scalars and temperatures F32. Published Q8 is approximate; current-source full F32 replay precedes new Q4. |Predeclared broad217-Q4/1-Q8-token-table/218-F32 versus input/PLE-protected168-Q4/268-F32. Actual145 original stages plus typed/head/bank controls must identify the first difference. No refitted temperature or newly relaxed tolerance. |
| MiniCPM |[Complete30-arm panel](minicpm-seven-arm-v3-result.md): Q4 selects186 decoder projections and preserves593 other tensors from the F16-source model, including vision. Q8 also quantizes vision linears. Tiny Q8 first vision block max0.0706520/RMS0.00215511/cosine0.999984931; F32 tiny first strict failure is much later, layer26 max0.0137939/magnitude1629.57. |For a new Q8 recipe preserve ALL nonselected source tensors, including vision tower/projector/norms, and select only the same186 decoder matrices for actual Q8. Authenticate593 byte-exact exclusions before replay. Patch embedding alone is already exact F16 and is insufficient. Keep F32 attribution and96 unmapped crop parts as separate gaps. |
| UltraBERT |[First conservative Q4 shard](ultrabert-first-q4-shard0-result.md) already protects206 F32 tensors and changes only12 late-FFN matrices.686-row cosine minimum0.98075227; strict vectors fail. |Complete the same candidate's remaining shards and unchanged300-query ranking gates before choosing another precision group. An embedding cosine alone does not determine retrieval quality or justify publication. |
| MoEdern |[Current document capture](../tests/results/moedern-colbert/first-row-softmax-initial-review.json) has not completed the actual softmax/router stage comparison. Existing hash/router/head/norm/depthwise protections remain F32. Authored top-k tests do not identify the real-document cause. |Complete genuine Python operation capture and native Ref comparison first. Do not try to repair an unresolved F32 mismatch with a quantized tensor exemption. |
| Unlimited-OCR-2 |[First retained candidate](unlimited-retained-v3-structural-result.md) has2 final expert projections Q4 and737 tensors F16. Structural preservation passes, but initial actual replay stops at a Ref-helper import before inference. |Corrected F16-before-Q4 panel must run before naming another tensor critical. Preserve source-derived router/expert/position shapes and compare local projections as well as decoded body/boxes. |

A protection claim needs an actual type/payload census: a Q4-labeled container
may have most tensors in F16/F32. Never infer lower-bit accuracy by comparing
recipes that protect different tensor groups. Keep critical tensor payload hashes
and storage/semantic axes in the quantization receipt.

For Josev calibration, the real `decoder_hidden_cpu` path forms normalized
2048-wide post-attention inputs directly before gate/up projections. The generic
scheduler imatrix collector is not connected there. A test-only capture must
supply real activation statistics and verify that the C quantizer consumes both
vectors; its nonfatal empty/missing-imatrix fallback cannot be accepted as a
calibrated run. These native calibration inputs are not original Python gold.

Every new recipe remains development-only until unchanged numerical checks,
complete decoded or typed outputs/retrieval rankings, public consumers and
immutable publication/readback gates are satisfied. Preserve negative flags
when task outputs happen to remain exact.
