# PCS `I'm OK` → `I'm oK`: root cause — 20 November 2026

**Status: root cause established. The PCS runtime is correct; the failure is a
property of the 4-bit artifact, independent of the runtime.**

## The failure

The `punct-parity` gate ran `pcs-xlmr-base-q4_k-imatrix.gguf`
(SHA-256 `faee4bf8…31da`, revision `27f17edb`) and required exact agreement
with the blueprint of the pinned original
`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` (revision `d1769a59`,
`model.onnx` SHA-256 `c43ca686…5d89`, decoded by the original
`punctuators` `PunctCapSegResultCollector` at `e0723a4f`). On AVX2 runners it
produced casing 66/67 and decoded 5/6: row37 `▁ok`, bit 1 (`o`) off, so
`I'm OK, how are you?` became `I'm oK, how are you?`. An Intel run passed.

## Evidence

1. **Same weights, original graph.** Replacing all 214 original initializers
   with this artifact's dequantized tensors and running the original graph and
   collector (`tools/pcs_weight_replay.py`) reproduces `I'm oK`, `o` logit
   −0.2383 (original weights: +1.0832). Native stages match that replay within
   7.63e−6 and the 16 row37 logits within 3.64e−6
   ([record](../tests/results/pcs/original-graph-stored-weight-blocker-20261118.json)).
   So the decoded error exists with the native runtime bypassed.
2. **Tensor-group bisection** (`tools/pcs_tensor_bisect.py`, new): only the
   named groups take stored values, everything else keeps the original F32
   ([record](../tests/results/pcs/tensor-group-attribution-20261120.json)).

   | groups from q4_k-imatrix | row37 `o` logit |
   |---|--:|
   | none (original) | +1.0832 |
   | all | −0.2383 |
   | `emb.` | +1.1458 |
   | `enc.` | +0.0347 |
   | `enc.0`–`enc.5` | +0.0243 |
   | `enc.6`–`enc.11` | +1.0988 |
   | `enc.0.` / `.2.` / `.3.` alone | +0.6479 / +0.8982 / +0.8237 |
   | `head.tc.` (FC2 is Q4_0) | +0.8148 |

   No group alone flips the sign; the largest contributions are enc.0 (−0.435),
   head.tc (−0.268), enc.3 (−0.260) and enc.2 (−0.185). It is summed 4-bit
   rounding, not one defective tensor or conversion step.
3. **Ground-truth artifact.** The whole `pcs-xlmr-base-q8_0.gguf` in the same
   replay gives +0.9921; the 2026-08 native measurement was 67/67 and 6/6
   exact. A what-if q4_k-imatrix with enc.0 and head.tc taken from q8_0
   gives +0.6465.
4. **Why the Intel run passed.** Native AVX2 gives −0.0982, same sign as the
   exact replay. The remaining margin is small enough that a different SIMD
   accumulation order lands on the other side; that pass was luck, not parity.

Earlier step-diff results stay valid but were not causal: the one-ULP embedding
sum and the ten F32 QK budget exceedances are sub-1e−3 effects against a
1.3-logit shift that the replay attributes to the stored weights.

## Fix

- `punct-parity` gates **q8_0 strictly** (all four heads, decoded exact) as the
  ground-truth arm.
- q4_k-imatrix is still gated exactly, except for one recorded loss bound to
  its SHA-256 (`tests/regression/pcs/q4_k-imatrix-recorded-loss.json`, passed
  with `--recorded-quant-loss`): token 37 bits `01` and line 3 `I'm oK, …` may
  appear; any other difference, or a different file, fails.
- No runtime source changed, so no environment gate applies. Changing native
  arithmetic to flip this one logit would hide a weight-precision loss.

## Open (artifact owner)

Publishing a mixed-precision 4-bit PCS artifact (enc.0 and the truecase head at
Q8_0, +0.65 in the what-if) would let the downloaded default match ground
truth on this corpus. That needs a re-quantization, native gate and Hub
publication, and the downstream registry default lives in CrispASR.
