# Precision acceptance policy

Owner decision, 1 November 2026. A converted or quantized model variant is accepted when the
cosine similarity of its **final output** against the original model's output, minimum over every
frozen case of its protocol, exceeds the bar for its precision:

| Precision | Minimum final-output cosine |
|---|---:|
| F32 | > 0.99 |
| F16 | > 0.98 |
| Q8_0 | > 0.97 |
| Q4_K (incl. mixed Q4_K recipes) | > 0.94 |

"Final output" is the model's decisive output tensor: logits at each generated position for
text generators (OCR/VLM), the embedding for encoders, the generated image (and, for inpainting,
the masked region) for image models, and final boxes/scores for detectors. Critical tensors may
stay at a higher precision (F16/F32) to meet a bar; record the recipe.

Always report, beside the accepted cosine:

- the worst intermediate-stage cosine and its stage, so a recipe whose output survives but whose
  internals collapse is visible (prefer recipes that keep internal stages healthy);
- decoded-output agreement: token IDs/text and EOS for generators, region match for detectors,
  PSNR/known-pixel exactness for images;
- the strict absolute stage budgets (e.g. 0.002 / 0.99999). These remain recorded diagnostics:
  a strict failure no longer blocks a variant that clears its bar, and no budget is relaxed or
  deleted.

Earlier decisions made under strict token-ID or absolute-stage exactness stay retained as
evidence. Re-scored under this policy they may change: for example OXR-1.0 Q8_0 with an F16
vision tower clears the Q8_0 bar (worst final-logit cosine 0.9988, worst stage 0.9956) although
the German page diverges at one token, while plain OXR Q4_K does not (table 0.888).
