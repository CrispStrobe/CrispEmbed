## Quality and status per file (card refresh, 7 November 2026)

| File | Status | Size | Token-ID agreement with the original (5 cases) | Final-logit cosine (min, teacher-forced) | Decoded differences |
|---|---|---:|---|---|---|
| `oxr-1.0-f16.gguf` | **recommended** | 4.44 GB | 5/5 exact | not measured (IDs exact) | none |
| `oxr-1.0-q8_0.gguf` (Q8_0-LARGE) | **experimental** | 2.03 GB | 4/5 exact; `german` differs | 0.9977 (`table`); `german` 0.9996 | `german` CER 0.0111 |
| `oxr-1.0-q4_k-head.gguf` (Q4_K-HEAD) | **experimental** | 1.88 GB | 4/5 exact; `german` differs | 0.9980 (`table`); `german` 0.9996 | `german` CER 0.0233 |

Both quantized recipes clear the owner bars on CPU (Q8_0 > 0.97, Q4_K > 0.94). On Metal (M1) and CUDA (T4) they also stay
above 0.998 against the F16 run on each backend. Sources: `tests/results/oxr/precision.json`,
`tests/results/oxr/precision-cosine.json`, `tests/results/oxr/metal-m1/` and `tests/results/oxr/kaggle-cuda-t4-v2/report.json`.

**Known divergence (German page).** On the German case, both quantized files pick a different token from the original at a
single decode step, and the rest of the output follows from that choice:

| Backend | First differing position, Q8_0-LARGE | First differing position, Q4_K-HEAD |
|---|---|---|
| CPU (hosted) | not recorded | not recorded |
| Metal (M1) | 35 | 209 |
| CUDA (T4) | 55 | 209 |

The cause is decode-step activation quantization, not the vision tower:

- Each decode step multiplies a single row with CrispEmbed's integer mat-vec kernel, which quantizes the activations to
  Q8_0 blocks. This kernel gives identical bits on CPU, Metal and CUDA.
- The prefill logits stay above cosine 0.998.
- An opt-in A/B that decodes with F32 activations is still open.

The four other cases match the original token for token. Strict per-stage numerical budgets are not met at any precision.
