## Quality and status per file (card refresh, 7 November 2026)

| File | Status | Size | Token-ID agreement with the original (8 cases) | Final-logit cosine (min, teacher-forced) | Decoded differences |
|---|---|---:|---|---|---|
| `mineru25-pro-2604-f16.gguf` | **recommended** | 2.59 GB | 8/8 exact | not measured (IDs exact) | none |
| `mineru25-pro-2604-q8_0.gguf` | recommended (smaller) | 1.92 GB | 8/8 exact | not measured (IDs exact) | none |
| `mineru25-pro-2604-q4_k-mix.gguf` (`mineru_q4k_v1`) | **experimental** | 0.77 GB | 6/8 exact; `german` from token 7, `layout-page` from token 52 (CPU) | CPU 0.9673 (`japanese`), Metal 0.9602 (`german`), CUDA 0.9611 (`japanese`) | `german`: masthead read as "Reichs-Gefehbl. 1900. Ausgegeben zu Berlin ..." for "Reichs-Gefehblatt. № 1.", and "Infrafttreten" for "Inkrafttreten"; `layout-page`: one box coordinate moves by 1 (`079` -> `078`) |

Sources: `tests/results/mineru25/precision.json`, `tests/results/qwen2vl-q4k/mineru-q4k-precision.json` and
`docs/qwen2vl-q4k-recipes.md`.

The `layout-page` divergence starts at a different token on each backend: 52 on CPU, 180 on Metal and 48 on CUDA. Q4_K_MIX meets
the Q4_K logit-cosine bar, but its decoded layout output is not identical to the original's. Use F16 or Q8_0 when you need
layout boxes and text to match exactly. CER/WER on a representative corpus is not measured.
