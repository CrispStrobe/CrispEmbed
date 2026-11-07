## Quality and status per file (card refresh, 7 November 2026)

| File | Status | Size | Token-ID agreement with the original (6 cases) | Final-logit cosine (min, teacher-forced) | Decoded differences |
|---|---|---:|---|---|---|
| `teleocr-f16.gguf` | **recommended** (experimental model) | 2.53 GB | 6/6 exact | not measured (IDs exact) | none |
| `teleocr-q8_0.gguf` | caveat (experimental model) | 2.11 GB | 6/6 exact | not measured (IDs exact) | none |
| `teleocr-q4_k-m.gguf` | caveat (experimental model) | 1.92 GB | 6/6 exact | not measured (IDs exact) | none |
| `teleocr-q4_k-mix.gguf` (`teleocr_q4k_v1`) | **experimental** | 1.20 GB | 5/6 exact; `table` differs from token 35 | CPU 0.9677, Metal 0.9677, CUDA 0.9570 (all `table`) | `table`: three CJK characters change (`指标`->`指`, `对位`->`对住`, `外观`->`外现`) |

Sources: `tests/results/teleocr/hosted-precision.json`, `tests/results/qwen2vl-q4k/teleocr-q4k-precision.json` and
`docs/qwen2vl-q4k-recipes.md`.

- **Strict numerics are false at every precision, F32 included.** The strict per-stage numerical qualification fails:
  174 stage passes against 276 failures, and no case passes strictly (`tests/results/teleocr/summary.json`). The decoded IDs of F32, F16, Q8_0 and Q4_K_M nevertheless match the original exactly.
- Not measured: CER/WER on a representative corpus. The six frozen cases are a smoke panel, not a quality benchmark.
- TeleOCR itself remains experimental and default-off (`CRISPEMBED_TELEOCR`).
