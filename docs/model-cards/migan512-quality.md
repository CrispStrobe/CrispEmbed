## Status per file (card refresh, 7 November 2026)

| File | Status | Min cosine, synthetic panel vs original | Min masked-region cosine, real photos vs native F32 |
|---|---|---|---|
| `migan-512-places2-f32.gguf` | reference | 1 | reference |
| `migan-512-places2-f16.gguf` | **normal / recommended** | 0.999998 | not separately listed here; see the receipts |
| `migan-512-places2-q8_0.gguf` | caveat | 0.9965 | not separately listed here; see the receipts |
| `migan-512-places2-q4_k.gguf` | **experimental** | 0.9770 | **0.9019** on the street crop with a DBNet text mask (below the 0.94 bar); exact-mask photos >= 0.962 |

Use F16 for normal work. Q4_K is experimental. It clears the bar on the synthetic panel, but its masked-region cosine falls to
0.9019 on one realistic text-removal crop. Sources: `tests/results/migan512/` and `docs/inpainting-precision.md`.
Representative subtitle and video quality is not measured.
