# Old-core (delivery 2) LaMa variant lane 41290095498 — internal research only

Run: https://github.com/CrispStrobe/CrispEmbed/actions/runs/41290095498 at a27dc8d5 on
`feat/issue71-lama-research` (`tools/ci-heavy/lama_quant_research.sh`, lama-v1 policy).
**Superseded**: delivery 3 replaced this LaMa core; this is evidence for the old ggml-graph core only
and must be requalified on the new core (see the newer `hosted-*` directories).

Sizes: F32 204,290,336 B; F16 102,545,824; Q8_0 54,849,472; Q4_K 29,413,312 (Q4_K file: 115 Q4_K
3x3 kernels in 256-row storage, 108 Q4_0 1x1 kernels). Hashes in `gguf-sha256.txt`, census in `census-*.json`.

Owner bar (generator-output cosine > 0.94 vs original and vs native F32), five fixtures:

| config | min output cos vs original | vs native F32 | min stage cos | masked RGB PSNR vs original (dB) |
| --- | --- | --- | --- | --- |
| F32 matmul gate | 1.000000 | 1.000000 | 1.00000 | inf ×3, 72.8 (full) |
| F16 storage | 0.999999 | 0.999999 | 0.99994 | 57–67 |
| Q8_0 storage / matmul | 0.99943 / 0.99951 | same | 0.980 | 30–51 |
| Q4_K storage / matmul | 0.97496 / 0.97457 | same | 0.24 (fu.rfft stages) | 14–33 |

All pass the owner bar; none passes the strict F32 stage budgets (not expected). **F16 with the matmul
or lowmem gate failed at graph build** ("non-F32 graph activation": the old core's activation-type
check flagged the F16 weight reshape node). Live (exact mask, 4 threads) masked-region cosine vs the
original PyTorch generator: F16 1.00000, Q8_0 ≥ 0.99968, Q4_K 0.9808–0.9992; masked PSNR vs clean
moves ≤ 0.15 dB (Q4_K) from F32; outside-mask pixels exact everywhere. Peak RSS on street
(1280×853): ~1.2 GB all storage configs, 1.09 GB Q4_K lowmem; wall ~65 s for all configs.
`f32-strict` is the known-red strict F32 protocol (absolute budget on large activations).
