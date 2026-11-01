# Delivery-3 core, StoredOps: LaMa lama-v2 fixtures lane 41292697462 — internal research only

Run https://github.com/CrispStrobe/CrispEmbed/actions/runs/41292697462 at 152a26ef
(`tools/ci-heavy/lama_quant_research.sh fixtures`). Live companion: `../hosted-41292699579-storedops`.

- Strict F32 protocol (`acceptance/`): F32 GGUF sha256 `24736f26…f304c` (204,299,072 B). 7/8 fixtures
  pass the unchanged strict thresholds; `full` fails (retained, thresholds unchanged).
- `f32-stored` (F32 through StoredOps, the A/B control) equals native F32 Engine bit-for-bit in RGB on all 8
  fixtures (masked PSNR vs native F32 = inf).
- Owner bar (prediction cosine > 0.94 vs original and native F32), min over 8 fixtures:
  F16 storage/matmul 1.000000 (pass); Q8_0 storage 0.999767, matmul 0.998136 (pass);
  **Q4_K storage 0.891205, matmul 0.890632 on `full` (FAIL)**; all other Q4_K fixtures ≥ 0.99944.
- Masked RGB PSNR vs original: F16 55.7–68.9 dB, Q8_0 storage 30.5–52.3, Q4_K 16.1–37.7.
