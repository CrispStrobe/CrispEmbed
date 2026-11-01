# Delivery-3 core, StoredOps: LaMa lama-v2 live lane 41292699579 — internal research only

Run https://github.com/CrispStrobe/CrispEmbed/actions/runs/41292699579 at 152a26ef
(`tools/ci-heavy/lama_quant_research.sh live`). Photos larger than the 1024-px profile were
Lanczos-downscaled first (landscape 553×1024, street 1024×682); see `photos.json`.
Strict F32 `subtitle_bar` passes here.

Masked-region (exact subtitle mask) cosine vs the original PyTorch generator on the same input:
F16 1.00000 everywhere; Q8_0 storage ≥ 0.99981, matmul ≥ 0.99964; Q4_K 0.9861 (street) – 0.9990 (face).
Masked PSNR vs the clean photo moves by ≤ 0.09 dB from F32 for every variant. Outside-mask pixels
exact everywhere. Peak RSS on street: ~891 MB storage configs, 790–797 MB with the matmul gate
(F16/Q8_0); wall 45–54 s (4 threads). Full numbers: `live-comparison.json`, `rss-*.txt`.
