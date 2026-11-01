# Hosted research lane 41287871947 (F32 baseline) — internal research only

Run: https://github.com/CrispStrobe/CrispEmbed/actions/runs/41287871947
(`tools/ci-heavy/lama_research.sh`, source in `provenance.txt`). Only small reports,
manifests and logs are retained here; stage arrays, images, the checkpoint and the GGUF
are not committed.

- Checkpoint `best.ckpt` sha256 `fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423`,
  `config.yaml` sha256 `4fdeed49926e13b101c4dd9e193acec9e58677dfdb4ba49dd6a3a8927964e2a7`
  (from the third-party archive sha256 `f1b358ca…575f6`).
- F32 GGUF sha256 `6369ae4ff76794fe90272f74b8dbffc610987416d49b32990e0295e5b8bf4a01`
  (a local M1 conversion reproduces the same hash).
- All steps pass except `lama-native`, which is **retained as failed** under the unchanged
  strict per-stage budget (max_abs ≤ 0.002). Every exceedance is absolute error only at large
  activation magnitudes (first: `model.5.conv1.ffc.convg2l.output`, max_abs 0.00208 at
  magnitude ~1.6e3, relative L2 4.5e-7); no stage fails relative-L2 or cosine. Final
  `generator.output` max_abs is 1.7e-6–7.5e-5 at cosine 1.0 on all five fixtures; decoded RGB
  is identical on four and differs by at most 1 level on 13 channels of `full`.
- Live exact-mask masked-region PSNR, LaMa F32: face 25.6 dB, landscape 21.2, street 14.4,
  fox 7.3 (fox's flat band makes the naive fill exact). Outside-mask pixels exact everywhere.
  CPU 53–201 s per image (4 vCPU).

This is the F32 reference for the F16/Q8_0/Q4_K comparisons.
