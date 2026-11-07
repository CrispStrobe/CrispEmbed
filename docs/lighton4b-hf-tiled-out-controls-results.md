# Original-F16 output-coordinate helper controls

Kaggle CPU worker 143168048/v1 passed in 76.18 seconds at source `617534eff56c7e0ea98781b783d3d118f85e1c52`. The helper passed 19 synthetic checks, then repeated all 19 using its retained executable/library bundle after the build tree was hidden. Five Python protocol and output-protection tests passed. No checkpoint was downloaded and no model inference ran.

The fixture uses 69,632 original F16 elements with nonuniform importance statistics. Both HF-order and tiled-order weighted Q4 payloads are invariant across one-, three- and sixteen-row chunks. An independently constructed tiled matrix reproduces the helper's tiled payload; incorrectly inverted statistics change the payload. Independent Python dequantization verifies all three value hashes and canonical-coordinate weight-error measurements. Malformed source/statistics, source identity changes and existing outputs fail closed.

The [raw receipts and logs](../tests/results/qwen35-ocr/lightonocr3-4b-hf-tiled-out/controls-v1/evidence.json) preserve every downloaded small artifact and its hash. Helper SHA256 is `c1bd905abfdb65ca4bcc5dc3d23cc41b312e93877b1dfbed67b2eb29c981bae1`; runtime manifest SHA256 is `b81c00a330f1f4fa030b5509e0ce533da66e89142f2538774dbf4a5404bcad8d`.

This proves bounded helper behavior, not original-tensor identity or decoded quality. The separate original-F16 24-matrix proof must reproduce both the retained HF candidate payloads and publisher-coordinate payloads before any distinct HF-tiled-output recipe is prepared. Existing form negatives and physical-profile scalar conventions remain separate.
