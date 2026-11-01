# Big-LaMa F16 / Q8_0 / Q4_K — internal research receipts

**Internal research only.** Big-LaMa checkpoint rights are unresolved: no public upload,
no registry entry or alias, and this line never merges into `main`/`feat/teleocr`. The GGUFs are
preserved only in the owner's private Hugging Face repo `cstr/big-lama-research-GGUF`
(see `private-hf-upload.json`).

## Source

- Archive https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip,
  sha256 `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`; inner `best.ckpt` sha256
  `fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423`.
- Upstream code: advimman/lama `786f5936b27fb3dacd2b1ad799e4de968ea697e7`, Apache-2.0
  (`docs/third-party/lama-LICENSE.txt`); that licence covers the code, not the weights.
- Converter: delivery-3 `models/convert-lama-to-gguf.py`. Quantizer: `tools/quantize.cpp` policy **lama-v2**.

## Files (delivery-3 core; hashes reproduced in lanes 41295796193, 41309379208, 41309381796)

| file | bytes | sha256 | tensor types | owner bar |
| --- | --- | --- | --- | --- |
| big-lama-f32.gguf | 204,299,072 | `24736f267c68e013f0000ccb69560c4353b6a6d7983fc3cc2537d9f2e03f304c` | 837 F32 | reference |
| big-lama-f16.gguf | 102,554,560 | `31d5a0c4dae9a0d638a5b75bb8921ea7e588e35c8c00060e4ce16fc0bc0d3f3b` | 223 F16 kernels, 614 F32 | pass |
| big-lama-q8_0.gguf | 54,858,208 | `c01175e15445494220480fe2a5ceb4f6d34cffa293675892b5217bfff34c1f42` | 223 Q8_0 | pass |
| big-lama-q4_k-s8e8.gguf | 35,504,608 | `ce64f5895fb8da51761e3a79aefc7aac1e2860f310fe01185928c0b299b44803` | 111 Q4_K, 112 Q8_0 (global spectral branch + encoder model.2–4) | pass |
| failed-owner-bar/big-lama-q4_k.gguf | 29,422,048 | `881b99ff035411d18d0d2b0790394fbbd64a79ea2249f18c3b15808a49a4c039` | 223 Q4_K | **FAIL** (full-mask fixture) |

Vectors (bias/BatchNorm, 614 tensors) and the first and last 7×7 convs are always F32. Q4_K bytes
are platform-dependent: an Apple-M1 quantization of the same recipes gives different Q4_K hashes
(F16/Q8_0 match), so only the x86-64 hosted bytes above were tested and preserved.

## Runtime paths (all default-off; the F32 Engine default is unchanged)

- **storage** (default for a lama-v2 file): every non-F32 kernel dequantized to F32 at load.
- **matmul** `CRISPEMBED_LAMA_QUANT_MATMUL=1`: patch-row kernels stay in their stored type as the
  `ggml_mul_mat` src0 (eager StoredOps, or the graph via `inpaint_graph::StoragePolicy`).
- **opt**: delivery 5's optimized CPU graph (`CRISPEMBED_INPAINT_GRAPH=1 CRISPEMBED_INPAINT_CPU_FFT=1
  CRISPEMBED_INPAINT_REFLECT_GATHER=1 CRISPEMBED_INPAINT_GEMM_LAYOUT=1`), optionally with matmul.

## Results on the optimized graph (lanes 41309379208 fixtures, 41309381796 live; 4-vCPU x86-64)

Eight original-capture fixtures, prediction (generator output) cosine; owner bar > 0.94 vs the
original and vs native F32:

| variant | opt: min cos vs original / F32 | opt-matmul: min cos | min stage cos | masked RGB PSNR vs original |
| --- | --- | --- | --- | --- |
| F32 | 1.000000 / 1.000000 | – | 1.0000 | 69.7–85.2 dB |
| F16 | 1.000000 / 1.000000 | 1.000000 | 1.0000 | 55.7–69.0 dB |
| Q8_0 | 0.999767 / 0.999767 | 0.998105 | 0.9716 (0.889 matmul) | 30.5–52.2 dB |
| Q4_K-s8e8 | 0.998058 / 0.998058 | 0.997813 | 0.8232 | 28.0–49.9 dB |
| Q4_K (plain) | **0.891205** / 0.891206 (`full`) | 0.890633 | 0.4805 | 16.1–37.7 dB |

Plain Q4_K passes the other seven fixtures (≥ 0.999443); only the all-fill `full` fixture fails.
Every config keeps known RGB exact. No variant meets the strict F32 stage thresholds (not expected);
the strict F32 protocol itself keeps its `full` failure (unchanged thresholds, see delivery 5).

Live: burned-in subtitles on four real photos (exact mask), masked-region cosine / PSNR vs the
original PyTorch generator; masked PSNR vs the clean photo in brackets (original: face 25.76,
fox 7.35, landscape 24.00, street 14.60 dB):

| variant (opt) | face | fox | landscape 553×1024 | street 1024×682 | peak RSS street | wall street |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | 1.00000 / 90.2 [25.76] | 1.00000 / 91.8 [7.35] | 1.00000 / 84.5 [24.00] | 1.00000 / 85.5 [14.60] | 956 MiB | 33.4 s |
| F16 | 1.00000 / 69.7 [25.76] | 1.00000 / 63.4 [7.35] | 1.00000 / 65.3 [24.00] | 1.00000 / 63.1 [14.60] | 960 MiB (866 matmul) | 32.9 s (43.6) |
| Q8_0 | 0.99999 / 52.4 [25.81] | 0.99995 / 41.9 [7.33] | 0.99997 / 48.5 [24.00] | 0.99981 / 43.1 [14.61] | 960 MiB (831 matmul) | 32.7 s (29.1) |
| Q4_K-s8e8 | 0.99963 / 36.3 [26.22] | 0.99829 / 26.3 [7.56] | 0.99924 / 35.0 [23.96] | 0.99331 / 27.3 [14.69] | 959 MiB (934 matmul) | 32.8 s (33.0) |
| Q4_K | 0.99897 / 31.8 [25.88] | 0.99650 / 23.3 [7.28] | 0.99839 / 31.2 [23.91] | 0.98610 / 24.1 [14.64] | 960 MiB | 32.8 s |

Outside-mask pixels are exact in every run. Activations dominate RSS, so quantized storage saves
disk (204 → 103 / 55 / 36 MB) far more than memory; the matmul gate saves ~90–130 MiB for F16/Q8_0.

## Directories

- `hosted-41287871947/` delivery-2 F32 baseline; `hosted-41290095498-old-core/` delivery-2 variants (superseded).
- `hosted-41292697462-storedops/`, `hosted-41292699579-storedops/` delivery-3 eager StoredOps.
- `hosted-41295796193-graph/`, `hosted-41295798015-graph/` portable graph + StoredOps, Q4_K recipes.
- `hosted-41309379208-optimized/`, `hosted-41309381796-optimized/` delivery-5 optimized graph (above).
