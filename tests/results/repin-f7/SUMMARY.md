# Reranker sub-Q8 re-pin — local Metal+CPU cross-check + decisions (2026-08-05, round 7)

The two OPEN decisions from the reranker imatrix re-collection (`87e11a4e`):
the Kaggle rerank-f7 A/B was x86-CPU-only, so per the G3 precedent the `-f7`
artifacts were cross-checked locally on Metal AND CPU before touching any pin.

## Method

`local_ab.py` (this directory) — imports `RERANK_EVAL` (30 pairs × 6 docs) and
`kendall_tau` directly from the canonical pipeline
(`tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py`), so the metric is
the Kaggle kernel's own code, not a reimplementation. Gold = the f16
artifact's scores on the SAME backend as the arm under test. Every run is a
separate process with `--gpu-backend` explicit; Metal arms assert `MTL0` in
their own stderr (the runner raises otherwise). Worktree binary by absolute
path, `GGML_METAL:BOOL=ON` in `build/CMakeCache.txt`. Runs serialized. Raw
per-arm data: `results.json`.

Downloaded arms SHA-checked before measuring: the "old" q4_k-imatrix files
match the shipped pins exactly (jina `c04e38c8…` = `model_hashes.h` pin;
q8_0 `424b590a…` likewise), so the comparison is against the genuinely
pinned artifacts.

## Comparability with the Kaggle record (`*-f7-imatrix-ab.txt` on HF)

- jina f16 raw scores, eval pair 0, local CPU: `[0] -1.024 [1] -0.239
  [2] +0.039 [3] -0.757 [4] -3.624 [5] -3.674` — matches the Kaggle
  full-precision row to ~3 decimals.
- bge f16 pair 0 local CPU: `[0] +0.899 [1] -0.119 [2] +1.638 [3] -0.334
  [4] -11.036 [5] -11.033` — same.
- jina q4_k-im-f7 dscore: 0.0796 local CPU vs 0.0792 Kaggle (4dp-close);
  bge q4_k-im-f7: 0.2268 local vs 0.2245 Kaggle.
- tau differs from Kaggle by ~0.009-0.013 per arm (2-3 pairwise near-tie
  flips out of 450); the q8_0 arm itself swings 0.009 between local CPU and
  Metal, so that is the cross-build/backend noise band for this metric.

## Results (mean Kendall-tau / mean|Δscore| vs same-backend f16 gold, 30 pairs)

### jina-reranker-v2-base-multilingual

| arm | CPU tau | CPU dscore | Metal tau | Metal dscore |
|---|--:|--:|--:|--:|
| q8_0 | 0.9956 | 0.0156 | 0.9867 | 0.0096 |
| q4_k-imatrix (old pin, leaf_N imatrix) | 0.9333 | 0.1060 | 0.9422 | 0.0985 |
| **q4_k-imatrix-f7** | 0.9333 | **0.0796** | 0.9333 | **0.0729** |
| iq4_xs-f7 | 0.9511 | 0.0940 | 0.9644 | 0.0840 |

### bge-reranker-v2-m3

| arm | CPU tau | CPU dscore | Metal tau | Metal dscore |
|---|--:|--:|--:|--:|
| q8_0 | 0.9778 | 0.0410 | 0.9822 | 0.0267 |
| q4_k-imatrix (old, leaf_N imatrix; never aliased) | 0.9200 | 0.3208 | 0.9200 | 0.2986 |
| **q4_k-imatrix-f7** | **0.9422** | **0.2268** | **0.9467** | **0.2013** |
| iq4_xs-f7 | 0.9378 | 0.2415 | 0.9289 | 0.2055 |

## Decisions (coordinator)

1. **jina `-q4k` alias RE-PINNED** to `jina-reranker-v2-base-multilingual-
   q4_k-imatrix-f7.gguf` (`7fe1439b…`). dscore improves ~25% on BOTH
   backends (0.1060→0.0796 CPU, 0.0985→0.0729 Metal); tau equal on CPU and
   −0.009 on Metal — inside the near-tie band above, and dscore is the
   continuous metric that imatrix quality is visible in
   (thresholded/rank metrics saturate; established F7 discipline).
   jina's iq4_xs-f7 posts the best tau of the 4-bit arms; recorded here,
   no alias exists for it and none is added (no demand signal).
2. **bge-reranker-v2-m3: first sub-Q8 alias ADDED** —
   `bge-reranker-v2-m3-q4k` → `bge-reranker-v2-m3-q4_k-imatrix-f7.gguf`
   (`21eadb97…`). The handover asked to "re-pin the sub-Q8 aliases"; none
   existed, so the decision became whether to add one: yes — q4_k-im-f7 wins
   tau AND dscore over the never-aliased old imatrix quant on both backends
   (tau .920→.942/.947), is 25% smaller than q8_0 (462 vs 613 MB), and the
   family pattern (jina/ms-marco/arctic) ships a sub-Q8 alias. q4_k-im-f7
   chosen over iq4_xs-f7 (better tau on both backends).
3. **q8_0 stays the default tier for both families** (tau .978-.996).

## Acceptance (rebuilt binary, `Building CXX object …model_mgr.cpp.o` seen)

Fresh-download spot-runs via `CRISPEMBED_CACHE_DIR=<empty dir>` on the
rebuilt worktree binary, `--gpu-backend metal`, MTL0 in each run's stderr:

- `-m jina-reranker-v2-base-multilingual-q4k --accept-license cc-by-nc-4.0`:
  downloads `…-q4_k-imatrix-f7.gguf`, `Verifying SHA-256...` passes, RC=0,
  scores HF-scale with the f16 ordering (ibuprofen −0.196 > triptans −1.079
  > bridge −3.629; f16: −0.239 > −1.024 > −3.624).
- `-m bge-reranker-v2-m3-q4k`: downloads `…-q4_k-imatrix-f7.gguf`,
  SHA-256 passes, RC=0, scores HF-scale, correct top-1 (triptans +0.540 >
  ibuprofen −0.422 > bridge −11.037).

No published HF file was replaced; old pin rows kept in `model_hashes.h`
(mxbai `-g7c` precedent).
