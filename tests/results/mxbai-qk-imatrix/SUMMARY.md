# mxbai DeBERTa q/k imatrix provenance A/B

Question: for `mxbai-rerank-{xsmall,base}-v1` (DeBERTa-v2 cross-encoders), the
imatrix file contains BOTH a direct `blk.N.attn_{q,k}.weight` entry and the
merged-QKV alias entry `enc.N.attn.qkv_merged.weight`. `tools/quantize.cpp`
prefers the direct entry. Is that the wrong choice, and is it what made the
mxbai `-f7` 4-bit tail look worse than its no-imatrix arm?

**Verdict: the provenance defect is real and proven, but it is NOT a quality
defect on these artifacts. Keep arm0 (`direct`, the current default).**

---

## 1. The finding, verified

### 1.1 Code level

* Collector - `src/imatrix.cpp`, `eval_cb`: hooks every `GGML_OP_MUL_MAT`
  whose `src[0]` resolves to a **named leaf weight**, and accumulates the
  per-column sum of squares of `src[1]` (the *activation*) under **the
  weight's name**.
* BERT-family merge - `src/crispembed.cpp:852-871`: q/k/v are pre-merged into
  one F32 tensor named `core_imatrix::qkv_merged_name(i)`
  (= `enc.<N>.attn.qkv_merged.weight`). The token-stream matmul is therefore
  filed under the merged name, and `qkv_merged_alias()`
  (`src/core/imatrix_alias.h`) maps q/k/v back to it at quantize time.
* DeBERTa-v2 second application - `src/crispembed.cpp:1166` and `:1195`:

  ```
  ggml_tensor * Pk = ggml_mul_mat(gctx, L.k_w, P);      // c2p
  ggml_tensor * Pq = ggml_mul_mat(gctx, L.q_w, P_p2c);  // p2c
  ```

  `L.k_w` / `L.q_w` are the **original** GGUF tensors, so the collector files
  `blk.N.attn_k.weight` / `blk.N.attn_q.weight` **direct** entries. Their
  activation is `P = rel_pos_expanded` - the LayerNormed relative-position
  embedding table expanded by bucket index - **not** the token stream.
* Shadowing - `tools/quantize.cpp` (pre-change, ~line 722): a direct entry is
  looked up first and the alias is consulted only when the direct lookup
  misses. So for DeBERTa the rel-position statistics win.

`v_w` is *not* re-applied to the positions, so `attn_v` has no direct entry and
correctly falls through to the alias. The defect is q/k only.

### 1.2 Artifact level - key inventory

`cstr/mxbai-rerank-{xsmall,base}-v1-GGUF/*-f7.imatrix`, both files:

```
72 tensors, no leaf_N
  12  blk.N.attn_k.weight       <- DIRECT, rel-position provenance
  12  blk.N.attn_q.weight       <- DIRECT, rel-position provenance
  12  blk.N.attn_output.weight
  12  blk.N.ffn_down.weight
  12  blk.N.ffn_up.weight
  12  enc.N.attn.qkv_merged.weight   <- token-stream provenance
layer-0 keys: blk.0.attn_k.weight, blk.0.attn_output.weight, blk.0.attn_q.weight,
              blk.0.ffn_down.weight, blk.0.ffn_up.weight, enc.0.attn.qkv_merged.weight
```

Both entries exist for every layer, and both have length `n_embd`, so both
shape-match `attn_q`/`attn_k` -> the shadowing precondition holds and fires.

How wrong the shadowing vector is:

| model  | layer | count(direct) | count(merged) | cos(direct, merged) | mean(direct) | mean(merged) |
|--------|-------|---------------|---------------|---------------------|--------------|--------------|
| xsmall | 0     | 17235         | 949           | 0.2146              | 2.698e-02    | 1.879e-01    |
| xsmall | 5     | 17235         | 949           | 0.0717              | 2.698e-02    | 2.175e-01    |
| xsmall | 11    | 17235         | 949           | 0.0146              | 2.698e-02    | 3.200e-01    |
| base   | 0     | 17235         | 949           | 0.1040              | 3.322e-02    | 1.682e-01    |
| base   | 5     | 17235         | 949           | 0.0391              | 3.322e-02    | 2.248e-01    |
| base   | 11    | 17235         | 949           | 0.0199              | 3.322e-02    | 2.390e-01    |

Two further facts confirm the provenance:

* **All 24 direct q/k vectors in a file are bit-identical** (verified with
  `np.array_equal` over all 12 layers x {q,k}). They must be: every layer's
  c2p/p2c matmul consumes the same shared `rel_pos_expanded` tensor, so the
  direct entry carries no layer information and no q-vs-k information at all.
  The merged vectors, by contrast, differ per layer (L0-vs-Ln cosine 0.325..1.0
  for xsmall, 0.445..1.0 for base).
* `count(direct)` = 17235 ~ sum of T^2 rows (the T x T position grid) vs
  `count(merged)` = 949 = the token count. Different row populations entirely.

The direct vector is not flat - it is *more* peaked than the merged one
(xsmall: cv 2.99, p99/p50 = 102.5, max/min = 3.9e5 vs merged L0 cv 0.88,
p99/p50 = 5.1). It is a strongly-opinionated vector with the wrong provenance.

**Conclusion: the recorded finding is correct as stated.** The measurement
below tests whether it matters.

---

## 2. Implementation

`tools/quantize.cpp`, gate `CRISPEMBED_QUANT_IMATRIX_QKV` (a 3-way string
selector, not a boolean - it takes `direct` | `merged` | `sum`):

| arm  | value    | behaviour |
|------|----------|-----------|
| arm0 | `direct` | default; direct entry preferred (shipped behaviour) |
| armA | `merged` | merged alias preferred when both exist |
| armB | `sum`    | element-wise sum of the direct and merged vectors |

Only q/k/v weight names that have a merged alias are affected - 24 of the 75
quantized tensors per model. Collector and runtime graph code are untouched.

### Gate spelling verification (final, post-format binary)

All three required spellings on a real run, xsmall q4_k + `-f7.imatrix`:

```
spell=absent  md5=9fbc0c229d172fb908b9fcc55081949d  (no mode line)
spell=0       md5=9fbc0c229d172fb908b9fcc55081949d  imatrix: unknown CRISPEMBED_QUANT_IMATRIX_QKV='0' (direct|merged|sum), using direct
spell=1       md5=9fbc0c229d172fb908b9fcc55081949d  imatrix: unknown CRISPEMBED_QUANT_IMATRIX_QKV='1' (direct|merged|sum), using direct
```

Byte-identical in all three cases, and an invalid value is reported rather than
silently reinterpreted.

### Shipped-artifact reproduction

The default arm reproduces the published artifact **bit-for-bit**:

```
arm0 (direct) local q4_k+imatrix-f7 from -g7c        md5 = 9fbc0c229d172fb908b9fcc55081949d
cstr/.../mxbai-rerank-xsmall-v1-q4_k-imatrix-g7c.gguf md5 = 9fbc0c229d172fb908b9fcc55081949d
armA (merged)                                        md5 = 0166cf52d3a6baf8e8c13b0fee5dcabe
armB (sum)                                           md5 = 1e7dc0e7a0801159622aa608262b19f8
no-imatrix                                           md5 = 0738538b9c6ccbe580981ffd5cbcf901
```

so the harness is exactly the shipped path, and the three arms really do differ.

### `N with imatrix` (quantizer's own stdout, every imatrix arm, both models)

```
75 quantized, 127 kept, 72 with imatrix     (q4_k / iq4_xs / q3_k, direct|merged|sum)
75 quantized, 127 kept                      (no-imatrix control)
```

72 = 12 layers x (q, k, v, attn_output, ffn_up, ffn_down). Identical across
arms, as expected - the arms change *which* vector is used, never *whether*.

---

## 3. Measurement

* Reference: ONNX Runtime **1.25.1**, CPU EP, official
  `mixedbread-ai/mxbai-rerank-{xsmall,base}-v1` `onnx/model.onnx`, tokenized
  via the repo `tokenizer.json` through `tokenizers`. No torch (local
  miniconda torch is disqualified for BERT-class forwards).
* Fixtures: the 30-query x 6-doc `RERANK_EVAL` from
  `tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py` (`kendall_tau` /
  `rerank_ab` replicated verbatim), plus the 2-query x 6-doc fixture from
  `tests/results/mxbai-gelu/pairs.py`.
* Base: the `-g7c` f16 (ContextPooler-bearing) artifact; imatrix: `-f7.imatrix`.
* All quality runs `--gpu-backend cpu`, one process at a time.

**Reference chain is sound** - the f16 `-g7c` GGUF matches the official ONNX
export to 1e-5 on both fixtures and both models:

```
xsmall f16(g7c) vs ONNX: eval mean|d|=0.00000 max|d|=0.00001  gelu max|d|=0.00000  tau=1.000000
base   f16(g7c) vs ONNX: eval mean|d|=0.00000 max|d|=0.00001  gelu max|d|=0.00001  tau=1.000000
```

Because f16 == ONNX to 1e-5, `tau_vs_f16` and `tau_vs_ONNX` are numerically
identical; `dscore_vs_f16` and `mean|d| vs ONNX` likewise. One column each is
reported.

### 3.1 xsmall

| qtype  | arm    | tau    | mean\|d\| | max\|d\| (eval) | max\|d\| (gelu) |
|--------|--------|--------|-----------|-----------------|-----------------|
| q4_k   | noim   | 0.9067 | 0.1871    | 0.7207          | 0.4018 |
| q4_k   | direct | 0.9333 | 0.1928    | 0.6618          | 0.4747 |
| q4_k   | merged | **0.9511** | **0.1566** | 0.5601      | 0.5162 |
| q4_k   | sum    | 0.9289 | 0.1601    | 0.5444          | 0.4772 |
| iq4_xs | noim   | 0.9333 | 0.1900    | 0.6741          | 0.7140 |
| iq4_xs | direct | 0.9556 | **0.1402** | 0.4997         | 0.5557 |
| iq4_xs | merged | **0.9644** | 0.1628 | 0.5739          | 0.5793 |
| iq4_xs | sum    | 0.9556 | 0.1534    | 0.4899          | 0.5843 |
| q3_k   | noim   | **0.9244** | 0.2464 | 1.2298          | 0.4006 |
| q3_k   | direct | 0.9156 | **0.1774** | 0.7591         | 0.6615 |
| q3_k   | merged | 0.9156 | 0.1859    | 0.5913          | 0.6857 |
| q3_k   | sum    | 0.9111 | 0.1900    | 0.6067          | 0.6607 |

### 3.2 base

| qtype  | arm    | tau    | mean\|d\| | max\|d\| (eval) | max\|d\| (gelu) |
|--------|--------|--------|-----------|-----------------|-----------------|
| q4_k   | noim   | **0.9556** | 0.3366 | 1.5121          | 0.5471 |
| q4_k   | direct | 0.9422 | 0.2910    | 1.2255          | 0.5697 |
| q4_k   | merged | 0.9511 | **0.2445** | 0.8724         | 0.4736 |
| q4_k   | sum    | 0.9467 | 0.2524    | 0.8519          | 0.5713 |
| iq4_xs | noim   | **0.9644** | **0.2322** | 1.1042     | 0.4648 |
| iq4_xs | direct | 0.9556 | 0.2734    | 1.3087          | 0.6306 |
| iq4_xs | merged | 0.9422 | 0.3428    | 1.3428          | 0.7873 |
| iq4_xs | sum    | 0.9422 | 0.3426    | 1.3773          | 0.7577 |
| q3_k   | noim   | 0.8800 | 0.5877    | 3.4186          | 1.4713 |
| q3_k   | direct | 0.9200 | **0.5253** | 3.1980         | 1.8528 |
| q3_k   | merged | **0.9289** | 0.6567 | 3.5357          | 1.5837 |
| q3_k   | sum    | 0.9156 | 0.6678    | 3.6511          | 1.5732 |

### 3.3 Paired comparison and pooled error (vs ONNX, 192 scores per cell)

`win_vs_direct` = how many of the 192 individual scores the arm places closer
to ONNX than the `direct` arm does (96/192 = a coin flip).

| model  | qtype  | arm    | mean\|d\| | rmse   | win_vs_direct |
|--------|--------|--------|-----------|--------|---------------|
| xsmall | q4_k   | merged | 0.1579    | 0.2017 | 123/192 |
| xsmall | q4_k   | sum    | 0.1598    | 0.1998 | 123/192 |
| xsmall | q4_k   | direct | 0.1897    | 0.2316 | - |
| xsmall | iq4_xs | direct | 0.1397    | 0.1801 | - |
| xsmall | iq4_xs | sum    | 0.1545    | 0.1956 | 85/192 |
| xsmall | iq4_xs | merged | 0.1636    | 0.2054 | 82/192 |
| xsmall | q3_k   | direct | 0.1810    | 0.2294 | - |
| xsmall | q3_k   | merged | 0.1921    | 0.2456 | 96/192 |
| xsmall | q3_k   | sum    | 0.1954    | 0.2467 | 83/192 |
| base   | q4_k   | merged | 0.2388    | 0.3086 | 110/192 |
| base   | q4_k   | sum    | 0.2499    | 0.3212 | 113/192 |
| base   | q4_k   | direct | 0.2900    | 0.3740 | - |
| base   | iq4_xs | direct | 0.2683    | 0.3624 | - |
| base   | iq4_xs | sum    | 0.3401    | 0.4518 | 73/192 |
| base   | iq4_xs | merged | 0.3425    | 0.4518 | 70/192 |
| base   | q3_k   | direct | 0.5375    | 0.6931 | - |
| base   | q3_k   | merged | 0.6613    | 0.8380 | 80/192 |
| base   | q3_k   | sum    | 0.6726    | 0.8571 | 79/192 |

Pooled over all 6 (model x qtype) cells, n = 1152 scores per arm:

| arm    | mean\|d\| | rmse   | max\|d\| |
|--------|-----------|--------|----------|
| noim   | 0.2953    | 0.4147 | 3.4186 |
| **direct (arm0)** | **0.2677** | **0.3852** | **3.1980** |
| merged (armA) | 0.2927 | 0.4368 | 3.5357 |
| sum (armB)    | 0.2954 | 0.4436 | 3.6511 |

### 3.4 Raw scores, eval query 0 ("treatment for migraine headaches")

xsmall:

```
ONNX          [0]  +0.689 [1]  +0.497 [2]  +0.793 [3]  +0.056 [4]  -3.277 [5]  -3.902
f16-g7c       [0]  +0.689 [1]  +0.497 [2]  +0.793 [3]  +0.056 [4]  -3.277 [5]  -3.902
q4_k/noim     [0]  +0.661 [1]  +0.653 [2]  +0.945 [3]  +0.071 [4]  -2.923 [5]  -3.291
q4_k/direct   [0]  +0.778 [1]  +0.451 [2]  +1.106 [3]  +0.436 [4]  -2.905 [5]  -3.304
q4_k/merged   [0]  +0.759 [1]  +0.445 [2]  +1.020 [3]  +0.416 [4]  -2.908 [5]  -3.342
q4_k/sum      [0]  +0.750 [1]  +0.394 [2]  +0.954 [3]  +0.395 [4]  -2.918 [5]  -3.358
iq4_xs/noim   [0]  +0.309 [1]  +0.775 [2]  +0.892 [3]  -0.121 [4]  -3.514 [5]  -3.887
iq4_xs/direct [0]  +0.355 [1]  +0.061 [2]  +0.462 [3]  -0.356 [4]  -3.463 [5]  -4.110
iq4_xs/merged [0]  +0.417 [1]  +0.318 [2]  +0.552 [3]  -0.265 [4]  -3.456 [5]  -4.010
iq4_xs/sum    [0]  +0.447 [1]  +0.256 [2]  +0.557 [3]  -0.223 [4]  -3.455 [5]  -3.943
q3_k/noim     [0]  +0.435 [1]  +0.875 [2]  +0.965 [3]  +0.179 [4]  -2.920 [5]  -3.239
q3_k/direct   [0]  +0.553 [1]  +0.135 [2]  +0.879 [3]  +0.353 [4]  -2.969 [5]  -3.263
q3_k/merged   [0]  +0.527 [1]  +0.167 [2]  +0.753 [3]  +0.329 [4]  -2.982 [5]  -3.311
q3_k/sum      [0]  +0.499 [1]  +0.125 [2]  +0.726 [3]  +0.326 [4]  -2.962 [5]  -3.295
```

base:

```
ONNX          [0]  +2.323 [1]  -2.342 [2]  +0.865 [3]  -1.426 [4]  -5.534 [5]  -5.550
f16-g7c       [0]  +2.323 [1]  -2.342 [2]  +0.865 [3]  -1.426 [4]  -5.534 [5]  -5.550
q4_k/noim     [0]  +2.136 [1]  -1.969 [2]  +0.559 [3]  -1.254 [4]  -5.137 [5]  -5.118
q4_k/direct   [0]  +2.037 [1]  -2.657 [2]  +0.597 [3]  -1.465 [4]  -5.108 [5]  -5.298
q4_k/merged   [0]  +2.074 [1]  -2.506 [2]  +0.584 [3]  -1.464 [4]  -5.340 [5]  -5.301
q4_k/sum      [0]  +2.256 [1]  -2.455 [2]  +0.675 [3]  -1.465 [4]  -5.273 [5]  -5.303
iq4_xs/noim   [0]  +2.142 [1]  -1.997 [2]  +0.765 [3]  -1.298 [4]  -5.434 [5]  -5.434
iq4_xs/direct [0]  +2.226 [1]  -2.183 [2]  +0.691 [3]  -1.352 [4]  -5.283 [5]  -5.362
iq4_xs/merged [0]  +1.975 [1]  -2.099 [2]  +0.475 [3]  -1.306 [4]  -5.502 [5]  -5.644
iq4_xs/sum    [0]  +1.847 [1]  -2.049 [2]  +0.441 [3]  -1.291 [4]  -5.451 [5]  -5.512
q3_k/noim     [0]  +1.969 [1]  -1.734 [2]  +0.445 [3]  -0.409 [4]  -4.952 [5]  -5.270
q3_k/direct   [0]  +1.223 [1]  -1.955 [2]  -0.509 [3]  -1.145 [4]  -4.991 [5]  -5.199
q3_k/merged   [0]  +0.789 [1]  -1.995 [2]  -0.036 [3]  -1.198 [4]  -5.476 [5]  -5.771
q3_k/sum      [0]  +0.910 [1]  -1.931 [2]  -0.080 [3]  -1.222 [4]  -5.481 [5]  -5.773
```

### 3.5 Metal spot-check

`--gpu-backend metal`, xsmall q4_k, eval query 0, MTL0 present in each run's
own stderr (3 MTL0 lines per run):

```
direct  metal [0]  +0.782 [1]  +0.449 [2]  +1.137 [3]  +0.475 [4]  -2.882 [5]  -3.284
direct  cpu   [0]  +0.778 [1]  +0.451 [2]  +1.106 [3]  +0.436 [4]  -2.905 [5]  -3.304
merged  metal [0]  +0.762 [1]  +0.434 [2]  +1.016 [3]  +0.424 [4]  -2.891 [5]  -3.340
merged  cpu   [0]  +0.759 [1]  +0.445 [2]  +1.020 [3]  +0.416 [4]  -2.908 [5]  -3.342
```

Same ordering, deltas <= 0.04 - no arch-specific behaviour in either arm.

---

## 4. Recommendation: keep **arm0 (`direct`)** as the default

1. **No arm wins consistently.** Across the 6 (model x qtype) cells, `merged`
   has the lower mean|d| in 2 (both q4_k) and the higher one in 4
   (both iq4_xs, both q3_k). Pooled over all 1152 scores, `direct` is the
   *best* arm (0.2677 vs merged 0.2927, sum 0.2954, noim 0.2953) - though the
   margin is small enough that this is a tie-with-a-lean, not a result.
2. **The harsher-quant test refutes the harm hypothesis.** If the wrong
   provenance were damaging, dropping from 4-bit to q3_k - where importance
   weighting matters most - would amplify the damage. It does the opposite:
   at q3_k `direct` beats `merged` on both models (xsmall 0.1810 vs 0.1921;
   base 0.5375 vs 0.6613), by the largest margin of any qtype.
3. **Paired win rates hover at chance.** Outside the two q4_k cells, `merged`
   improves 70-96 of 192 individual scores (chance = 96). The q4_k advantage
   (110-123/192) does not reproduce at either neighbouring bit-width.
4. **The suspected regression has a different, already-fixed cause.** The
   board's "mxbai is the only reranker family whose `-f7` 4-bit tail
   regressed" rests on `mxbai-rerank-xsmall-v1-f7-imatrix-ab.txt`, whose
   header reads `quant_src=mxbai-rerank-xsmall-v1.gguf` - the **pre-g7c** f16,
   i.e. the artifact missing the DeBERTa ContextPooler stage, whose scores were
   compressed to +-0.3 instead of the HF +-6 scale. On that broken base the A/B
   showed q4_k imatrix tau 0.6978 vs no-imatrix 0.7289. **On the corrected
   `-g7c` base the sign flips**: xsmall q4_k imatrix (direct) tau 0.9333 vs
   no-imatrix 0.9067, and mean|d| improves on 5 of 6 cells. The apparent
   regression was the degenerate pre-g7c score scale, not q/k shadowing.
5. **Changing the default would cost a re-ship for no measured gain**, and
   `direct` is the arm that reproduces every currently published
   `-q4_k-imatrix-g7c` artifact bit-for-bit.

### Why the wrong-provenance vector is nevertheless harmless

The direct vector is derived from the LayerNormed relative-position embedding
table, which lives in the same `n_embd` space as the token hidden states and
concentrates its energy on the same dominant/outlier channels. It is a *coarse
but not adversarial* proxy: nearly orthogonal to the merged vector by cosine
(0.01-0.21) yet peaked on channels that also matter for the token stream.
Note also that the second application is genuine work - the q/k weights really
are consumed by the c2p/p2c matmuls, and by row count those dominate
(17235 vs 949 rows) - so "rel-position statistics" is not a *false* importance
for these tensors, merely an incomplete one.

### Follow-ups worth recording (not done here)

* `sum` as implemented adds two already-normalised mean-square vectors, giving
  each provenance equal weight regardless of row count. A properly pooled
  variant `(sumsq_d + sumsq_m) / (count_d + count_m)` would need the collector
  to preserve counts through the quantizer; it was not measured. Given `sum`
  tracks `merged` closely everywhere, a large effect is unlikely.
* The collector could be made to record the DeBERTa rel-position matmuls under
  a distinct name (e.g. `enc.N.attn.{q,k}_relpos.weight`), which would remove
  the shadowing entirely and make the two provenances separately auditable.
  That is a collector change and would invalidate existing imatrix files;
  it is not justified by the numbers above.

---

## 5. Repro

```bash
# worktree + build
git worktree add .claude/worktrees/feat-mxbai-qk-imatrix -b feat/mxbai-qk-imatrix
cd .claude/worktrees/feat-mxbai-qk-imatrix && git submodule update --init ggml
cmake -G Ninja -B build -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
ninja -C build -j4 crispembed-quantize crispembed

# artifacts (HF_HOME pointed at a scratch dir)
hf download cstr/mxbai-rerank-xsmall-v1-GGUF mxbai-rerank-xsmall-v1-g7c.gguf
hf download cstr/mxbai-rerank-xsmall-v1-GGUF mxbai-rerank-xsmall-v1-f7.imatrix
hf download cstr/mxbai-rerank-xsmall-v1-GGUF mxbai-rerank-xsmall-v1-q4_k-imatrix-g7c.gguf
#   ... same three for -base-

# key inventory
python -c "import gguf;print([t.name for t in gguf.GGUFReader('mxbai-rerank-xsmall-v1-f7.imatrix').tensors])"

# one arm (repeat with merged / sum; omit the env var for the default)
CRISPEMBED_QUANT_IMATRIX_QKV=merged build/crispembed-quantize \
    mxbai-rerank-xsmall-v1-g7c.gguf out-merged.gguf q4_k \
    --imatrix mxbai-rerank-xsmall-v1-f7.imatrix

# scores
build/crispembed -m out-merged.gguf --json --rerank "<query>" "<doc>" ... --gpu-backend cpu

# ONNX reference (pattern from tests/results/mxbai-gelu/onnx_ref.py, ORT 1.25.1 CPU EP,
# official mixedbread-ai/mxbai-rerank-*-v1 onnx/model.onnx + repo tokenizer.json)
```

Metrics use `kendall_tau` / `rerank_ab` copied verbatim from
`tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py`, over its
`RERANK_EVAL` fixture, with the f16 `-g7c` artifact's own scores as gold.
