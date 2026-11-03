# MoEdern Q4_K lane: plain padded and mixed recipes (3 November 2026)

Goal: for each MoEdern ColBERT size, the smallest Q4_K-class package whose final
output clears the owner's Q4_K bar ([precision-acceptance.md](precision-acceptance.md),
> 0.94), measured against the frozen ORIGINAL model, with Q8_0 as the baseline
(bar > 0.97). Method as in [qwen2vl-q4k-recipes.md](qwen2vl-q4k-recipes.md):
start from Q8_0, lower one tensor group to Q4_K at a time, combine the safe groups.

## Protocol

- Reference: the frozen original token vectors of the ten-case raw-text panel
  (`ordinary.features.token_embeddings` and `ordinary.selected`), pinned by report
  SHA-256 in [moedern-q4k-padded.json](moedern-q4k-padded.json) and every array
  SHA in that report. 1.5M comes from GitHub run 40412164041, 27M/115M from the
  explicit Kaggle versions (v3/v1). Transport for hosted CPU is the public HF
  dataset `cstr/crispembed-regression-fixtures` at revision
  `3528982465132c835ea0fc7c0b8b4b90ea99ecae`; all hashes are re-checked.
- Model source: the published self-contained F32 packages (exact SHA).
- Metric: `final_cos_min` = minimum row cosine over **every token vector** of
  every case (all tokens and the selected scoring vectors). This is the strict
  reading of "embedding" for a multi-vector model; the per-case cosine of the
  whole vector matrix (`flattened_cos_min`), MaxSim error and exact IDs/masks
  are reported beside it.
- Build: hosted CPU, portable (`GGML_NATIVE=OFF`, `GGML_LLAMAFILE=ON`,
  `-ffp-contract=off`), x86 (AMD EPYC 7763) and ARM (Neoverse-N2). CUDA is not
  applicable: the MoEdern family engine is a CPU-only ggml graph.
- Driver: [`tools/ci-heavy/moedern_q4k_lanes.py`](../tools/ci-heavy/moedern_q4k_lanes.py);
  recipes: [`tools/moedern-colbert/quantize_q4k_mixed.py`](../tools/moedern-colbert/quantize_q4k_mixed.py).
  Receipts: `tests/results/moedern-colbert/q4k-lane-{1p5m,27m,115m}.json`.

## Runtime

`CRISPEMBED_MOEDERN_Q4_K_MIXED` (default off, needs the padded Q4_K gates
`Q4_K_PADDED`/`QUANT`/`FAST` and the family gates) admits a file with recipe
`q4_k_padded_mixed_v1` and a named `moedern.q4k.policy`: every declared tensor is
real zero-padded Q4_K (as the plain padded profile), every other engine-quantizable
matmul weight Q8_0 (32-aligned rows) or F32, and embeddings, router, norms,
depthwise conv kernels and the dense head stay F32. The bounded header reader
([`src/moedern_q4k_mixed.h`](../src/moedern_q4k_mixed.h)) is the padded one plus
Q8_0 extents. The plain padded and named Q8_0 profiles, their loaders and the
fast graph are unchanged; with the gate off nothing changes.

## Results (hosted CPU, portable)

Size relative to the named Q8_0/F32-critical package of the same size.

### 115M (Q8_0 134.55 MiB, F32 438.59 MiB)

| Arm | MiB | vs Q8_0 | final cos min x86 / ARM | flat cos min | MaxSim max abs |
|---|---|---|---|---|---|
| F32 control | 438.59 | 3.260 | 1.0000 / 1.0000 | 1.0000 | 0.0000 |
| Q8_0 (named) | 134.55 | 1.000 | **0.9670** / 0.9672 (d-unicode) | 0.9949 | 0.539 |
| plain padded Q4_K = `moedern115m_q4k_v1` | **85.34** | **0.634** | **0.9793** / 0.9789 | 0.9933 | 0.520 |
| q8 + attention Q4_K | 128.56 | 0.955 | 0.9644 | 0.9969 | 0.142 |
| q8 + attn_o Q4_K | 132.56 | 0.985 | 0.9864 | 0.9984 | 0.059 |
| q8 + experts Q4_K | 106.81 | 0.794 | 0.9696 | 0.9959 | 0.466 |
| q8 + conv Q4_K | 122.56 | 0.911 | 0.9653 | 0.9939 | 0.137 |
| q8 + layers 0-4 Q4_K | 117.25 | 0.871 | 0.9863 | 0.9975 | 0.354 |
| q8 + layers 5-9 Q4_K | 102.64 | 0.763 | 0.9665 | 0.9940 | 0.215 |

Every single-group arm is in the receipt. The smallest recipe is the plain
padded Q4_K (52 Q4_K / 42 F32 tensors, 1024- and 640-wide rows): it clears 0.94
on both ISAs with every case at or above 0.979 and degrades all tokens
uniformly (per-case mean row cosine 0.993-0.998).

### 27M (Q8_0 35.67 MiB, F32 104.52 MiB)

| Arm | MiB | vs Q8_0 | final cos min x86 / ARM | flat cos min | MaxSim max abs |
|---|---|---|---|---|---|
| Q8_0 (named) | 35.67 | 1.000 | **0.9315** / 0.9315 (q-de) | 0.9959 | 0.070 |
| plain padded Q4_K | 26.59 | 0.745 | 0.9250 / 0.9219 | 0.9848 | 0.369 |
| q8 + exps_up + exps_down + attn_q = `moedern27m_q4k_v1` | **30.53** | **0.856** | **0.9685** / 0.9688 | 0.9953 | 0.183 |
| q8 + exps_up + exps_down (earlier policy draft, same bytes as its named arm) | 30.83 | 0.864 | 0.9683 / 0.9681 | 0.9955 | 0.204 |
| q8 + all experts + attn_q/k | 27.95 | 0.784 | 0.9405 / **0.9390** | 0.9944 | 0.193 |
| q8 + all experts | 28.41 | 0.796 | 0.9233 | 0.9946 | 0.251 |
| q4 with attn_o/shexp Q8_0 | 27.80 | 0.779 | 0.9390 | 0.9927 | 0.196 |
| q8 + layer 0 / layer 1 | 31.13 | 0.873 | 0.9311 / 0.9309 | 0.99 | 0.33 / 0.22 |

Single-group sweep (x86): attn_q 0.9680, attn_k 0.9684, attn_v 0.9515, attn_o
0.9230, exps_up 0.9680, exps_gate 0.9301, exps_down 0.9679, shexp 0.9278.
`q8+exps+attn_q+attn_k` passes on x86 by 0.0005 and fails on ARM, so it is
rejected as fragile. The smallest recipe that clears 0.94 on both ISAs is
`moedern27m_q4k_v1` (routed-expert up/down and attn_q Q4_K padded 640->768,
everything else as the Q8_0 package). A named-policy arm and its recipe string
produce identical vectors (run 41618191791).

### 1.5M (Q8_0 3.96 MiB, F32 6.71 MiB)

| Arm | MiB | vs Q8_0 | final cos min | flat cos min |
|---|---|---|---|---|
| Q8_0 (named) | 3.96 | 1.000 | 0.8752 (d-punctuation) | 0.9799 |
| plain padded Q4_K | 4.02 | 1.015 | 0.8040 (q-empty) | 0.9304 |
| q8 + attention / experts / shared Q4_K | 3.96-4.01 | 1.00-1.01 | 0.832 / 0.876 / 0.876 | 0.94-0.97 |

No recipe clears 0.94, and none can be smaller than Q8_0: every 1.5M matmul row
is 128 wide, so a Q4_K row pads to 256 columns (144 bytes per 128 weights,
9 bits/weight versus Q8_0's 8.5). Q4_K is not useful for this size.

## What limits the score: discrete routing flips

The token-level minimum is not monotone in precision. Q8_0 fails its own 0.97
bar on 27M (0.9315) and 115M (0.9670) through **one token each** (q-de row 8;
d-unicode row 15) while the case mean stays above 0.995, and several 4-bit arms
score higher than Q8_0. This is the near-boundary top-2 expert choice already
recorded for the F32 capture (PLAN, MoEdern rows): a small perturbation of the
hidden state flips one expert for one token. Plain Q4_K on 115M instead lowers
every token a little (mean 0.993-0.998) without a flip below 0.979. Consequences:

- A 27M recipe that clears 0.94 does so partly because it does not trip that
  token; `moedern27m_q4k_v1` was therefore confirmed on a second ISA and its
  worst token (q-empty, 0.9685) is the same one the Q8_0 package has at 0.968.
- Q8_0 is **not** qualified under its bar for 27M or 115M on this metric, and
  neither size's Q8_0 has a mixed fix in this lane (the router is already F32).
- All IDs/masks are exact in every arm; MaxSim errors up to 0.54 (115M) are
  recorded, and retrieval ranking quality is a separate, unmet gate.

## Named policies and publication

`POLICIES` in `quantize_q4k_mixed.py`: `moedern115m_q4k_v1` (`q4`) and
`moedern27m_q4k_v1` (`q8+exps_up+exps_down+attn_q`), bound to their variant and
refused for any other. The 115M policy writes the same tensor payloads as the
plain padded converter (identical vectors in the receipt), stamped as the mixed
recipe. [moedern-q4k-publication-v1.json](moedern-q4k-publication-v1.json)
holds `Q4_K_MIX` recipes for 27M and 115M with `precision_gate: cosine`;
nothing is published, no default changes, and retrieval (SciFact) qualification
of a quantized package remains open.

## Initial failures kept on record

- `moedern-q4k-padded.yml` panel at source `83eac856` (runs 41587636658/9310/41989):
  1.5M completed (plain padded Q4_K min row cosine 0.8040, F32 control exact);
  27M and 115M stopped in authenticated Kaggle input preparation
  (`ModuleNotFoundError` in the pinned Kaggle client path) before any model
  access. The lane driver takes the same pinned originals from the fixture
  dataset instead; that workflow and its guards are unchanged.
- `tests/test_moedern_q8_profile.py::test_exact_source_closure_and_preserved_old_paths`
  pins the frozen Q8 review source (`dd592b15`) by SHA of
  `src/moedern_family_engine.cpp`; this branch's engine change makes that
  frozen-source assertion reject the current tree, as the padded Q4_K source
  closure already rejected main. The pin is not edited.
