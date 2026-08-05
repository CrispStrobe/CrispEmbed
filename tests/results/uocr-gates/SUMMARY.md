# UOCR_* value-parse audit — acceptance summary (2026-08-05)

The exact mirror of the DS_* audit (`91ebb55d`, `tests/results/ds-gates/`)
applied to `src/unlimited_ocr.cpp`. Every presence-based boolean gate there
(`getenv(X) != nullptr`, or bare `if (getenv(X))`) inverted `X=0` into
"enabled". All 17 now route through the shared value-parsed helper
`core_env::on()` from `src/core/env_gate.h` (set, non-empty, not `"0"` => on),
guarded hermetically by `test-env-gate`.

**Headline: on `main`, `UOCR_PD=0` turned the persistent-decode path ON and the
process SEGFAULTED with empty stdout.** This defect class was not merely
diagnostic noise in this engine — it destroyed the OCR result.

## Inventory (all 47 `UOCR_*` env sites; line numbers post-fix)

Boolean gates — **converted** (40 call sites, 17 variables):

| line(s) | variable | old form | inverted? | what it selects | output-affecting |
|---|---|---|---|---|---|
| 497 | `UOCR_MMAP` | `!= nullptr` | no | no-copy mmap GGUF load | no (same weights) |
| 640, 2820 | `UOCR_MOE_CPU` | bare / `if (!…)` | 2820 inverted | builds per-expert MoE views; **2820 disables the prestacked/Metal MoE** | yes (CPU vs Metal MoE) |
| 912, 963, 969, 1005, 1071, 1259, 2298, 2307, 2714, 2730, 2803, 2871, 2910, 2972, 3163, 3190 | `UOCR_DBG` | `!= nullptr` / bare / `if (!…)` | 912, 2803 inverted | stderr diagnostics + stage timings | stderr only |
| 934, 1167 | `UOCR_SAM_CONV_CPU` | `if (!…)` | **yes** | SAM patch-embed + neck convs on CPU instead of a ggml graph | yes (different numerics path) |
| 1047 | `UOCR_OPT_GRAPH_LN` | `!= nullptr` | no | SAM LN inside the ggml graph vs CPU LN + separate residual | **yes — measured** |
| 1295, 1397, 1429, 1442 | `UOCR_CLIP_DBG` | `!= nullptr` / bare | no | CLIP graph node budget 8192 vs 4096 + intermediate dumps | stderr (structure only) |
| 1715 | `UOCR_OPT_PD_F32` | bare (ternary) | no | PD KV cache F32 instead of F16 | yes, but PD is unreachable (see D) |
| 1796, 2394, 2585 | `UOCR_PD_DBG` | bare | no | adds `ggml_cont`+`set_output` per PD layer; `[pd_dbg]`/`[rb_dbg]` dumps | stderr (structure only) |
| 1903 | `UOCR_FA_F32` | bare | no | `ggml_flash_attn_ext_set_prec(F32)` in the LLM attention graph | potentially (numerics) |
| 2260 | `UOCR_LMHEAD_CPU` | `!= nullptr` | no | LM head matmul on CPU | yes (different numerics path) |
| 2264 | `UOCR_NO_KV` | `!= nullptr` | no | disables the KV cache; re-prefills every step | yes (perf; also blocks PD) |
| 2265 | `UOCR_OPT_FUSED_DECODE` | `!= nullptr` | no | fused multi-layer decode graph | yes (different graph) |
| 2280 | `UOCR_PD` | bare, compound | no | persistent-decode graph — **segfaults on this fixture** | **yes — crash** |
| 2280, 2734 | `UOCR_DECODE_REBUILD` | bare, compound | **yes** (`&& !…`) | forces the per-step rebuild path (vetoes PD) | yes (vetoes PD) |
| 2624 | `UOCR_DECODE_TIMING` | bare | no | `[rb_timing]` print | stderr only |
| 3009 | `UOCR_INJECT_VIS` | bare, compound | no | feeds reference vision features instead of SAM/CLIP | yes — but a no-op unless `UOCR_REF` is set |
| 3043 | `UOCR_INJECT_REF` | bare, compound | no | replaces SAM output with the reference's | yes — but a no-op unless `UOCR_REF` is set |

Value-carrying — **left untouched** (read with `atoi` / used as a string):
`UOCR_REF` (2774), `UOCR_TEXT_TEST` (2874), `UOCR_OPT_SAM_RES` (2906),
`UOCR_INSTR` (3115), `UOCR_MAX_NEW` (3175), `UOCR_NO_REPEAT_NGRAM` (2676),
`UOCR_NGRAM_WINDOW` (2677).

Out of scope, already done: `CRISPEMBED_UNLIMITED_OCR_BENCH` (converted by the
BENCH audit). `UOCR_TIMING` appears in the task brief but **does not exist** in
this file or anywhere in the tree — no such gate.

**New diagnostic:** under `UOCR_DBG=1` the engine prints one gate-resolution
line in the run's own stderr. Baseline reads:

```
  [dbg] gates: mmap=0 moe_cpu=0 sam_conv_cpu=0 opt_graph_ln=0 clip_dbg=0 opt_pd_f32=0 pd_dbg=0 fa_f32=0 lmhead_cpu=0 no_kv=0 opt_fused_decode=0 pd=0 decode_rebuild=0 decode_timing=0 inject_vis=0 inject_ref=0 dbg=1
```

## Verification

`run_gates.sh` — 44 model runs, strictly serialized (one heavy process at a
time), `unlimited-ocr-q4_k-stacked.gguf` at the regression-manifest pin
`b11fef884fee`, fixture `tests/regression/images/fox.png`, `--gpu-backend
metal` explicit on every run, `MTL0` present in every run's own stderr
(`mtl0=2` on all 44 — no arm silently fell back to CPU). `compare.sh` — **83
checks, 0 failures** (`compare.log`). Per-arm stdout/stderr are the `.txt` /
`.err` files here; per-run wall times and exit codes are in `run.log`.

### A. Pre-fix controls — `=0` used to engage (parent-commit binary)

| arm | result |
|---|---|
| `pre-base` (default env) | rc=0, 45 s, decodes the gold text |
| `pre-DBG-0` (`UOCR_DBG=0`) | `[dbg]` lines **present** — `=0` was ON |
| `pre-PD-0` (`UOCR_PD=0`) | stderr `decode step gen=1 n_past=277 T=1 pd=1` → **Segmentation fault: 11**, rc=139, **stdout EMPTY** |
| `pre-NO_KV-0` (`UOCR_NO_KV=0`) | `gen=1 n_past=0` — KV cache disabled; 71 s vs 45 s baseline |

Post-fix the same spellings are inert: `UOCR_DBG=0` → no `[dbg]` lines;
`UOCR_PD=0` → `pd=0`, rc=0, gold text; `UOCR_NO_KV=0` → KV cache retained.

### B. Post-fix: every `=0` arm is byte-identical to the shared baseline

All 13 always-relevant gates plus the 3 PD-conditional gates: the
gate-resolution line reads `0` in every `=0` arm, and every `=0` arm's stdout
is byte-identical to the corresponding absent-arm baseline (`base.txt`, or
`pd-base.txt` for the PD-conditional group). `UOCR_DBG=1` and `UOCR_DBG=0`
stdout both equal `base.txt` (it writes stderr only).

### C. `=1` engagement proof, per gate, from that run's own output

| gate | proof | wall (base ≈ 17-21 s) |
|---|---|---|
| `UOCR_DBG` | `[dbg]` lines + gate-resolution line present; absent/`=0` have none | 17 s |
| `UOCR_MOE_CPU` | stderr **lacks** `using prestacked MoE experts` (present in every other arm) | **276 s** (CPU MoE per token) |
| `UOCR_SAM_CONV_CPU` | wall-time signature (threaded CPU conv chain) | **64 s** vs 22 s |
| `UOCR_LMHEAD_CPU` | wall-time signature (CPU LM-head matmul) | **35 s** vs 18 s |
| `UOCR_NO_KV` | stderr `decode step gen=1 n_past=0` (n_past never advances) | **51 s** vs 17 s |
| `UOCR_CLIP_DBG` | stderr `[dbg] clip input …` dumps; `=0` has none | 17 s |
| `UOCR_DECODE_TIMING` | stderr `[rb_timing] gen=…`; `=0` has none | 18 s |
| `UOCR_PD` | stderr `decode step … T=1 pd=1` (absent/`=0` say `pd=0`), then **segfault** | 13 s (crash) |
| `UOCR_DECODE_REBUILD` | under `UOCR_PD=1`: `=1` → `pd=0`, rc=0, gold text; `=0`/absent → `pd=1`, segfault | 28 s |
| `UOCR_PD_DBG` | 24 × `[rb_dbg] gen=… layer=…` lines on the rebuild path; `=0` → 0 lines | 19 s |
| `UOCR_OPT_FUSED_DECODE` | with `UOCR_DECODE_TIMING=1` as reporter: `=1` → **0** `[rb_timing]` lines (the fused path sets `did_pd`, skipping the `if (!did_pd)` rebuild block that owns the print); `=0` → 3 lines | 14 s |
| `UOCR_OPT_GRAPH_LN` | **decoded output differs** — see D | 17 s |
| `UOCR_MMAP` | gate-resolution line `mmap=1`; the no-copy loader path itself is covered hermetically by `test-gguf-loader-mmap`, re-run on this build: `no-copy mmap path taken (used_mmap=1)`, `PASS: gguf_loader no-copy mmap == copy` | 15 s |
| `UOCR_FA_F32` | gate-resolution line `fa_f32=1` only — no runtime marker; output byte-identical to base and wall within noise on this fixture. **Engagement is parse-level, not behaviour-level.** | 20 s |
| `UOCR_OPT_PD_F32` | gate-resolution line `opt_pd_f32=1` only — it re-types the PD KV cache, and the PD path segfaults before finishing either way. **Not behaviourally verifiable on this fixture.** | 27 s (crash) |
| `UOCR_INJECT_VIS` | gate-resolution line `inject_vis=1` only — the site is `core_env::on(…) && !diff_ref_path.empty()`, and no `UOCR_REF` reference dump exists for this model, so it is a guaranteed no-op here (stdout == base confirms) | 19 s |
| `UOCR_INJECT_REF` | same as above (`inject_ref=1`, compound-guarded by `UOCR_REF`) | 17 s |

### D. Output differences, recorded verbatim

Baseline `base.txt` matches the manifest gold exactly — **CER 0.0000**
(max_cer 0.15):

```
title [33, 196, 636, 360]The quick brown fox jumps
title [34, 548, 588, 740]over the lazy dog. 12345
```

Eleven of the 13 `=1` arms are byte-identical to that. The two that are not:

**`UOCR_OPT_GRAPH_LN=1` — different bounding-box coordinates (text unchanged):**

```
- title [33, 196, 636, 360]The quick brown fox jumps
- title [34, 548, 588, 740]over the lazy dog. 12345
+ title [33, 232, 636, 360]The quick brown fox jumps
+ title [33, 498, 588, 660]over the lazy dog. 12345
```

The recognized text is identical; the grounding coordinates move (y0 196→232,
x0 34→33, y0 548→498, y1 740→660). This is an `OPT_*` numerics path (SAM LN
in-graph vs CPU LN + separate residual), so a difference is legitimate — it is
recorded, not hidden, and not "fixed". Evidence for the coordinator, not a
defect introduced by this change.

**`UOCR_PD=1` (and `pd-base`, `UOCR_OPT_PD_F32=0/1`, `UOCR_PD_DBG=0/1` under
`UOCR_PD=1`, `UOCR_DECODE_REBUILD=0` under `UOCR_PD=1`) — empty stdout:**

```
- title [33, 196, 636, 360]The quick brown fox jumps
- title [34, 548, 588, 740]over the lazy dog. 12345
+ (nothing — Segmentation fault: 11, rc=139)
```

7 of the 44 runs crashed, all of them PD-path runs. The crash lands in the
`gen=2` PD compute (last stderr line is always `decode step gen=2 n_past=278
T=1 pd=1`). **This is pre-existing on `main`** — the parent-commit binary
crashes identically under `UOCR_PD=0`, so the fix did not introduce it; the fix
is what stops `UOCR_PD=0` from reaching it. The in-source comment at line 2272
already warns the PD path "diverges from the per-step rebuild path"; on this
model/fixture it does not merely diverge, it faults. Not fixed here — out of
scope for a gate audit, flagged for the coordinator.

### E. No-op vs `main`

`pre-base.txt` (parent-commit binary, default env) is **byte-identical** to
`base.txt` (fixed binary, default env). Unset was OFF before and is OFF now;
nothing about the default path changed.

### F. Hermetic gates

- `./build/test-env-gate` → `env-gate: 10 checks, 0 failure(s)`
- `./build/test-gguf-loader-mmap` → `PASS: gguf_loader no-copy mmap == copy`
- `tests/test_ocr_backend_matrix.py` → `OCR backend matrix OK: 12 required families, 16 schema-valid rows`

## Provenance / caveats

- `*.txt` / `*.err` — per-arm stdout/stderr, one file per run. `run.log` — the
  runner's own per-run rc / wall / MTL0 tally. `compare.log` — the 83-check
  output. `run_gates.sh` accepts `STAGE=pre|post|rb|fd` to re-run one block;
  the `rb` and `fd` blocks were added after the first 40-run pass and re-run
  standalone (the earlier runs were unaffected).
- macOS bash 3.2 throughout — no `declare -A`.
- Wall times are single-sample, taken after the coordinator's reranker matrix
  finished (marker `MATRIX_DONE`). Only order-of-magnitude engagement
  signatures are claimed from them, never a performance result.
- `UOCR_FA_F32`, `UOCR_OPT_PD_F32`, `UOCR_INJECT_VIS`, `UOCR_INJECT_REF` are
  verified at the parse level (gate-resolution line) but have **no behavioural
  proof on this fixture** — stated explicitly above rather than folded into a
  pass count.
