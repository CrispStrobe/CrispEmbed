# G6 (F6) — DS2_KV_F16 measurement vs the F32-KV baseline

Measurement only; `DS2_KV_F16` stays opt-in. The promotion decision is not made
here. Branch `feat/ds2-kv-f16-quant` (worktree of main at e81c827e); binary
built with `GGML_METAL:BOOL=ON` (verified in CMakeCache.txt); Metal evidence
noted in §5.

Baseline = `tests/results/f1/{m,c}-guard-persist-{cc0,synth}` (F32 KV, guard on,
`DS2_FAST_DECODE=1`). Current main reproduces the f1 CPU cc0 baseline
byte-identically (re-checked here: `g2/c-base-cc0` == `f1/c-guard-persist-cc0`,
5/5 files), so every byte difference below is attributable to `DS2_KV_F16=1` —
the only variable changed. All runs serialized, one model process at a time.

## 1. Decoded-text quality — byte identity vs f1

| arm | backend / corpus | ok | byte-identical vs baseline | differing |
|---|---|---|---|---|
| m-kvf16-synth | metal, synth (20) | 20/20 | 20/20 | — |
| m-kvf16-cc0 | metal, cc0 labelled (5) | 5/5 | 4/5 | german_official_print |
| c-kvf16-synth | cpu, synth limit 5 | 5/5 | 5/5 | — |
| c-kvf16-cc0 | cpu, cc0 labelled (5) | 5/5 | 0/5 | all 5 |

Pattern: on Metal the f16 cache is almost transparent (Metal matmuls already
round activations to f16); on CPU (F32 GEMM) the stored-KV rounding perturbs
every logit stream, so all long-page cc0 decodes drift. Short/clean synth pages
are byte-stable on both backends.

## 2. CER where transcripts differ (score_gold_transcripts.py, engine=native)

Per-fixture CER, F32 baseline vs F16 arm (delta = f16 − f32; negative = f16 better):

| fixture | metal f32 | metal f16 | delta | cpu f32 | cpu f16 | delta |
|---|--:|--:|--:|--:|--:|--:|
| commons_example_receipt | 0.2211 | 0.2211 | +0.0000 | 0.2211 | 0.2187 | −0.0025 |
| commons_test_ocr_document | 0.3331 | 0.3331 | +0.0000 | 0.1040 | 0.2717 | +0.1677 |
| german_official_print | 2.1387 | 0.5233 | −1.6155 | 0.5907 | 0.3816 | −0.2091 |
| receipt_historical | 0.1380 | 0.1380 | +0.0000 | 0.2044 | 0.1289 | −0.0755 |
| simple_form | 0.4534 | 0.4534 | +0.0000 | 0.2753 | 0.4332 | +0.1579 |
| **aggregate (scorer)** | **0.6569** | **0.3338** | **−0.3231** | **0.2791** | **0.2868** | **+0.0077** |

- Metal: the only differing page is german_official_print, where the f32
  baseline rambles to the 1024-token cap (gen_tokens=1024, 2980 chars,
  CER 2.14 — CER>1 because output is much longer than truth) while the f16 run
  stops at 442 tokens (CER 0.52). One fixture, one direction — luck of a
  perturbed greedy path, not evidence f16 is "better".
- CPU: mixed, both directions (2 worse by ~+0.16, 2 better, 1 ~flat); aggregate
  +0.008 worse. Consistent with precision-perturbed greedy decode noise on an
  already-hard corpus, but it is a real text change on every long page.

## 3. Memory

KV allocation line (deterministic; captured from each run's own stderr):

```
deepseek_ocr2: KV cache: 12 layers, max_seq=1408, kv_dim=1280, f32, 165.0 MB
deepseek_ocr2: KV cache: 12 layers, max_seq=1408, kv_dim=1280, f16, 82.5 MB
```

Same on metal and cpu: −82.5 MB (−50%).

Peak footprint (`/usr/bin/footprint --sample 0.5`, one interleaved pair,
receipt_historical, metal): f32 3499 MB vs f16 3415 MB → −84 MB, matching the
KV-line delta. (Total is dominated by the q4_k model itself.)

## 4. Decode time

Interleaved A/B (receipt_historical.png, metal, 4 scored pairs + discarded cold
pair, load 1.7–2.0 « 8, no pairs dropped, transcripts identical across arms):

| arm | decode median (ms) | min–max | spread |
|---|--:|---|--:|
| kv_f32 | 19288.9 | 19142.9–19405.3 | 1.4% |
| kv_f16 | 19108.5 | 19048.5–19159.8 | 0.6% |

Per-pair f16/f32 decode ratios: 0.9999, 0.9816, 0.9969, 0.9854 → ~1% faster,
within noise. F16 KV is memory savings, not a speedup, on this fixture.

Corpus medians vs f1 runs.json (decode ms, byte-identical fixtures only) look
dramatically better (m-synth 1291.8 vs 1546.4, ratio 0.80; c-synth 516.4 vs
726.9, ratio 0.71; m-cc0 13399.6 vs 29130.4, ratio 0.46) — do not use these:
f1 explicitly recorded its timings under concurrent agent load and disclaims
timing claims. The interleaved A/B above is the timing verdict.

## 5. Observations / anomalies

- The gate is presence-based: `ds_kv_type()` is
  `getenv("DS2_KV_F16") ? F16 : F32`, so `DS2_KV_F16=0` also enables f16.
  Worth normalizing to value-parsing if the gate is ever promoted.
- Build note: the fresh worktree's `crispembed` target builds only the
  libraries; the CLI binary at `build/crispembed` is target `crispembed-cli`.
- Metal verification: MTL0 confirmed in stderr for the same-session smoke
  captures and enforced per-run inside the interleave harness (`ok` requires
  MTL0). The corpus runner discards stderr on success, so the matrix arms'
  Metal use is evidenced by the identical binary + `--gpu-backend metal` + the
  same-session smoke capture, not per-run MTL0.

## Files

- `run_matrix.sh`, `matrix.log` — the four arms (all ok, 0 failures).
- `m-kvf16-*/`, `c-kvf16-*/` — transcripts + runs.json (bench line per page).
- `compare_vs_f1.py`, `compare_vs_f1.json` — byte-identity + medians.
- `decode_medians.json` — corpus decode medians (identical-text subset).
- `m-kvf16-cc0/score.json`, `c-kvf16-cc0/score.json`,
  `score_f1_{m,c}-guard-persist-cc0.json` — CER scoring inputs to §2.
- `interleave_kv.py`, `interleave_kv.log`,
  `interleaved_kv_receipt_historical.json` — the timing A/B.
