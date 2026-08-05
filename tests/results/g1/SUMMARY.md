# G1 (F4) — SmolDocling vision split residency: acceptance summary (2026-08-05)

Port on branch `feat/smoldocling-metal`: SigLIP vision graphs (patch embed +
12-layer transformer) run on the GPU backend when one is available; the `vis.*`
weights are GPU-resident via the new `core_gguf::load_weights_split` (logic
synced from CrispASR PLAN #69a). The connector, 135M LLM decode, LM head and
KV cache stay CPU-resident (per the persistent-decode LEARNINGS: per-token
small-matmul steps are CPU-shaped). `SMOLDOCLING_FORCE_CPU=1` (value-parsed:
`=0` is off) and `--gpu-backend cpu` both restore the historical all-CPU
engine. Model: pinned `smoldocling-q8_0.gguf`. Build: worktree of main at
`a083aa28`, `GGML_METAL:BOOL=ON` verified in CMakeCache; every Metal run's own
stderr carries MTL0 (enforced by the matrix runner).

## Gates

**A. CPU arm is a byte-level no-op of the refactor** — all 5 T15 pages match
the recorded T15 CPU outputs (`ours_raw_doctags` in
`tests/results/ocr_parity_smoldocling_2026-08-04.json`) exactly (5/5,
`compare.json` `cpu_matches_t15_ours`).

**B. CPU vs Metal decoded output (the 5-page T15 set):**

| page | raw byte-identical | stripped payload identical |
|---|---|---|
| fox.png | no (class tag + ±2 loc) | **yes** |
| scan_page_pd.png | **yes** | yes |
| commons_test_ocr_document.jpg | no | no (1 char-class delta) |
| simple_form.png | **yes** | yes |
| receipt_historical.png | no | no (segmentation/trajectory) |

**C. Quality vs cc0 ground truth and vs the reference (markup-stripped CER):**

| page | metal vs GT | cpu vs GT | reference vs GT (T15) |
|---|--:|--:|--:|
| commons_test_ocr_document.jpg | 0.0077 | 0.0077 | 0.0956 |
| simple_form.png | 1.0 | 1.0 | 3.2267 |
| receipt_historical.png | 0.3724 | 0.2383 | 0.4935 |

The Metal arm **matches or beats the reference implementation on every
GT-labelled page** (fox payload CER vs ref = 0.0 for both arms). The one
CPU-vs-Metal quality gap, `receipt_historical` (0.238 → 0.372 vs GT), is a
decode-trajectory divergence from Metal F16 activation rounding in the vision
tower: content is coherent (finer line segmentation, no garbage — transcripts
read), and the Metal trajectory is much CLOSER to the reference's own output
(CER vs reference payload 0.416 vs CPU's 0.758). Same accepted mechanism as
the deepseek-ocr2 Metal-vs-CPU divergence (T14/G2).

**D. Timing (same-window interleaved (metal,cpu) pairs, loadavg 2.3-5.1 all
« 8, pair 0 cold discarded, median of 3 scored pairs, min-max un-trimmed;
raw lines in `timing/summary.txt`):**

| page | stage | metal median (min-max) | cpu median (min-max) | speedup |
|---|---|---|---|--:|
| fox.png (5 tiles) | vision+connector | 1241 ms (1228-1249) | 3562 ms (3553-3574) | 2.87x |
| fox.png | total | 1876 ms (1862-1881) | 4216 ms (4196-4251) | 2.25x |
| scan_page_pd.png (13 tiles) | vision+connector | 3163 ms (3102-3295) | 14500 ms (11627-16863) | 4.58x |
| scan_page_pd.png | total | 11021 ms (10306-13710) | 22681 ms (18717-26347) | 2.06x |

Per-pair vision ratios (scan): 3.68x / 4.67x / 5.12x — ambient load crept
2.7→5.1 during the scan pairs and inflates the CPU arm more than Metal; the
floor-vs-excursion argument (floors: 11627/3102 = 3.75x) bounds the true
vision win at ≥3.7x on the 13-tile page. Decode step time is unchanged
(~12 ms fox / ~19 ms scan on both arms — decode never moved off CPU).
Run-to-run determinism: all 12 timing-run transcripts byte-identical to the
matrix outputs (per arm).

## Default disposition (coordinator decision)

GPU vision ships as the DEFAULT when a GPU device exists, matching every other
GPU-capable OCR/VLM engine (deepseek precedent: DS2 ships Metal-default with a
larger documented Metal-vs-CPU cc0 gap). Quality bar used: match/beat the
reference implementation — satisfied on all pages. The old path remains fully
reachable (`SMOLDOCLING_FORCE_CPU=1`, `--gpu-backend cpu`) as the
regression-bisection gate.

## Provenance

`metal/`, `cpu/` — raw doctags + per-run stderr (bench lines, MTL0).
`compare.json` — identity + CER-vs-reference table. `run_matrix.sh`,
`interleave.sh`, `timing/` — runners and timing raw data. Matrix ran with
ambient load ~4-6 (loadavg logged per run, all « 8); a parallel CrispASR
session's script held one core during part of the matrix — identity results
are load-independent, timing came from the dedicated interleaved windows.
