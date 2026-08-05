# G2 (F5) — DeepSeek-OCR2 dynamic-crop port: acceptance summary (2026-08-05)

Port at `82a45a59` (branch `feat/ds2-dynamic-crop`), opt-in `DS2_CROP_MODE=1`,
blueprint = `modeling_deepseekocr2.py` + `deepencoderv2.py` at the contract's
pinned revision `aaa02f38`. Model: `deepseek-ocr2-q4_k-stacked.gguf` (pinned).
Guard (F1) at its default ngram=20 in every arm. Decoded text judged only; NO
timing claims (box carried concurrent agent load + a parallel session).

## Gates

**A. Crop-off is a byte-level no-op of the refactor** — all 5 labelled cc0
transcripts byte-identical to the F1 baselines (`tests/results/f1/
{m,c}-guard-persist-cc0`) on Metal AND CPU (10/10 files).

**B. Crop-on is a no-op where the reference takes no crops (≤768px)** —
`simple_form` byte-identical both backends; synth 9/10 byte-identical (the one
differing page, `synth_01_noise`, is >768px so crops legitimately engage).

**C. cc0 CER vs the A4 reference (0.18743 raw):**

| fixture (raw CER)          | m-crop | m-base | c-crop | c-base |
|---------------------------|-------:|-------:|-------:|-------:|
| commons_example_receipt   | 0.2211 | 0.2211 | 0.2211 | 0.2211 |
| commons_test_ocr_document | 0.0074 | 0.3331 | 0.0074 | 0.1040 |
| german_official_print     | 0.1952 | 2.1387 | 0.2854 | 0.5907 |
| receipt_historical        | 0.3047 | 0.1380 | 0.1354 | 0.2044 |
| simple_form (no crops)    | 0.4534 | 0.4534 | 0.2753 | 0.2753 |
| **mean**                  | **0.2364** | 0.6569 | **0.1849** | 0.2791 |

CPU crop-mode mean **0.1849 beats the reference's 0.187 raw**. Metal mean
0.657→0.236. `commons_test` is near-perfect (0.0074) on BOTH backends and the
two backends' decode trajectories converge (identical gen counts on 3/5 pages
— crop features condition the decode much better).

**D. The F1 "Metal german cap" is FIXED by crop mode** — `german_official_print`
Metal: 1024-cap → 366 tokens, CER 2.139→0.195. (F1 predicted F5 was the fix.)

**Gold gate (crop off = default):** fox.png cer=0.000 + garbage-guard PASS on
BOTH manifest entries (per-expert and stacked), run with the final binary.

## Regressions found (why the gate stays OPT-IN this round)

1. `receipt_historical` on **Metal**: 0.1380 → 0.3047 with crops (CPU improves
   0.204→0.135 on the same page). Metal-specific crop-path quality issue.
2. `synth_01_noise` (noisy synthetic >768): 0.0149 → 0.0448.

Per the A/B discipline (flip only when it wins everywhere; keep plausible
paths gated), `DS2_CROP_MODE` ships opt-in/default-OFF. Follow-up (G2b):
diagnose the Metal receipt_historical regression, then decide the default
flip — the reference contract itself runs crop_mode=True, so default-ON is
the contract-faithful end state once the regression is understood.

## Provenance

Arms in this directory (`run_matrix.sh`): `{m,c}-{crop,base}-cc0`,
`m-{crop,base}-synth` + `*-score.json` (tests/score_gold_transcripts.py, raw)
and `runs.json` per arm. The 69-minute `commons_test` decode in the first arm
was box contention (parallel G4 agent + downloads), not the port — timing in
`runs.json` is not evidence.
