# G2b — DeepSeek-OCR2 crop-mode follow-up: diagnosis + default flip (2026-08-05)

## Part 1 — diagnosis of the two G2-recorded regressions (from recorded arms, no new compute)

**receipt_historical, Metal, 0.138 → 0.305 raw CER under crops.** The Metal-
crop and CPU-crop transcripts are byte-identical for the first 82 characters;
the fork is a single formatting decision on the fourth line — CPU emits plain
`CUST 94 REG 8 QPR 171`, Metal emits `**CUST 94 REG 8 OPR**` — after which the
Metal trajectory self-conditions into a markdown list style (`- item: price`).
Content is NOT degraded: on an alphanumeric-content view (markup and
punctuation stripped), the four arms score

| arm | alnum-content CER |
|---|--:|
| m-crop | 0.1286 |
| m-base | 0.1247 |
| c-crop | **0.1102** |
| c-base | 0.1522 |

i.e. crop mode is content-neutral on Metal (+0.004) and a content WIN on CPU
(−0.042). Ground truth reads `OPR 171` — the Metal-crop arm reads that token
correctly where CPU-crop reads `QPR`. Mechanism: one near-tie
formatting-token flip under Metal F16 activation rounding, then trajectory
self-conditioning — the T14/G2 class. The raw-CER penalty is entirely
markdown characters against a plain-text ground truth.

**synth_01_noise 0.0149 → 0.0448.** The crop and base transcripts differ ONLY
in four inserted colons after field labels (`Invoice number:`, `Subtotal:`,
`tax:`, `total:`); every content character is identical.

Both regressions are formatting drift; neither is a vision or crop-geometry
bug. No code fix is applicable — greedy decode legitimately takes a different
style trajectory when conditioned on crop features.

## Part 2 — default flip (coordinator decision)

`DS2_CROP_MODE` defaults ON; `DS2_CROP_MODE=0` restores the single-view path
(gate stays value-parsed in both directions). Rationale: the reference
contract runs `crop_mode=True` (contract-faithful configuration); cc0 raw CER
mean improves Metal 0.657→0.236 and CPU 0.279→0.185 (beating the A4
reference's 0.187); the F1 Metal german 1024-cap is fixed by crops; and the
two recorded regressions are formatting-only (above).

**Gates run (this directory; MTL0 verified in every Metal run's stderr):**
- Gold gate BOTH manifest entries with crops engaging on fox (800×200 > 768):
  `deepseek-ocr2` PASS cer=0.000, `deepseek-ocr2-stacked` PASS cer=0.000 —
  run twice: once on the flip-only binary, once on the FINAL binary
  (post-LEGACY-DECODE fix).
- Byte-identity (`identity.log`, 12/12 + 2/2): default-ON runs equal the G2
  crop arms byte-for-byte (Metal cc0 5/5, CPU cc0 5/5) and `DS2_CROP_MODE=0`
  equals the G2 base arms (2 spot pages, Metal) — all modulo one framing
  artifact: the g2 runner stripped the trailing newline the raw CLI emits.
- Synth spot check: `synth_00_clean` + `synth_01_noise` default-ON Metal
  IDENTICAL to the G2 crop-synth arm (2/2 comparable pages).

Also in this change, per the value-parse audit rule: `DS2_LEGACY_DECODE` was
presence-based (`=0` selected the LEGACY path); now value-parsed. Verified on
the final binary: env absent → persistent, output byte-equal to the identity
matrix (fix is a no-op on defaults); `=0` → persistent (stderr
`decode path = persistent step graph`), output byte-equal to default; `=1` →
legacy. Remaining presence-based `DS_*` gates (`DS_MMAP`, `DS_MOE_CPU`,
`DS_DBG`, `DS_SAM_CONV_CPU`, `DS_QWEN2_ENC_FLASH`, `DS_QWEN2_SCALAR`,
`DS_LLM_FLASH`, `DS_NO_KV`, `DS_LMHEAD_CPU`) are recorded in PLAN as a
follow-up audit item — not changed here (one variable per A/B).
