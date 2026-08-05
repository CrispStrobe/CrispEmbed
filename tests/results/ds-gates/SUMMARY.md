# DS_* value-parse audit — acceptance summary (2026-08-05)

The G2b follow-up: every remaining presence-based boolean gate in
`src/deepseek_ocr2.cpp` inverted `=0` into "enabled" (`getenv(X)` truthiness).
All now route through one value-parsed helper, `ds_env_on()` (set and not "0"
=> on), the same semantics the DS2_KV_F16 (`73beea9f`) and DS2_LEGACY_DECODE
(`8c210291`) fixes established.

**Converted:** the 9 gates enumerated in the G2b board row — `DS_MMAP`,
`DS_MOE_CPU`, `DS_DBG`, `DS_SAM_CONV_CPU`, `DS_QWEN2_ENC_FLASH`,
`DS_QWEN2_SCALAR`, `DS_LLM_FLASH`, `DS_NO_KV`, `DS_LMHEAD_CPU` — plus two
same-class gates the audit found beyond that list: `DS2_FORCE_CPU` (backend
selector; `=0` used to force CPU) and `DS_PROFILE` (diagnostic print).
Already-value-parsed gates (`DS2_KV_F16`, `DS2_NO_REPEAT_NGRAM`,
`DS2_KV_BUCKET`, `DS2_LEGACY_DECODE`, `DS2_CROP_MODE`) and value-carrying vars
(`DS_REF`, `DS_TEXT_TEST`) untouched.

**Out of scope, recorded:** `CRISPEMBED_DEEPSEEK_OCR2_BENCH` is presence-based
but follows the codebase-wide `CRISPEMBED_*_BENCH != nullptr` convention (8+
engines) — changing only this engine's copy would fork the convention; that is
a separate codebase-wide audit if wanted.

**New diagnostic:** under `DS_DBG=1` the engine prints one gate-resolution
line (`[dbg] gates: mmap=0 moe_cpu=0 ...`) in the run's own stderr. Several
gates select paths with no other observable marker, which is exactly how the
`=0` inversion went unnoticed until the G6/G2b audits.

## Verification (`run_gates.sh`, this directory; 26 runs, serialized, fox.png
= crop-engaging gold fixture, `--gpu-backend metal` explicit, MTL0 verified
per Metal run)

All 42 checks pass (`compare.log`). Per gate, three spellings:

- **absent == `=0`:** every `=0` arm's stdout is byte-identical to the shared
  absent-arm baseline (9/9 gates + `DS_LLM_FLASH` vs its `DS2_KV_F16=1`-held
  baseline). This is the behavior the fix changes — pre-fix, `=0` enabled each
  feature. The gate-resolution line reads 0 in every `=0` arm.
- **`=1` engages:** gate-resolution line reads 1, plus per-gate evidence in
  the run's own output:
  - `DS_MOE_CPU=1` → stderr `decode path = legacy per-layer (CPU MoE …)`,
    wall 13→300 s (per-token CPU MoE).
  - `DS_NO_KV=1` → stderr `… (DS_NO_KV=1)`, wall 13→28 s.
  - `DS2_FORCE_CPU=1` → stderr `DS2_FORCE_CPU=1 — CPU backend`, **mtl0=0**
    (every other run mtl0=2), wall 73 s.
  - `DS_PROFILE=1` → `[ds-profile] decode:` line.
  - `DS_LLM_FLASH=1` (under `DS2_KV_F16=1`) → decode-path line says
    `kv=f16, flash`; `=0` says `kv=f16` only.
  - `DS_QWEN2_SCALAR=1` → wall 20→576 s (CPU-scalar 24-layer encoder).
  - `DS_SAM_CONV_CPU=1` → wall 31→79 s (threaded CPU conv chain).
  - `DS_MMAP=1` → stdout byte-identical to base (same weights either way);
    the no-copy path itself is covered by hermetic `test-gguf-loader-mmap`
    (run on this binary: PASS, copy==mmap==written).
  - `DS_DBG=1` → `[dbg]` lines present; absent and `=0` → none.
- **Decoded output read, not just diffed** (HARD RULE #3): base fox transcript
  is the correct pangram text; every `=1` arm decodes byte-identically to base
  on fox — the experimental paths agree on this fixture, so engagement is
  proven by the markers/wall-times above, not output difference.
- **No-op vs main:** a default-env receipt_historical run on this binary is
  byte-identical to the recorded g2b default arm
  (`tests/results/g2b/m-default-cc0/receipt_historical.txt`, raw CLI capture —
  no newline framing caveat).

## Provenance

`*.txt` / `*.err` — per-arm stdout/stderr. `run_gates.sh` — runner +
comparisons (comparison block re-run standalone after a macOS bash-3.2
`declare -A` failure; runs themselves were unaffected). `compare.log` — the
42-check output. Ambient load during the matrix: loadavg 2.5 at start (1-min);
no timing claims are made beyond order-of-magnitude engagement signatures.
