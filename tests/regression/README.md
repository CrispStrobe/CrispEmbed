# CrispEmbed OCR portfolio regression suite

A model-output regression suite for the OCR engines. It exists because a
vision-neck permute regression (`3fb1f8e`, Jun 2026) shipped **garbage OCR**
(`colorcolorcolor…`) that the existing Kaggle `ocr-gpu-bench` kernel could not
catch — that kernel only checked process exit codes (garbage still exits 0) and
did not even include got-ocr2. See `docs/got-ocr2.md` and `LEARNINGS.md`.

## What each model run checks

For every model in `manifest.json`, `run_one.py`:

1. **Downloads** the GGUF under test (pinned to an HF revision SHA, so an
   upstream re-quantise can't silently change what we test).
2. **Runs** `crispembed -m <gguf> --ocr <image>` and captures stdout.
3. **No-garbage guard** — rejects the `colorcolor…` degeneration signature
   (a single word or short substring repeated far beyond real text). This is
   the check that would have caught `3fb1f8e`.
4. **Lenient text match** — normalises case/punctuation/whitespace and requires
   the character error rate (CER) vs `expected_text` to be ≤ `match.max_cer`
   (default 0.10). OCR is not byte-exact across builds; punctuation/spacing
   drift is fine, a wrong/empty transcript is not. (Think TTS→ASR round-trip
   tolerance.)
5. **Optional diff harness** — if the model declares a `diff` block *and* its
   reference GGUF exists on HF, downloads `<model>-ref.gguf` and runs
   `test-<model>-diff <gguf> <ref>`, asserting every stage's `cos_min` ≥ its
   pinned threshold. If the ref is absent on HF the diff step is **skipped, not
   failed** — diff testing is opt-in by the mere presence of the ref.

A model **passes** only if all applicable checks pass.

## manifest.json

```jsonc
{
  "version": 1,
  "match_defaults": { "max_cer": 0.10 },
  "diff_defaults":  { "thresholds": { "*": 0.999 } },
  "models": [
    {
      "name": "got-ocr2",
      "engine": "got-ocr2",
      "gguf": { "repo": "cstr/…-GGUF", "file": "…-q4_k.gguf", "revision": "<sha>" },
      "sample": "tests/regression/images/fox.png",
      "expected_text": "The quick brown fox …",   // null = not captured yet
      "match": { "max_cer": 0.10 },
      "ocr_args": [],                                // extra CLI flags (optional)
      "diff": {                                      // optional
        "binary": "test-got-ocr-diff",
        "ref": { "repo": "…-GGUF", "file": "…-ref-full.gguf", "revision": "main" },
        "thresholds": { "*": 0.995 }                 // global floor, or per-stage
      }
    }
  ]
}
```

`expected_text: null` means "not captured yet" — the key is kept so gaps are
visible. Seed it via the rebake workflow below.

## Rebake → validate workflow (adding / updating a model)

New models start `expected_text: null`. To capture a proven-correct output:

```bash
# GPU (Kaggle) or a machine with the model downloaded:
python tests/regression/run_one.py --name <model> --rebake
```

`--rebake` prints the captured OCR text instead of asserting. **Eyeball it**,
confirm it's correct, then paste it into `manifest.json` as `expected_text`
(with an `expected_text_source` note: date/commit/backend). Thereafter the
nightly / Kaggle run validates against it.

To enable the exact diff check for a model, dump its reference GGUF
(`tools/dump_<model>_reference.py` → a `-ref.gguf` with the captured stages) and
upload it to the model's HF repo as `<model>-ref.gguf`. The driver picks it up
automatically on the next run.

## Running

```bash
# one model (downloads GGUF from HF into $REGRESSION_WORK)
BUILD_DIR=build python tests/regression/run_one.py --name got-ocr2

# whole portfolio
BUILD_DIR=build python tests/regression/run_one.py --all

# force CPU backend (portable, matches CI)
GOT_OCR_FORCE_CPU=1 BUILD_DIR=build python tests/regression/run_one.py --name got-ocr2
```

Binaries: `crispembed` and `test-*-diff` are found under `$BUILD_DIR`
(override with `CRISPEMBED_BIN` / `DIFF_BIN_DIR`).

## CI (`.github/workflows/regression.yml`)

Tiered, mirroring CrispASR:

- **Tier 0 — smoke** (PR, <2 s, no network/binary): `test_driver_smoke.py`
  validates the manifest schema, the diff parser, the CER/normalisation, and the
  garbage guard. A malformed manifest fails here before anything heavy runs.
- **Tier 1 — preflight** (PR, ~5 s, HF API only): HEAD-check every pinned HF
  artifact the manifest references, so a dead pin is caught without downloading.
- **Tier 2 — full** (nightly / dispatch): build + download + run a CPU-only
  subset of the portfolio end-to-end.

The full GPU portfolio (all models, larger images, timing) runs on Kaggle:
`tools/kaggle/ocr-portfolio-regression/`.

## Design notes

- **Lenient by design.** OCR output varies slightly across builds/backends;
  byte-exact locking (as in CrispASR's ASR transcripts) would be too brittle
  here. The garbage guard + CER threshold + optional exact diff give three
  independent nets at increasing strictness.
- **Diff is opt-in.** We don't require a reference dump for every model to get
  value — the garbage guard + text match already catch the class of bug that
  motivated this suite. Adding a `-ref.gguf` upgrades a model to exact per-layer
  cosine checking.
- **Pin GGUF revisions.** For third-party or re-quantised weights, pin a SHA so
  a silent upstream change doesn't masquerade as a code regression. Our own
  repos may use `"main"` when we want to test current code against current
  shipped weights.
