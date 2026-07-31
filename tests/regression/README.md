# CrispEmbed OCR portfolio regression suite

## Real-world public-domain robustness fixtures

The checked-in corpus now has a small seed set under `images/cc0/` for cases
that synthetic fixtures do not cover: receipts, a historical receipt scan,
Arabic printed/handwritten text, a handwritten letter, and a form.  Sources,
license declarations, URLs, and SHA-256 checksums are recorded in
`images/cc0/MANIFEST.json`; the source catalog is
`cc0_sources.json`.  Fetch or refresh them with:

```sh
python3 tests/regression/fetch_cc0_fixtures.py
```

Stage coverage and the distinction between real-world and deterministic
reference inputs are recorded in `corpus_manifest.json`.

The complete engine inventory—including engines whose runtime exists but whose
GGUF still needs downloading or porting—is in `ocr_engine_matrix.json`.
`ocr_engine_benchmark.py --download-missing` resolves manifest-pinned GGUFs
from their Hugging Face repositories instead of treating an empty local cache
as lack of support.

For large GGUFs on an external volume, add `--mmap` to use the no-copy loader
when the engine supports it:

```sh
python tests/ocr_engine_benchmark.py --only unlimited-ocr-stacked \
  --gpu-backend metal --mmap --repeats 1 --timeout 300
```

The seed set now also includes German public-domain/CC0 inputs: an 1848 Berlin
citizenship document, a German official-print page, and German Kurrent
handwriting.  These are sourced from Wikimedia Commons and are tracked with
the same checksum/license metadata.  A larger German historical OCR source is
the CC0/public-domain [German PD Newspapers dataset](https://huggingface.co/datasets/storytracer/German-PD-Newspapers),
which should be sampled rather than vendored wholesale.

The seed set is intentionally not treated as gold transcription data until a
human verifies each transcription.  It is suitable immediately for live
robustness, preprocessing, orientation, layout, and language-routing checks.
Rotated and skewed derivatives may be generated from these public-domain
images without adding third-party licensing obligations.  The next expansion
should use the CC0 ExpressExpense receipt set (200 real restaurant receipts)
and the CC0 Arabic Documents OCR set (10K images with page/text annotations),
but those sources require a separate download/acceptance step and are therefore
not vendored by default.

To measure preprocessing independently on the real corpus, use the live
preprocessor harness. It materializes each cleanup variant, records checksums,
dimensions, pixel statistics, and timing, and can feed every variant through
the detector/recognizer pipeline:

```bash
python3 tests/ocr_preprocessor_benchmark.py \
  --output /tmp/ocr-preprocessors.json \
  --pipeline-binary build/crispembed \
  --det /Volumes/backups/ai/crispembed-gguf/dbnet-ic15-f16.gguf \
  --rec /Volumes/backups/ai/crispembed-gguf/trocr-small-printed-q8_0.gguf
```

Use `--include-dewarp` for the classical dewarper and repeat `--model
STAGE=MODEL` for an explicitly selected learned stage, for example
`--model nafnet=/path/nafnet-sidd-w32-q8_0.gguf`. Missing fixtures or models
are reported as explicit rows; they are not treated as passing quality
results.

Generate auditable problematic-input variants with:

```bash
python3 tests/regression/generate_problematic_variants.py \
  --output-dir tests/regression/images/derived \
  --source german_official_document.jpg \
  --source receipt_example.png
```

The generated `MANIFEST.json` records each parent checksum and exact recipe;
the derived images inherit the parent CC0/public-domain status and are not
gold transcriptions.

The latest CPU preprocessor-only run across 17 fixtures is checked in at
`tests/regression/results/ocr_preprocessors-20260731.json`: 136 rows, 129
successful stages, and 7 explicitly `unavailable` dewarp rows where the input
does not contain enough text-line geometry. No stage returned an unexpected
runtime error.

The harness and provenance generator have a dependency-light unit guard:

```bash
python3 tests/regression/test_preprocessing_tools.py
```

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
      "sample": "tests/regression/images/fox.png",   // local image, OR use sample_hf (below)
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

**`sample_hf` (license-restricted fixture images).** When the only in-domain
test image is under a license we can't bundle into this MIT/Apache repo (e.g.
CROHME handwritten-math = CC-BY-NC-SA), replace `sample` with a `sample_hf`
block that extracts one image from an HF **dataset** parquet at test time (like
the GGUFs, the image is fetched from its original source and never committed):

```jsonc
"sample_hf": {
  "dataset": "Kitajiang/test2_CROHME2014",
  "file": "CROHME2014/test-00000-of-00001.parquet",
  "revision": "<commit-sha>",          // PIN it so the row is stable
  "row": 23,                            // integer index into the parquet
  "image_column": "image",
  "label_column": "latex_formula",      // optional sanity gate:
  "expect_label": "\\[C_{t}=C+C=2C\\]"  // fail loudly if the row shifts
}
```

Needs `pandas` in the test env (present on Kaggle + the conda base). Pick a row
the model reads correctly + deterministically (CPU==Metal) so the pinned
`expected_text` is a real guard, and attribute the source dataset in `_comment`.

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
