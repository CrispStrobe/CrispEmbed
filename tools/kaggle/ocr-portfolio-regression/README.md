# OCR Portfolio Regression (Kaggle GPU)

GPU counterpart to the CPU nightly in `.github/workflows/regression.yml`.
Builds CrispEmbed `main` with CUDA and runs the full OCR portfolio through
the shared driver `tests/regression/run_one.py`, one model per subprocess,
then prints a pass/fail summary + writes `results.json`.

Per model it checks: no-garbage guard, lenient CER text match vs the pinned
`expected_text`, and (if a `<model>-ref.gguf` exists on HF) the exact
`test-<model>-diff` per-layer cosine. See `tests/regression/README.md`.

## Run

```bash
# push + run (needs the crispasr-hf-token dataset attached for HF auth)
kaggle kernels push -p tools/kaggle/ocr-portfolio-regression
```

## Rebake (capture expected_text for new models)

Models with `expected_text: null` only get the garbage guard until a
proven-correct output is pinned. To capture:

```
# set the kernel to pass --rebake (edit argv or CRISPEMBED args), run,
# read the CAPTURED OUTPUT for each model in the log, eyeball it, and
# paste into tests/regression/manifest.json as expected_text.
```

`--rebake` never fails the run — it prints, it doesn't assert.
