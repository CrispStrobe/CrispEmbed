# crispembed-imatrix-t19 (Kaggle, chr1s4)

imatrix quants for the T19 German-retrieval targets: `arctic-embed-m-v2` and the
F2LLM-v2 family (80m / 160m / 330m / 0.6b).

A **thin driver**: all the logic lives in
`../crispembed-imatrix-quant/imatrix_quant.py` and is executed FROM THE CLONE,
so there is one copy of the pipeline and the calibration corpora travel with it.
The driver only (a) resolves the HF token, (b) clones this branch, (c) sets
`MODELS` / `FORCE`, (d) `exec`s the shared script.

```bash
cd tools/kaggle/crispembed-imatrix-t19
KAGGLE_API_TOKEN=<chr1s4 token> python -c \
  "from kaggle import KaggleApi; a=KaggleApi(); a.authenticate(); print(a.kernels_push('.'))"
```

Runs under **chr1s4** and attaches the chr1s4 copies of both datasets —
cross-account attach is rejected (`kaggle_usage.md` #13). `enable_gpu` is true
only because Kaggle CPU workers get no internet (#3); the build is CPU-only.

## Two traps this kernel was built to survive

1. **A script kernel ships only its `code_file`** (#26/#19). Bundled siblings
   are unreadable at runtime, which is why the earlier
   `crispembed-imatrix-quant` runs silently calibrated on a 10-sentence English
   fallback instead of `calib_corpus.txt`. Everything real is read from the git
   clone, and a missing corpus now raises instead of falling back.
2. **The token dataset may mount ONLY under the long path.** Run 1 completed the
   whole pipeline for all five models in 21 minutes and then lost every artifact
   to `401 RepositoryNotFound` on each upload, because
   `kaggle_harness.resolve_hf_token()` does not glob
   `/kaggle/input/datasets/<acct>/<slug>/` — the layout that worker actually
   had (its log: `HF auth: /kaggle/input contains 1 entries: ['datasets']`,
   then `hf_token_ok: False`). The ccache warm globs that path and succeeded on
   the same run, which is what made the asymmetry visible. The driver now globs
   both layouts, exports `HF_TOKEN` (which `resolve_hf_token` consults first),
   and **aborts up front** if no token is found rather than burning the quota.

## Naming

New artifacts use the canonical `-q4_k-imatrix.gguf` / `-iq4_xs.gguf` names.
`f2llm-v2-0.6b` is the exception: it already ships those two files and their
SHA256 is **pinned in `examples/cli/model_hashes.h`**, so overwriting them would
break the pin for existing users. Its re-calibration publishes under `-c2`
names and its `.imatrix` / A/B summary under a `-c2` meta prefix.

Results and the verdict live in `PLAN.md` → **T19-E3 status**.
