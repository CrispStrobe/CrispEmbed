# CrispEmbed regression + benchmark suite

Pinned, reproducible parity + smoke + perf coverage for the CrispEmbed engine
fleet. Mirrors CrispASR's `tests/regression/` but adapted to CrispEmbed's
per-engine `test-<engine>-diff` harnesses, the `crispembed` CLI, and the
~140-model CLI registry.

## Files

| file | role |
|---|---|
| `manifest.json` | source of truth: one entry per backend, pinning the GGUF (repo + revision SHA + file) and per-tier check config |
| `run_one.py` | runs one entry / a tier locally; also a library (`parse_diff_stdout`, `evaluate_diff`, `run_diff`, `run_smoke`, `hf_download`) the Kaggle kernel imports |
| `sample.bmp` | 64×64 test image for image/OCR/SR/detect smoke checks |
| `../../tools/gen_manifest_entries.py` | generates the bulk of `manifest.json` from `examples/cli/model_mgr.cpp` |
| `../../tools/audit_diff_coverage.py` | writes `docs/diff-harness-coverage.md` (harness × dumper × ref) |
| `../../tools/benchmark_all_engines.py` | local perf sweep |
| `../../tools/kaggle/crispembed-regression/` | full validate/rebake on Kaggle |
| `../../tools/kaggle/crispembed-benchmark/` | full perf sweep on Kaggle |

## Tiers

| tier | what runs | pass condition |
|---|---|---|
| `diff` | `test-<engine>-diff <model> <ref>` | every parsed stage `cos_min ≥ threshold` (`diff_thresholds`: `"*"` = global default + per-stage overrides; `require_stages` must be present) |
| `smoke` | the `crispembed` CLI on a fixed input | `embedding` (finite, norm>0, optional `dim`) **or** `nonempty` (exit 0 + non-empty stdout — "it loaded and produced output") |
| `skip` | nothing | tracked-but-not-run; `skip_reason` says why (no standalone CLI mode, etc.) |

`diff` is gold-standard parity and needs a frozen `*-ref.gguf`. `smoke` is the
breadth tier — the confirmation that an engine loads and emits sane output —
and covers ~all CLI-runnable backends. Promote `smoke → diff` once a ref is
baked (see "Baking refs" below).

## How the manifest is built

Most entries are **generated** from the CLI model registry:

```bash
python tools/gen_manifest_entries.py            # rewrite manifest, pin SHAs from HF
python tools/gen_manifest_entries.py --print     # show modality classification only
python tools/gen_manifest_entries.py --no-pin    # use "main" instead of HF SHA lookup
```

It parses `examples/cli/model_mgr.cpp`, classifies each model by modality
(embedding / reranker / sparse / colbert / ner / vision-embed / ocr / math-ocr /
sr / denoise / restore / face / detect / layout / pix2struct), and emits a
`smoke` (+ `bench`) entry with the right `crispembed` invocation and a
conservative `expect`. Modalities with no standalone CLI (LID, punctuation,
LiLT, bare text-detectors) become `tier: skip` with a reason rather than a
guessed command. The merge is **idempotent**: hand-written ("curated") entries
— the `diff` tier and anything without a `modality` field — are preserved;
generated entries (which carry `modality`) are rebuilt each run.

> The generated `smoke` invocations are **best-effort**. The Kaggle sweep is
> what validates them empirically — an entry that errors reveals a wrong flag
> to fix in the modality dispatch table (`build_entry` in the generator).

## Running locally

```bash
# Build the CLI + the diff binaries
cmake --build build --target crispembed-cli test-pan-diff test-instructir-diff \
    test-hat-diff test-granite-vision-diff -j

python tests/regression/run_one.py --dry-run            # validate pins, no downloads
python tests/regression/run_one.py --backend pan --build-dir build
HF_HOME=/tmp/ce-hf python tests/regression/run_one.py --tier diff  --build-dir build
HF_HOME=/tmp/ce-hf python tests/regression/run_one.py --tier smoke --build-dir build
python tools/benchmark_all_engines.py --build-dir build --backend bge-small-en-v1.5
```

`run_one` needs `huggingface_hub` + a token (`~/.cache/huggingface/token` or
`$HF_TOKEN`); point `HF_HOME` at a writable cache.

## Running the full sweep (Kaggle)

The fleet is too big/heavy for local iteration (granite q8_0 ~2.7 GB, 140 models
cumulative). The kernels build CrispEmbed, run the whole manifest, and push
**each engine's verdict to the `cstr/crispembed-kaggle-progress` HF dataset
after every step** — so a crashed/timed-out kernel still leaves every finished
result on HF (`results/` for validate, `benchmarks/` for perf, `runs/` for the
rolling progress log).

```bash
export KAGGLE_API_TOKEN=<chr1s4 token>      # CrispEmbed kernels run under chr1s4
kaggle kernels push -p tools/kaggle/crispembed-regression   # validate (default)
kaggle kernels push -p tools/kaggle/crispembed-benchmark    # perf sweep
```

Kernel knobs (env, read inside the kernel):
- `CRISPEMBED_REGRESSION_MODE` = `validate` (default) | `rebake`
- `CRISPEMBED_REGRESSION_TIER` = `diff` | `smoke` | "" (all)
- `CRISPEMBED_REGRESSION_BACKENDS` = csv filter (chunk the sweep)
- `CRISPEMBED_REF` = branch/SHA (defaults to `regression-suite` until merged)
- `CRISPEMBED_REGRESSION_BUILD` = `cpu` (default) | `cuda`

Builds warm from `chr1s4/crispembed-ccache` (seeded from a prior run; the kernel
re-exports `ccache.tar` and pushes it to HF each run for refresh). Per-engine HF
cache cleanup keeps a 140-model sweep inside Kaggle's ~20 GB scratch.

A non-zero exit (red kernel) means a regression: the validate kernel
`sys.exit(n_fail)`, so "ERROR" status with a `SUMMARY ok=N/M` line is an
intentional regression signal, **not** a crash.

## Baking refs (promote smoke → diff)

See [`BAKING_REFS.md`](BAKING_REFS.md). In short: bake `<engine>-ref.gguf` with
`tools/dump_<engine>_reference*.py`, upload it to that model's GGUF repo under
`diff-harness-ref/`, then add a `diff`-tier `ref` + `diff_thresholds` to the
entry (replacing its `smoke` block) and re-run `tools/audit_diff_coverage.py`.

## Adding / fixing an entry

- **New model in the registry** → re-run `gen_manifest_entries.py`.
- **A generated invocation is wrong** (sweep shows it erroring) → fix the
  modality branch in `gen_manifest_entries.py::build_entry`, regenerate.
- **Promote to diff** → bake a ref (above), edit the entry by hand (it loses
  its `modality` field, so it's preserved as curated on regenerate).
