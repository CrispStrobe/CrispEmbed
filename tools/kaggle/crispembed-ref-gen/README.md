# CrispEmbed reference-generation batch (Kaggle)

Generates the missing per-stage reference GGUFs so the regression manifest's
diff step auto-enables for engines that currently have a `test-*-diff` harness
but no reference on HF.

## What it does (per engine)
download upstream source → `tools/dump_<engine>_reference.py` → `<engine>-ref.gguf`
→ build+run `test-<engine>-diff` to VERIFY (cos on the output/final stage)
→ on PASS, upload the ref to the engine's HF GGUF repo.

## Kaggle regime (per ../../../../kaggle_usage.md)
- Runs as **chr1s4**. `enable_gpu:"true"` (also guarantees internet — CPU workers get none),
  `enable_internet:"true"`, all booleans are strings.
- **Both per-account datasets**: `chr1s4/crispasr-hf-token` (HF upload creds) +
  `chr1s4/crispembed-ccache` (warms the CUDA build of the diff harnesses, ~20→3 min).
- Heartbeat + progress via `kaggle_harness` (`init_progress`, `build_heartbeat`);
  `kaggle_harness.py` is bundled as the clone-fallback.
- HF token via `kh.resolve_hf_token()`; does **not** `pip install torch`.
- Writes `progress.txt` + `ref_gen_results.json` to `/kaggle/working`
  (kernels_output does not capture logs).

## Push
```bash
cd tools/kaggle/crispembed-ref-gen
export KAGGLE_API_TOKEN=KGAT_...chr1s4...
python -m kaggle kernels push -p .
# status / output:
python -m kaggle kernels status chr1s4/crispembed-ref-gen
```

## Engine coverage (all sources found + wired 2026-07)
- **HF model id** (dumper loads directly): gliner, lilt, lfm2, lfm2_colbert, layout.
- **HF-hosted `.pth`** (`hf_hub_download`):
  - safmn → `Meloo/SAFMN` / `SAFMN_DF2K_x4.pth` (author's mirror)
  - nafnet → `mikestealth/nafnet-models` / `NAFNet-SIDD-width32.pth`
- **Release URL** (`wget`):
  - esrgan → `github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth`
- **bert_ner:** no `tools/dump_bert_ner_reference.py` exists — write one first (not in this batch).

## After refs land on HF
Add a `diff_only` entry per engine to `tests/regression/manifest.json` (mirror the
restormer / SR entries) so the suite runs them automatically. The `test-nafnet-diff`
harness already exists (committed 9045f62); it just needs `nafnet-ref.gguf`.

## Caveats
Per-engine recipes (dumper `--model` source, diff invocation) are best-effort and
unverified on Kaggle — expect a push→check→fix cycle for the fiddly ones
(layout's source model id, the gliner GLINER_DIFF_REF env path). The framework
(clone/build/ccache/heartbeat/token/upload, per-engine try/except/continue,
results JSON) is solid; failures are isolated per engine.
