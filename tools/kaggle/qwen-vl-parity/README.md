# Qwen2.5-VL OCR parity arm (Kaggle GPU)

Runs `tests/ocr_external_parity.py --qwen` over both parity corpora and brings
back the per-fixture transcripts that serve as the CER gate for the native VL
lane (`src/qwen2vl_ocr.cpp`).

## Why not on the dev Mac

The reference checkpoint is `Qwen/Qwen2.5-VL-7B-Instruct`: 16 584 414 560 bytes
of safetensors, i.e. **15.45 GiB of 16-bit weights**, against **16 GiB of total
unified memory** on the dev machine. There is no arrangement in which the
weights plus a page's vision tokens fit, so a local run would be measuring swap,
not the model. The 3B sibling does fit (7.0 GiB, ~7.4 s/page on MPS) and is used
to smoke-test the script — but it ships under a research licence and its numbers
are **never** quotable as the reference.

## Hardware handling

Kaggle hands out one 16 GB card or two 15 GB cards at random, and neither fits
15.45 GiB of weights comfortably alone. Placement therefore goes through
accelerate (`--qwen-device-map auto`). dtype follows compute capability: bf16
needs Ampere (8.0+), and on an older card torch accepts the dtype and then runs
an emulated path, so the kernel picks fp16 explicitly and records what it used.

Quality (CER/WER) transfers across hosts; **timing does not**. The harness JSON
records the hardware, and GPU timings from this kernel must never be quoted next
to Mac timings.

## Run

```bash
export KAGGLE_API_TOKEN=<chr1s4 token>   # see ../../../../kaggle_usage.md
python -c "from kaggle import KaggleApi; a=KaggleApi(); a.authenticate(); \
    print(a.kernels_push('tools/kaggle/qwen-vl-parity'))"
```

Datasets attached: `chr1s4/crispasr-hf-token` (HF rate limits) and
`chr1s4/crispembed-ocr-synth`. The synthetic corpus is shipped as a dataset
rather than regenerated in-kernel because `tests/ocr_synth_corpus.py` renders
from macOS system fonts — regenerating it on Linux would produce different
pixels, and the transcripts would no longer be gold for the fixtures the lane
actually runs on. The CC0 fixtures come with the repo clone.

## Outputs (`/kaggle/working`)

| file | contents |
|---|---|
| `parity_{synth,cc0}.json` | full harness result incl. `summary` and `vl_manifest` |
| `parity_{synth,cc0}.md` | rendered comparison table |
| `gold/{synth,cc0}/*.txt` | one transcript per fixture |
| `gold/{synth,cc0}/manifest.json` | model id, revision, prompt, decoding, hardware |
| `summary.json` | hardware/dtype/versions + both aggregates |
| `run.log` | full tee'd log (`kernels_output` does not expose stderr) |
