# olmOCR toolkit parity arm (Kaggle GPU) — brief A3

Captures the olmOCR toolkit's **page-request contract at runtime** and produces
**document-level gold** for both parity corpora, for the native olmOCR lane
(`src/qwen2vl_ocr.cpp`, engine id 18) to be measured against.

## What "captured at runtime" means here

Nothing in this kernel re-implements the request. It imports
`olmocr.pipeline.build_page_query`, so the prompt string, the message order,
the render geometry and the sampling parameters are whatever the installed
toolkit actually does — a source reading can be wrong, an import cannot. The
result lands in `contract.json`, which is the reference the native lane's
prompt/preprocess contract is checked against.

## The PDF wrap

The toolkit consumes PDFs, the fixtures are images. Each fixture becomes a
single-page PDF:

```bash
img2pdf --output <stem>.pdf <fixture image>
```

That embed is lossless — a PNG becomes a Flate stream, a JPEG passes through as
DCT — so no resampling happens before the toolkit's own render. The toolkit
then rasterises the page itself with
`pdftoppm -png -r (1288*72/longest_mediabox_point)`, which means small fixtures
are **upscaled** to 1288 on the longest side just as large scans are
downscaled. The exact per-fixture render dimensions are in `contract.json`.

## Serving layer (a deviation, recorded)

The toolkit drives a vLLM OpenAI-compatible server. `olmocr[gpu]` pins
`vllm==0.11.2`, which is V1-only, and Kaggle's cards are Turing (sm_75).
Generation therefore runs through `transformers` (pinned to `4.57.3`, the
version `olmocr[gpu]` pins) while the *request* is still built by the toolkit's
own code. Whether vLLM would have started at all is measured, not assumed: a
probe runs in a subprocess **after every artifact is on disk**, so it cannot
cost anything, and its outcome is written to `vllm_probe.json`.

The checkpoint is also a deviation: the toolkit defaults to
`allenai/olmOCR-2-7B-1025-FP8`, whose quantised kernels need sm_89+, so the
unquantised `allenai/olmOCR-2-7B-1025` runs instead.

## Two passes, and why gold is greedy

`build_page_query` sets `temperature: 0.0`, and `try_single_page` then
overwrites it with `TEMPERATURE_BY_ATTEMPT[attempt]` — attempt 0 included. The
toolkit's *first* attempt is therefore **sampling at 0.1**, not greedy.

* `gold/` — attempt 0 greedy (`do_sample=False`), so the gold set is
  reproducible. If a page comes back invalid (front matter unparseable,
  truncated, or rotation-invalid) the toolkit's own retry ladder runs, at the
  toolkit's own temperatures; those pages are flagged and are not deterministic
  gold.
* `sampled/` — attempt 0 exactly as the toolkit issues it (temperature 0.1,
  `top_p=1.0`, `top_k` disabled). The gap between the two passes is the size of
  the non-determinism the toolkit ships with.

## Run

```bash
export KAGGLE_API_TOKEN=<chr1s4 token>   # see ../../../../kaggle_usage.md
python -c "from kaggle import KaggleApi; a=KaggleApi(); a.authenticate(); \
    print(a.kernels_push('tools/kaggle/olmocr-parity'))"
```

Datasets attached: `chr1s4/crispasr-hf-token` and `chr1s4/crispembed-ocr-synth`.
The synthetic corpus ships as a dataset rather than being regenerated because
`tests/ocr_synth_corpus.py` renders from macOS system fonts; regenerating it on
Linux would produce different pixels and the transcripts would no longer be gold
for the fixtures the lane runs on. The CC0 fixtures come with the repo clone.

## Outputs (`/kaggle/working`)

| file | contents |
|---|---|
| `contract.json` | prompt + sha256, request skeleton, rendered chat prompt, per-fixture render and post-`smart_resize` dims, sampling params, versions |
| `gold/<corpus>/<stem>.raw.txt` | raw model output, front matter included |
| `gold/<corpus>/<stem>.txt` | parsed `natural_text` |
| `gold/<corpus>/pages.json` | per-page attempts, temperatures, token counts, seconds |
| `gold/<corpus>/manifest.json` | model id + revision, prompt, render, dtype, serving stack, hardware, img2pdf recipe |
| `sampled/<corpus>/…` | the same, for the temperature-0.1 attempt-0 pass |
| `summary.json` | versions, placements, per-page attempt/timing summary |
| `vllm_probe.json` | vLLM-on-Turing outcome |
| `run.log` | full tee'd log (`kernels_output` does not expose stderr) |

Timing in these files is **generation seconds on a Kaggle T4 pair**, recorded
for context only. It is not a latency measurement and must never be quoted next
to a Mac number.
