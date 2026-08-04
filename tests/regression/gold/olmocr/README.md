# olmOCR-2 toolkit reference transcripts

Document-level gold for the native olmOCR lane (`src/qwen2vl_ocr.cpp`, engine
id 18), produced by `allenai/olmOCR-2-7B-1025` under the olmOCR toolkit's own
page-request contract.

Two files per fixture, and both matter:

| file | contents |
|---|---|
| `<stem>.raw.txt` | the raw model output, **YAML front matter included** |
| `<stem>.txt` | `natural_text` as the toolkit's `FrontMatterParser` extracts it |

The native lane strips the front matter itself, so the raw file is what a
byte-level comparison of the lane's own stripping should be read against; the
parsed file is what a CER gate should score.

`contract.json` and the per-corpus `manifest.json` pin what produced these —
model id and resolved revision, the prompt string and its sha256, the applied
chat template, the render geometry per fixture, sampling params, dtype, serving
stack and hardware. If any of those change, the transcripts are stale.

## What makes these gold rather than "some model output"

Nothing here re-implements the request. The kernel imports
`olmocr.pipeline.build_page_query` and `olmocr.prompts.build_no_anchoring_v4_yaml_prompt`,
so the prompt, the message order, the page render and the sampling parameters
are the toolkit's own code executing — a source reading can be wrong about any
of those, an import cannot.

The toolkit consumes PDFs, so each fixture was wrapped losslessly:

```bash
img2pdf --output <stem>.pdf <fixture image>     # img2pdf 0.6.3
```

A PNG becomes a Flate stream and a JPEG passes through as DCT, so no resampling
happens before the toolkit's own `pdftoppm` render at 1288 px on the longest
side. Small fixtures are therefore **upscaled** (626x188 -> 1288x387), not only
large scans downscaled.

## Determinism — read this before writing a gate

`build_page_query` sets `temperature: 0.0`, and `try_single_page` immediately
overwrites it with `TEMPERATURE_BY_ATTEMPT[attempt]`, whose element 0 is
**0.1**. The toolkit's own first attempt is sampling, not greedy.

This gold is greedy, so it is reproducible. A second pass at the toolkit's own
0.1 was run over the same 25 pages: it agrees byte-for-byte on **20/20** synth
fixtures and **2/5** cc0 fixtures, the three that differ at CER 0.00106–0.00637
against this gold. So: gate the native lane on CER against these files, never
on byte-identity with output from the toolkit's default configuration.

All 25 pages passed the toolkit's validity rules on attempt 0 in both passes.
No page here was produced at temperature > 0, none needed a rotation retry, and
none is a `pdftotext` fallback.

## Coverage

| corpus | fixtures | transcripts | mean CER | mean CER (markup stripped) |
|---|--:|--:|--:|--:|
| `synth/` | 20 | 20 | 0.0 | 0.0 |
| `cc0/` | 5 | 5 | 0.17735 | 0.04057 |

The two CER columns differ because the prompt **instructs** the model to convert
tables to HTML and to label figures with markdown image syntax, while the ground
truth is plain text. `commons_example_receipt.png` comes back as a `<table>`,
word for word correct, and plain-text CER charges that at 0.70 (0.00983 with the
markup stripped). The primary column keeps the markup because that is what the
arm returns to a caller; the stripped column is a diagnostic, and not a free one
— on `receipt_historical.png` stripping removes real characters and makes CER
slightly *worse* (0.05469 -> 0.06120). Quote both.

Per-fixture numbers, including which pages the toolkit's own temperature makes
non-reproducible, are in
`tests/results/ocr_parity_olmocr_2026-08-04.json`.

## Known deviations from the toolkit

* **Serving layer.** `olmocr[gpu]` pins `vllm==0.11.2`; the gold here was
  generated through `transformers==4.57.3` (the version that same extra pins).
  Only the serving layer differs — the request is the toolkit's own. vLLM
  *does* run on this hardware, and `vllm-serving/` holds its transcripts; see
  the section below for how far the two stacks agree, which is the single most
  important number in this directory.
* **Checkpoint.** The toolkit defaults to `allenai/olmOCR-2-7B-1025-FP8`, whose
  quantised kernels need sm_89+. The unquantised sibling ran instead.
* **dtype.** fp16, not the checkpoint's bf16: the reference cards are compute
  capability 7.5, where torch emulates bf16. Same open follow-up as the
  Qwen2.5-VL arm — spot-check on Ampere-or-newer before treating a byte-level
  difference against this gold as a lane bug.

## Regenerating

See `tools/kaggle/olmocr-parity/README.md`. The dev Mac cannot host it: 15.45
GiB of 16-bit weights against 16 GiB of total unified memory.

## `sampled-t0.1/`

The same 25 pages at the toolkit's own attempt-0 temperature, kept so the
determinism claim above is checkable rather than asserted. It is **not** gold:
it is what the reference produces by default, sampled, and it is the reason a
gate must be a CER threshold and not a diff.

## `vllm-serving/` — the same request through the toolkit's own server

`vllm==0.11.2` serving `allenai/olmOCR-2-7B-1025` on the same 2x T4 pair, fp16,
tensor-parallel 2, `temperature=0.0` — greedy, like the gold, and the same
request object. It is faster (median 3.55 s/page synth against 5.9 s; 20.5 s
against 40.4 s on `commons_example_receipt.png`).

**Two greedy decodes of the same weights do not agree on document pages.**

| corpus | byte-identical to the gold |
|---|--:|
| synth | 20/20 |
| cc0 | 1/5 |

The three small ones are cosmetic (CER 0.00101–0.02423 against the gold). The
fourth is not: on `simple_form.png` the vLLM decode emits an early stop token
and returns about half the page — five of the ten blocks — with
`finish_reason: stop`, so nothing flags it as truncated. Against ground truth
that is CER 0.37247 where the transformers decode scores 0.04858. The gold ships
from the transformers pass for exactly this reason: it is complete on every page
and reproducible.

The number that matters for the native lane: **the reference does not agree with
itself across serving stacks at greedy**, so any lane gate must be a CER
threshold with headroom for this, and a byte diff against either set is
meaningless.

`probe.json` records how the engine was brought up, including the two
environment fixes it needs on this hardware — see below.

### Getting vLLM up on Turing (sm_75)

Four measured failures, only one of them about the platform:

1. Probing inside the gold kernel — *"Free memory on device (7.14/14.56 GiB) ...
   less than desired GPU memory utilization"*. The transformers model was still
   resident in the same process.
2. `VLLM_WORKER_MULTIPROC_METHOD=spawn` with `torch.cuda` touched first — the
   parent's CUDA context forces spawn, and a spawned worker re-imports a Kaggle
   script kernel top to bottom.
3. FlashAttention-2 needs compute capability >= 8.0, so vLLM falls back to
   FlashInfer, which JIT-builds its prefill kernels and links `-lcuda`. Kaggle
   ships the driver as `libcuda.so.1` with no linker symlink, so ninja dies with
   `cannot find -lcuda` after ~4 minutes of nvcc. **Fix: symlink it.**
4. Forcing a backend to dodge the JIT fails on *every* backend sm_75 offers —
   `Qwen2.5-VL does not support AttentionBackendEnum.{FLASHINFER,TRITON_ATTN,
   FLEX_ATTENTION} backend now` — because `VLLM_ATTENTION_BACKEND` also forces
   the vision tower. **Fix: leave it unset** and let the ViT choose.

With the symlink in place and the variable unset, the engine starts in 356 s and
reads all 25 pages.
