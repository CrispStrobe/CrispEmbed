# DeepSeek-OCR-2 reference gold (brief A4)

Reference transcripts from `deepseek-ai/DeepSeek-OCR-2`, revision
`aaa02f3811945a91062062994c5c4a3f4c0af2b0`, produced by the checkpoint's **own**
`modeling_deepseekocr2.py::infer()` called with the model card's arguments.

**This is the set T14 needs.** `src/deepseek_ocr2.cpp` loads DeepSeek-OCR-2, so
these are the transcripts a native-lane CER row must be measured against. The
sibling `../deepseek-ocr/` is the brief's literal reference (DeepSeek-OCR v1)
and is kept for completeness, not as a gate.

Layout, contract fields, deviations and reproduction commands are identical in
structure to `../deepseek-ocr/README.md`; only the values below differ.

## The contract, as captured at runtime

| | |
|---|---|
| plain-OCR prompt | `<image>\nFree OCR. ` |
| document prompt | `<image>\n<|grounding|>Convert the document to markdown. ` |
| applied prompt | the same string minus its trailing space (`plain` template, `.strip()`ed content) |
| token layout | `[bos=0]` + `<image>`×N (id 128815) + the instruction tokens |
| preset | `base_size=1024, image_size=768, crop_mode=True` (the card's dynamic resolution: (0-6)×768×768 + 1×1024×1024) |
| generation | `temperature=0.0`, `max_new_tokens=8192`, `no_repeat_ngram_size=20`, `use_cache=True`, `eos_token_id=1`; greedy (`generation_config` sets no `do_sample`) |
| stop string | `<｜end▁of▁sentence｜>`, stripped, then `.strip()` |

The native lane's assembled prompt — `[bos] + <image>×N + <view_sep> +
tokenize("\nFree OCR.")` in `src/deepseek_ocr2.cpp` — matches the `free_ocr`
contract captured here.

### Image-token accounting, measured

`v2` emits `([image_token_id] * num_queries_base) * num_queries_base + 1` for the
global view, i.e. **257** tokens at `base_size=1024` — note the absent per-row
newline token that v1 still inserts (v1: 17×16 + 1 = 273). Crops add
`(num_queries * w_crops) * (num_queries * h_crops)` with `num_queries = 12` at
`image_size=768`.

Cropping is skipped when both sides are ≤ **768** (v1's threshold is 640). Over
the 25 fixtures v2 takes the single-view path 19 times, v1 12 times. Measured
per fixture, from the tensors handed to `generate`:

| fixture | source | crop ratio | crop tensor | image tokens |
|---|---|---|---|---|
| `commons_example_receipt.png` | 500×650 | 1×1 | `[1,3,1024,1024]` | 257 |
| `commons_test_ocr_document.jpg` | 1920×2485 | 2×3 | `[6,3,768,768]` | 1121 |
| `german_official_print.jpg` | 1920×2518 | 2×3 | `[6,3,768,768]` | 1121 |
| `receipt_historical.png` | 768×1552 | 1×2 | `[2,3,768,768]` | 545 |
| `simple_form.png` | 452×317 | 1×1 | `[1,3,1024,1024]` | 257 |

## Deviations from the model card

Identical to the v1 set: attention forced to `eager` (sm_75 has no FA2 and the
checkpoint's `ATTENTION_CLASSES` has no `sdpa` entry; verified at runtime as
`LlamaAttention`), dtype **not** a deviation (bf16 probed and used), weights
materialised at the target dtype rather than cast after the copy, serving
through `transformers 4.46.3` on `torch 2.10.0+cu128`, and a 240 s per-page
stopping criterion that fired on no page. v2's Qwen2 encoder runs at its own
default `attn_implementation='sdpa'`, untouched.

## Determinism

Greedy and reproducible: a re-run of `synth_00_blur.png` and
`commons_example_receipt.png` came back byte-identical.

## Scores

`tests/results/ocr_parity_deepseek_2026-08-04.json`. Headline (`free_ocr`,
markup-stripped in brackets): synth CER 0.00199 [0.00199] over 20 pages, cc0 CER
0.18743 [0.11063] over 5 pages.
