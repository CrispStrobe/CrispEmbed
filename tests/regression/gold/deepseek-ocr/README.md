# DeepSeek-OCR reference gold (brief A4)

Reference transcripts from `deepseek-ai/DeepSeek-OCR`, revision
`9f30c71f441d010e5429c532364a86705536c53a`, produced by the checkpoint's **own**
`modeling_deepseekocr.py::infer()` called with the model card's arguments.
Nothing in the producing kernel rebuilds the request; it wraps `model.generate`
and records what `infer()` handed it.

Sibling set: `../deepseek-ocr2/` — same fixtures, same prompts, run against
`deepseek-ai/DeepSeek-OCR-2`, which is the checkpoint `src/deepseek_ocr2.cpp`
actually loads and therefore the only one whose CER is a gate for T14.

## Layout

```
contract.json              runtime-captured contract, both checkpoints
runtime.json               attention class, generation_config, determinism re-run
{synth,cc0}/manifest.json  provenance: revision, prompts, infer kwargs, deviations
{synth,cc0}/pages.json     per page, per mode: token counts, tensor shapes, timing
{synth,cc0}/<stem>.free_ocr.txt              plain-OCR mode, raw model output
{synth,cc0}/<stem>.grounding_markdown.txt    document mode, raw (boxes included)
{synth,cc0}/<stem>.grounding_markdown.mmd    document mode after infer()'s own
                                             post-processing (result.mmd)
```

There is no `<stem>.free_ocr.mmd`: `infer()`'s post-processing is a no-op when
the prompt carries no `<|grounding|>`, verified byte-identical on all 50
plain-mode pages before deletion.

## The contract, as captured at runtime

| | |
|---|---|
| plain-OCR prompt | `<image>\nFree OCR. ` |
| document prompt | `<image>\n<|grounding|>Convert the document to markdown. ` |
| applied prompt | the same string minus its trailing space — `format_messages` uses the `plain` conversation template (`SeparatorStyle.PLAIN`, empty roles and separators) and `.strip()`s the content |
| token layout | `[bos=0]` + `<image>`×N (id 128815) + the instruction tokens |
| preset | `base_size=1024, image_size=640, crop_mode=True` (the card's Gundam) |
| generation | `temperature=0.0`, `max_new_tokens=8192`, `no_repeat_ngram_size=20`, `use_cache=True`, `eos_token_id=1`; `generation_config` sets no `do_sample`, so decoding is greedy |
| stop string | `<｜end▁of▁sentence｜>`, stripped, then `.strip()` |

Preprocessing is per fixture and is in `pages.json` as the shapes of the tensors
`infer()` passed to `generate`, not as a formula: `images_ori_shape`,
`images_crop_shape`, `images_spatial_crop`, `n_image_tokens`.

**The crop threshold is hardcoded and differs between the two checkpoints.** v1
skips cropping when both sides are ≤ 640, v2 when both are ≤ 768. So
`commons_example_receipt.png` (500×650) is 6 crops of 640×640 and 903 image
tokens here, and a single 1024×1024 global view with 257 image tokens under v2.
Over the 25 fixtures v1 takes the single-view path 12 times, v2 19 times.

## Deviations from the model card

* **Attention: `eager`, forced.** The card passes
  `_attn_implementation='flash_attention_2'`; the reference host is Turing
  (sm_75), which has no FA2, and the checkpoint's `ATTENTION_CLASSES` offers
  only `eager` and `flash_attention_2` — there is no `sdpa` entry to substitute,
  so this is not a choice between two options. Verified at runtime:
  `config._attn_implementation == 'eager'` and the instantiated decoder
  attention class is `LlamaAttention`, not `LlamaFlashAttention2`. The vision
  towers never used FA2 (v1's ViT is built with `use_flash_attn=False`).
* **dtype: none.** bf16 was probed on the card (matmul, conv2d, SDPA) before
  use and all three passed, so this ran at the card's `bfloat16`. The fp16
  fallback in the kernel never fired.
* **Load order.** The card writes `.eval().cuda().to(dtype)`, which
  materialises 3B parameters in fp32 on the card before casting — 18 GiB peak
  on a 15 GiB T4. Weights are materialised at the target dtype instead; the
  stored weights are bf16 and the fp32 step is exact, so the numbers are the
  same.
* **Serving stack.** `transformers 4.46.3` (the card's pin) on
  `torch 2.10.0+cu128`, single CUDA device. The card's vLLM recipe was not
  used — and it is not equivalent: see below.
* **A 240 s stopping criterion** was injected per page. `max_new_tokens` stays
  at the checkpoint's 8192. It fired on no page.

## Known failure on this corpus

`simple_form.png` under the grounding prompt: 2073 output tokens of a repeating
single-character box list (`F`/`R`/`E`), 6066 raw chars against 247 reference
chars, 89.1 s, raw CER 23.38. The cause is in the contract, not the page — the
transformers path the card documents has no repetition guard equivalent to the
one the card's *own vLLM* snippet installs (`NGramPerReqLogitsProcessor`,
`ngram_size=30`, `window_size=90`, `<td>`/`</td>` whitelist). `infer()`'s
`no_repeat_ngram_size=20` cannot catch it because every repeat carries different
coordinates and so never forms a repeated 20-gram. DeepSeek-OCR-2 degrades on
the same page far less (518 tokens, raw CER 5.41, document-view CER 0.1215).

`receipt_historical.png` under the *plain* prompt is answered with HTML
`<table>` rows: CER 2.0417 against 0.1094 markup-stripped. Read both columns.

## Reproducing

Scores are in `tests/results/ocr_parity_deepseek_2026-08-04.json`; the kernel is
`tools/kaggle/deepseek-ocr-parity/`. To rescore without re-running the GPU:

```
python tests/score_gold_transcripts.py \
    --images ~/crispembed-ocr-synth \
    --gold tests/regression/gold/deepseek-ocr/synth \
    --engine deepseek-ocr-free_ocr --suffix .free_ocr.txt --strip-markup \
    --output /tmp/ds_synth.json
```
