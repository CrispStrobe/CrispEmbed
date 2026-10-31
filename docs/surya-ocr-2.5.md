# Surya OCR 2.5 (`datalab-to/surya-ocr-2.5`) — source contract and port plan

Status (31 October 2026): **native output equals the original model on every
tested crop; not yet faster than the reference, and not qualified for quality
or publication.**
Revision `00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3` is pinned. The retained
evidence describes nine owned crops: six grid-aligned, tested at F32 and F16
both locally and in hosted run 41191185159, and three off-grid crops tested
locally at both precisions. The local off-grid summary does not retain original
token IDs or input hashes, so that part cannot be independently replayed from
the summary alone. For the grid-aligned crops, native ids, decoded text and EOS
step matched `transformers` 5.16.1
`Qwen3_5ForConditionalGeneration` (float32 CPU, greedy, block prompt, vLLM pixel
kwargs). This was checked twice: in the session container
([local](../tests/results/surya25/native-ab-local.json),
[scale_to_fit end to end](../tests/results/surya25/native-scale-to-fit-e2e.json))
and on hosted CI ([run 41191185159](../tests/results/surya25/native-ab-hosted.json)).
The hosted receipt is a summary; the local receipt does not retain GGUF digests,
so cross-host byte identity cannot be verified from the retained files alone.
There is no CrispEmbed registry entry or published CrispEmbed GGUF; Datalab has
published its own llama.cpp GGUFs. Quality on real documents is unmeasured.

The private [Kaggle CPU quantization probe](../tests/results/surya25/kaggle-quant-v2-summary.json)
converted the pinned checkpoint and produced Q8_0 and Q4_K files, but failed
before decoding because the worker lacked the reference font. These temporary
files were not retained; the probe does not qualify either quantized recipe.
The [font-fixed retry](../tests/results/surya25/kaggle-quant-v3-summary.json)
matched the frozen original's token IDs, text and stop on all six synthetic
crops with F16, Q8_0 and Q4_K. All six input hashes matched the frozen inputs,
and no norm or bias was quantized. This is a narrow measurement, not real-scan
quality qualification or a release; its GGUFs were temporary and not retained.
The output confidences differ between precisions. The default-off direct-F16
matmul path passed a [weight-free operator test](../tests/results/surya25/direct-f16-green-v3.json)
that also covers leaf-only dispatch and atomic output.
On six synthetic crops the gate-on arm reproduced the frozen original's ids,
text and stop exactly, and a three-pair timing diagnostic on one case measured
a 4.5x median recognition speedup over the materialising path
([private receipt](../tests/results/surya25/direct-f16-speed-v1.json)). That
receipt predates later source hardening and its direct dispatch is requested,
not observed. An Ovis regression is still outstanding.

Two genuine scanned source crops — a German Imperial Law Gazette line and a
historical receipt line — were then compared against the Python blueprint
([private receipt](../tests/results/surya25/real-scan-v1.json)). Both native
arms matched the blueprint's token IDs, text and stop/EOS on both crops. The
blueprint was still faster: native direct recognition took 7.9-8.0 s against
2.9 s, and 9.9-10.0 s against 5.8 s. Those two numbers have different
boundaries (the Python figure excludes scaling, preprocessing and loading, and
reuses a loaded model), so they are not a like-for-like ratio. No accuracy
metric exists because there is no independently reviewed transcript gold.

Usage (experimental):

```bash
python models/convert-qwen35-ocr-to-gguf.py --model datalab-to/surya-ocr-2.5 \
    --revision 00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3 --profile surya25 --dtype f16 --output surya25-f16.gguf
```

```bash
CRISPEMBED_QWEN35_OCR=1 CRISPEMBED_SURYA25_FRONTEND=1 CRISPEMBED_SURYA25_INFERENCE=1 \
    ./build/test-surya25-run surya25-f16.gguf crop.png 512 4 ["<package prompt>"]
```

Through the C API this is `qwen35_ocr_init` + `qwen35_ocr_recognize_*`; a
null or empty prompt means `OCR this block image to HTML.`.

### Confirmed checkpoint facts (weight-free observation)

- `config.json`: `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`,
  `mtp_num_hidden_layers: 0`. Text and vision dimensions are *identical* to the
  0.8B LightOn/Ovis shapes already implemented natively: 1024 hidden, 24 layers
  (linear×3 + full repeating), 3584 FFN, 8/2 heads of 256, partial rotary 0.25
  with MRoPE [11,11,10], θ 1e7, tied head; vision 12×768, patch 16, merge 2,
  2304 position embeddings, output 1024. **Only the vocabulary differs: 88,526
  ids.** Special ids: `<|endoftext|>`=0 (pad), `<|im_start|>`=1,
  `<|im_end|>`=2 (EOS, the only generation stop), `<|vision_start|>`=9,
  `<|vision_end|>`=10, `<|image_pad|>`=11, `<unk>`=65424, U+0000=1947.
- Header and schema ([receipt](../tests/results/surya25/header-schema.json), hosted
  run 41190926697): 473 BF16 tensors, no MTP tensors, the official
  `Qwen3_5ForConditionalGeneration` schema matches exactly except the tied
  `lm_head`, and the tensor names are the ones the native mapper already uses.
- Single `model.safetensors`, bf16, 1,378,772,208 bytes, LFS sha256
  `6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17`.
  The card licence is `openrail`.
- Tokenizer: WordLevel model of 65,425 entries, mostly single characters, plus
  25,049 added tokens (22,240 non-normalized, 2,809 normalized, 22 special).
  There is no normalizer, the pre-tokenizer is `Split [\s\S]` Isolated, and
  the decoder is `Fuse`. Encoding runs leftmost-longest over the non-normalized
  added tokens, then over the normalized ones, then looks up each remaining code
  point (or `<unk>`). Added tokens such as `.<` and `).<` make HF spell out the
  `<|im_end|>` after every package prompt, and the native encoder reproduces
  this exactly.
- The chat template renders `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n`.
  There is no `<think>` block and no system prompt.
- Processor: Qwen2VL (patch 16, merge 2, mean/std 0.5, bicubic). vLLM serves
  with `min_pixels=3136, max_pixels=6291456`, which changes the grid for small
  crops compared with the processor defaults (`448×112` gives 8×28 rather than
  8×32). The native format-4 path uses the vLLM values.

### Native implementation status

| Piece | State | Evidence |
|---|---|---|
| `src/core/wordlevel_added.h` tokenizer | exact | 20,028 encode cases (prompts, spelled marker, random added-token, character and `<unk>` mixes) and 2,000 decodes identical to HF; the guard fails when the two added-token passes are merged |
| Format-4 frontend (`CRISPEMBED_SURYA25_FRONTEND`) | exact | rendered ids for all 4 package prompts, the empty (block) prompt and a custom prompt equal HF `apply_chat_template` + encode; `=0` and unset are off |
| Image preprocessing (vLLM kwargs) | exact | five block-sized images: identical grids, patch max abs 0.0; a 192-DPI page is rejected (>1024 patches) |
| Converter `--profile surya25` | written | NUL token carried as an explicit id, because gguf string arrays read back as C strings |
| Inference (`CRISPEMBED_SURYA25_INFERENCE`) | exact on six hosted crops | 12 F32/F16 arms match original HF ids/text/EOS; three additional off-grid crops have a local-only summary |
| Package `scale_to_fit` (Pillow LANCZOS, 28-px grid) | exact, on by default for this profile | `core/pil_resample_u8.h` matches Pillow on 32.1M samples; the port is byte-exact against the pinned package function on 13 sizes; `CRISPEMBED_SURYA25_SCALE_TO_FIT=0` passes pixels through |
| Speed | **still slower than HF** | Direct F16 removed the per-step F32 re-materialisation (4.5x on one synthetic case) and cuts two real-crop recognitions to 7.9-10.0 s, but the Python blueprint still finishes those crops in 2.9 s and 5.8 s on unmatched boundaries. Portable no-BLAS build, scalar vision/attention and decoder weight streaming are the open levers. | The existing `surya-det` text detector is a
different, older model (EfficientViT segformer) and is unaffected.

## What the model is (from the released package source)

Evidence: `surya-ocr==0.23.0` wheel, sha256
`42d84a89c5eb81db9d2e681dd181bc451a1e0d70ed14bd797a33beb553a0264f`, published
29 October 2026 from `datalab-to/surya` commit
`9d5aadfebb07939755c24e8a80d0e61b3be80895` ("Surya 2.5 support, release 0.23.0").
The four contract files below are byte-identical in the wheel and that commit
and are pinned by `tools/ci-heavy/surya25_metadata.py`:

| File | sha256 |
|---|---|
| `surya/inference/prompts.py` | `e5557f0fb72283f86aea36e7a2bfeab885f962d324251e241983f87365b015d2` |
| `surya/inference/util.py` | `d853246db9767d01c8648058b779047ad0d93b3c74c0f7783dcf9faeef33b11b` |
| `surya/scripts/build_gguf.py` | `d20a1cd1526243ffd63798484463dce8f7377ae497a8f4cb99b8937b9856c383` |
| `surya/inference/backends/openai_client.py` | `431672016c553f1b30ba611d3bdcb24fb139c369d30ec622dfe65f4a72db9981` |

- **Architecture: a Qwen3.5 vision-language model.** `build_gguf.py` patches
  the checkpoint `config.json` to `architectures: ["Qwen3_5ForConditionalGeneration"]`
  and converts it with stock llama.cpp (pinned `bbeb89d7`) into an LLM GGUF plus
  an mmproj GGUF. The package itself no longer runs the model in PyTorch: it
  calls vLLM (default on NVIDIA), llama.cpp `llama-server`, or MLX over an
  OpenAI-compatible chat API. The official llama.cpp artifacts are
  `datalab-to/surya-ocr-2.5-gguf` (`surya-2.5.gguf`, `surya-2.5-mmproj.gguf`).
- **Tokenizer: character-level `WordLevel`, not BPE.** The pre-tokenizer is a
  `Split` on regex `.`, with no merges. Surya 2.5 added roughly 23k vocabulary
  entries, including multi-character units ending in `<`, such as `.<`. The
  current native Qwen3.5 frontend only loads the 248,320-entry byte-level BPE
  vocabulary, so this is the largest frontend gap.
- **End marker spelling quirk.** In the original tokenizer, a prompt followed by
  `<|im_end|>` is encoded with the marker spelled out as characters, not as the
  special EOS id. HF, vLLM and MLX all feed the model this spelled form, which
  is what it was trained on. llama.cpp would parse the real special token, so
  `build_gguf.py` renames the EOS token text to `<|im_end!>` (same id) whenever
  every package prompt spells it. That changes block-label behaviour: 61 vs 1
  spelled labels over 8 test pages. Separately, character-level checkpoints can
  spell the end marker after punctuation in the *output*, so the EOS id never
  fires; the package documents `SURYA_STOP_STRINGS="<|im_end|>"` for that case.
  A native port must reproduce the original tokenizer's prompt ids exactly and
  must handle a text stop as well as the id stop.
- **MTP.** The `surya-ocr-2.5` checkpoint has no MTP head. MTP weights were a
  separate `surya-ocr-2.5-mtp` repository and were dropped from the README in
  `c4072feb`. The probe still records any `mtp` tensors in the header.
- **Licence.** Package code is Apache-2.0; the checkpoint has a separately
  pinned [modified AI Pubs OpenRAIL-M licence](https://huggingface.co/datalab-to/surya-ocr-2.5/raw/00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3/LICENSE).
  Attachment A includes revenue and funding thresholds (with personal and
  research exceptions) and a competitor restriction. Section 8 purports to
  apply the licence to output as well as model derivatives. Distribution has
  notice, attribution and licence conditions. This is not simply a revenue
  threshold; review the actual terms before deployment or redistribution.
  No CrispEmbed GGUF is authorised for publication by this note.

### Inference contract (package defaults)

- Request: one user message with the image first, then the text prompt,
  rendered through the checkpoint's own chat template. Decoding is greedy
  (`temperature=0`, `top_p=0.1`). On an error or a detected repeat loop the
  client retries with temperature 0.2/0.4/0.6 and `top_p=0.95`; that retry is
  client policy, not model semantics.
- Prompts are a training-time contract, never to be paraphrased:
  - `block`: `OCR this block image to HTML.`
  - `high_accuracy_bbox` (full page): `OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).`
  - `layout` and `table_rec`: JSON outputs, optionally constrained by guided decoding.
- Output is HTML restricted to an allow-list of tags and attributes
  (`prompts.py`), with bboxes normalised to 0–1000.
- Image scaling (`scale_to_fit`, LANCZOS): sizes are snapped to a 28 px grid,
  with at most 3072×2048 pixels and at least 1792×28. vLLM is started with
  processor `min_pixels=3136, max_pixels=6291456`. Recognition renders pages at
  192 DPI, about 4.4k visual tokens for a letter page. Package token budgets
  are 12,288 for a full page, a block ceiling of 8,192, and 3,072 for layout or
  table_rec.

## Gaps against the existing native Qwen3.5 OCR backend

The backend (`src/qwen35_*.cpp`, `models/convert-qwen35-ocr-to-gguf.py`) already
implements the Qwen3.5 dense decoder, with gated-delta linear attention and gated
full attention using interleaved MRoPE, and the Qwen3.5 vision tower. It is pinned
to the LightOnOCR-3-0.8B and OvisOCR2/3 sources:

1. **Shapes.** Resolved: every decoder and vision shape equals the implemented
   0.8B schedule, and only the vocabulary row count is new (now taken from the
   tokenizer).
2. **Tokenizer.** Resolved: the WordLevel path above.
3. **Context and image bounds.** Native limits are 1024 vision patches (256
   merged image tokens) and 4096 total tokens. A 192-DPI letter page is 13,464
   patches (3,366 merged tokens). Full-page Surya needs about 4.4k visual tokens, about 2.4k
   output tokens and up to 20.4k in the worst case. Only block-crop OCR fits
   under the current bounds, so full-page support requires a bounded,
   separately gated context extension.
4. **Preprocessing.** The processor resize is exact (see the table above). The
   package's LANCZOS `scale_to_fit` (minimum 1792×28 area, so small crops are
   upscaled) is caller-side and not yet ported.
5. **Stops.** By default the package, vLLM and HF stop only on id 2, and the
   native path matches that. The package's optional `SURYA_STOP_STRINGS`
   (a spelled `<|im_end|>` in the output) is a later, separately gated option.

## Reused from the codebase

- The Qwen3.5 decoder, vision tower, MRoPE positions and torchvision-exact
  resize (`src/qwen35_*`, `core/torch_bicubic_u8.h`) are used unchanged.
- Pillow fixed-point resampling is adapted from
  `image_preproc::resize_bicubic_pil_u8_hwc` into `core/pil_resample_u8.h`
  with a LANCZOS filter. The original is untouched; the smoldocling float
  Lanczos is not Pillow-exact.
- Candidates for the next steps: `surya_det` text-line detection to produce
  crops for full-page OCR inside the 1024-patch bound, and the
  `tools/dump_surya_*` forward-hook pattern for stage dumps if a divergence
  appears.

## Next steps

1. **Speed (shared backend).** Behind a new gate, run decoder matmuls directly
   on F16/quantised weights instead of re-materialising F32 every step. A/B it
   against the current path, which is also used by the published OvisOCR2 F16,
   before changing any default.
2. **Real-document quality.** Freeze original HF/vLLM outputs on owned real
   scans and compare CER/WER, not only synthetic crops.
3. **Full page.** Either a `surya_det` line-crop pipeline (fits the current
   bounds) or a separately gated larger patch/context bound for the
   `high_accuracy_bbox` full-page prompt (a page is about 13.5k patches).
4. **Q8_0/Q4_K** after the above, under PLAN's quantisation rules. Then
   registry with licence acceptance (openrail) and an immutable upload.

## Staged plan (each stage gated and evidenced before the next)

1. **Weight-free metadata (this change).** The workflow
   `.github/workflows/surya25-metadata.yml` runs
   `tools/ci-heavy/surya25_metadata.py` on hosted CI. It records the resolved
   revision, inventory, configs, chat template, tokenizer summary, header
   shapes, the official meta-model schema check, rendered prompt ids, the EOS
   spelling report, and processor grids for representative `scale_to_fit`
   sizes.
2. **Original reference.** Pin the revision. On an authorised host, run the
   original HF `Qwen3_5ForConditionalGeneration` (the same weights vLLM serves)
   with the exact rendered prompt ids on owned fixtures (a block crop and a
   page), and freeze the decoded text, ids, stops and per-stage dumps.
3. **Converter profile.** Add a `surya25` profile to
   `convert-qwen35-ocr-to-gguf.py`: WordLevel vocabulary export, an explicit
   stop policy, preserved source metadata, and F16/F32 output only.
4. **Native frontend and runtime behind gates.** Add a
   `CRISPEMBED_SURYA25_FRONTEND` gate (via `core_env::on`) for the tokenizer and
   prompt, and a `CRISPEMBED_SURYA25_INFERENCE` gate for shape-generalised
   weight mapping. Start with block-crop OCR within the existing 1024-patch and
   4096-token bounds. Compare the earliest divergence against stage-2 dumps and
   decoded text before any quality claim.
5. **Full-page and quantised variants.** Extend context and patch bounds
   separately and gated, then follow the Q8_0/Q4_K task-qualification rules
   from PLAN. After that come registry, licence acceptance and consumers.
