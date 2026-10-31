# Surya OCR 2.5 (`datalab-to/surya-ocr-2.5`) — source contract and port plan

Status: **source audit only.** No converter profile, native runtime path, registry
entry or quality claim exists yet. The existing `surya-det` text detector is a
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
- **Licence.** Package code is Apache-2.0; the model weights are OpenRAIL-M per
  the package README badge (free under a revenue threshold, the same terms as
  `surya-det`). Per POLICY §7 this belongs in the restricted registry class
  behind CLI licence acceptance, and outside PLAN's permissive OCR queue. The
  hosted probe records the card's declared licence; confirm it before any upload.

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

1. **Shapes.** `valid_weights` and the decoder mapping hard-code the 0.8B
   dimensions (1024 hidden, 24 layers, 248,320 vocabulary, 12 vision blocks).
   The decoder kernels are config-driven within bounds, and a 2048-wide mapping
   also exists. The Surya 2.5 sizes come from the hosted probe below.
2. **Tokenizer.** A WordLevel character tokenizer needs a new encode/decode
   path, plus exact added-token matching and the spelled-`<|im_end|>` prompt
   behaviour. Decode must keep NORMAL tokens as raw characters, not GPT-2
   byte-unicode.
3. **Context and image bounds.** Native limits are 1024 merged patches and 4096
   total tokens. Full-page Surya needs about 4.4k visual tokens, about 2.4k
   output tokens and up to 20.4k in the worst case. Only block-crop OCR fits
   under the current bounds, so full-page support requires a bounded,
   separately gated context extension.
4. **Preprocessing.** The probe records the checkpoint processor (patch, merge,
   resize and normalisation) next to the package's LANCZOS `scale_to_fit`. Both
   must be reproduced, the second as the caller-side contract.
5. **Stops.** Stop on the EOS id and on the spelled `<|im_end|>` text, and
   record which one fired.

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
