# MiniCPM-OCR 1.0: inference spec and native port plan

Status (2026-10-28): **source audited, no native code yet.** Nothing here is
qualified; every "verify" item must be confirmed by the hosted original reference
capture (`tools/minicpm-ocr/reference.py`) before native work relies on it.

## Identity and pins

- Checkpoint: [`openbmb/MiniCPM-OCR-1.0`](https://huggingface.co/openbmb/MiniCPM-OCR-1.0/tree/6796197bb3a8e224b2f64ae462bfbb3f8a15ac4b),
  revision `6796197bb3a8e224b2f64ae462bfbb3f8a15ac4b` (created 2026-10-21). It is the
  first dedicated OCR model in the MiniCPM-V/o family, trained from MiniCPM-V 4.6.
- Full file pins (bytes, git oid, sha256) are in [`minicpm-ocr-source.json`](minicpm-ocr-source.json).
  `model.safetensors` is a single BF16 file of 2,600,969,816 bytes, with
  sha256 `83f4e433…f8fd4`. It holds 779 tensors and 1,300,434,160 parameters.
- Reference implementation: Transformers **5.21.0**, the native `minicpm_ocr` model
  (`MiniCPMOcrForConditionalGeneration`, `MiniCPMOcrProcessor`,
  `MiniCPMOcrImageProcessor`). The Hub repo has no remote code. Pin the exact
  Transformers wheel, because moving `main` may differ.
- Publisher llama.cpp export: `openbmb/MiniCPM-OCR-1.0-gguf` at revision `674e62bf…`.
  It contains a text GGUF plus `mmproj-model-f16-4x.gguf`, and the publisher's
  guide says it runs on upstream `ggml-org/llama.cpp` master (mtmd, `llama-server --mmproj`).
  We can use it as a second, independent cross-check, but not as the source of
  truth or the CrispEmbed format.

## License verdict

The model card metadata says `apache-2.0`, and the card text says "Model weights,
code, and documentation are released under the Apache License 2.0". There is no
MiniCPM Model License clause, no commercial registration requirement and no
usage restriction. Older MiniCPM-V releases did have one of those. The
publisher itself redistributes GGUF/AWQ/GPTQ derivatives under Apache-2.0.
**Publishing CrispEmbed GGUFs is permitted**, provided we:

1. include the Apache-2.0 text,
2. credit OpenBMB and link the source revision, and
3. state the modifications (format conversion and any quantization).

Fetch the LICENSE from the code repository at publication time and pin its hash.
The Hub repo does not contain it.

## Architecture (verified from config.json and the safetensors header)

| Part | Facts |
| --- | --- |
| Vision | SigLIP-so400m-style ViT: 27 layers, width 1152, 16 heads, MLP 4304 with `gelu_pytorch_tanh`, pre-LN with eps 1e-6, patch 14, conv patch embedding with bias, `post_layernorm`. All q/k/v/out projections have biases. |
| Positions | Learned `position_embedding` of shape [4900, 1152], which is a 70×70 grid (980/14). Variable-resolution (NaViT) inputs get position ids by nearest bucketing of each patch's fractional (h,w) coordinate into that grid (MiniCPM-V `bucketize` scheme). **Verify the boundary rule** against reference `position_ids`. |
| `vit_merger` | Window-attention merger inserted after vision layer `insert_layer_id=6`. It is used **only in 16x mode**. This checkpoint runs `downsample_mode="4x"`, so it is skipped. Its weights are present (linear_1 is 4608→17216) and must be kept in the GGUF, unused. |
| Projector | `model.merger.mlp.0` (only one stage). Each 2×2 patch block is concatenated into 4608 channels, then LayerNorm(4608, with bias) → Linear 4608→4608 → GELU → Linear 4608→1024. Check the reference for exact (tanh vs erf) GELU and the concat order (row-major within the 2×2 window). |
| LLM | Qwen3.5 text model with the same shapes as Qwen3.5-0.8B and LightOnOCR-3. It has 24 blocks of width 1024, laid out as 3×linear (gated delta-net) then 1×full attention, repeated. FFN is SwiGLU 3584. Full attention has 8 q heads, 2 kv heads and head_dim 256, with a gated q_proj (4096 = 8×256×2) and q/k RMSNorm. Linear attention has 16 k/v heads of width 128, conv kernel 4, and A_log/dt_bias. Vocabulary is 248,100, with tied embeddings and no `lm_head` tensor. RMSNorm is zero-centered. |
| RoPE | Partial: 64 of 256 dims, theta 1e7, `rope_type: default`. There is **no mrope_section**, so positions are plain 1-D sequential over the whole sequence, image tokens included. In the qwen35 code this is the same as using equal t/h/w axes. Verify in the reference. |

## Preprocessing (`MiniCPMOcrImageProcessor`, preprocessor_config.json)

Parameters: `scale_resolution=448`, `max_slice_nums=9`, `patch_size=14`,
`slice_mode=true`, `use_image_id=true`, `downsample_mode="4x"` (merge factor 2,
so sizes snap to multiples of 28), and mean/std 0.5/0.5 (pixels map to [-1, 1]).
Defaults: `small_image_upsize="always"`.

1. Convert to RGB. `ratio = W*H / 448²` and `multiple = min(ceil(ratio), 9)`.
2. **Overview image** (`find_best_resize`). Scale so that the area is 448² while
   keeping the aspect ratio. Small images are upscaled in `always` mode when
   there is no slice grid. Each side is rounded with `ensure_divide` to the
   nearest multiple of 28, with a minimum of 28. Resample: bicubic (verify the
   PIL filter and whether antialias applies).
3. **Slices** (`get_sliced_grid`). If `multiple <= 1`, there are none. Otherwise
   consider counts `multiple-1, multiple, multiple+1` (skipping 1 and anything
   above `max_slice_nums`). For each count, enumerate the factorizations rows×cols.
   Pick the one that minimizes `|log(cols/rows) - log(W/H)|`. Ties go to more rows,
   but **verify the tie order** against the reference.
4. `get_refine_size`: derive the per-cell size, take `find_best_resize` of the
   cell (allowing upscale), and multiply by the grid. Resize the whole image to
   that size with bicubic, then cut it into equal tiles in row-major order.
5. Each image (overview first, then the tiles) is normalized and unfolded patch-wise.
   `target_sizes = [h/14, w/14]`. Tokens per image are `(h/14)*(w/14)/4`.
6. The adaptive-upsize helpers (`SMALL_FONT_PX=12`, `DENSE_GLYPHS_PER_KPX=2`,
   `TINY_PIXELS=6272`, `TINY_RESOLUTION=84`) apply only in `small_image_upsize="adaptive"`.
   That mode is not the default and is out of scope for parity v1.

## Token layout and prompt

Special token ids: `<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046,
`<|image_pad|>` 248056, `<image>` 248078, `</image>` 248079, `<slice>` 248088,
`</slice>` 248089, `<image_id>` 248090, `</image_id>` 248091, `<think>` 248068,
`</think>` 248069. Generation EOS is `[248044, 248046]`.

The chat template renders as follows, with the image item first and then the
task text. Content parts are joined with `\n`, and thinking is off by default:

```
<|im_start|>user\n<|image_pad|>\nText Recognition:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The processor then replaces each `<|image_pad|>`:

```
<image_id>0</image_id><image>{pad × N0}</image>{rows joined by "\n"; each row = (<slice>{pad × Ni}</slice>)×cols}
```

Image ids start at 0 for each sample. Whether a `\n` separates `</image>` from
the first `<slice>` must be **verified** from the reference `input_ids`.
Projector outputs are `masked_scatter`ed into the pad positions in order:
overview first, then the slices in row-major order.

Task prompts (from the model card): `Text Recognition:`, `Table Recognition:`
(HTML), `Formula Recognition:` (LaTeX), `Seal Recognition:`, `Chart to Table:`.
There is also a KIE JSON prompt. The model is meant for **layout-cropped regions**:
the official SDK runs PP-DocLayoutV3 first. Whole pages also work, which the
"page" row in the card confirms.

## Decoding

Greedy decoding (`do_sample=False`) in the reference. The generation config sets
no temperature or repetition penalty. Stop on 248044 or 248046. The publisher
serves with context 40960. Use `max_new_tokens` 1024 for the parity cases and
record whether the cap was hit.

## Implementation plan (reuse)

1. **LLM: reuse `src/qwen35_*` as is.** The decoder (delta-net linear blocks,
   gated full attention, zero-centered norms, tokenizer `qwen35`) is the
   Qwen3.5-0.8B shape already qualified for LightOnOCR-3 and OvisOCR2/3. Needed
   changes:
   - position ids: 1-D sequential, written as equal axes;
   - embedding injection at `<|image_pad|>` from an external feature buffer
     (`qwen35_ocr.cpp` already scatters Qwen-VL features);
   - the two-id EOS list. The OvisOCR3 profile already handles `[248044, 248046]`.
2. **Vision: a new `src/minicpm_ocr_vision.cpp`.**
   - Port the SigLIP block from the existing SigLIP users (`lfm2_vl_ocr.cpp` is
     SigLIP2 NaFlex, plus `granite_vision_ocr.cpp` and `smoldocling_ocr.cpp`).
   - Add MiniCPM bucketed 70×70 positions.
   - Run each image (overview and each slice) as an independent sequence, with
     no cross-image attention (`cu_seqlens`).
   - Take the post-LN, then the 2×2 merge MLP.
   - Skip `vit_merger` behind the 4x mode. A 16x path could be a later gate.
3. **Preprocessing: a new `src/minicpm_ocr_tiles.h`.** Follow the existing
   tiling helpers (`unlimited_ocr2_tiles.h`, `lfm2_vl_tiling.h`) and implement
   steps 1–5 above. The deepseek/unlimited tilers are a different algorithm, so
   only their structure and tests carry over, not their math.
4. **Orchestration.** Add an ocr_orchestrator backend `minicpm-ocr` behind a
   `core_env::on()` gate (default off) with a task-prompt option. Do not change
   any default.
5. **Converter.** `models/convert-minicpm-ocr-to-gguf.py` (skeleton committed)
   has these properties:
   - arch `minicpm_ocr`;
   - prefixes `l.` (language), `v.` (vision tower), `m.` (merger);
   - qwen35 tokenizer metadata;
   - processor config and chat template embedded;
   - source revision and checkpoint sha256 recorded;
   - it fails on any unaudited tensor;
   - it keeps `vit_merger` tensors.

## Parity plan

1. **Hosted original capture** with `tools/kaggle/minicpm-ocr-reference`, which
   runs `tools/minicpm-ocr/reference.py` on Kaggle CPU in F32 with the pinned
   transformers 5.21.0.
   - Inputs: frozen synthetic regions (a single line, a paragraph, a table, a
     tall/wide aspect to force slicing, and a tiny image to force upscaling).
   - Saved preprocessing outputs: pixel_values, target_sizes, rendered prompt
     and full `input_ids`.
   - Saved hooks: vision embeddings, vision layers 0/6/26 and post_layernorm,
     merger output, decoder per-layer prefill outputs, final norm and the first
     logits.
   - Saved results: greedy generated IDs, text, EOS/cap status.
   - The driver verifies every pinned file before loading.
2. **Native stages**, compared at the earliest point of divergence:
   1. Tokenizer and placeholder IDs, which must match exactly.
   2. Pixel values, which must match within bicubic tolerance. Document any
      residual from the resampler.
   3. Vision and merger tensors.
   4. Decoder layers.
   5. Exact greedy ID equality and EOS on all cases, F32 then F16.
3. **Quant (Q8/Q4) only afterwards**, under the same eleven-case style quality
   gates as OvisOCR2. Publication follows only after F16 qualification.

## Next concrete steps

1. Run the reference kernel (push via `tools/kaggle/kpush.py`) and freeze its report under
   `tests/results/minicpm-ocr/`.
2. Resolve the "verify" items: bucketing rule, slice tie-break, newline before
   `<slice>`, GELU variant, and resample filter.
3. Finish the converter on a hosted runner and compare tensor names and shapes against meta-init.
4. Write the tiler and vision encoder, then wire up the qwen35 decoder and run
   the token-ID parity.
