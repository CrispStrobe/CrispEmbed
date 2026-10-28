# MiniCPM-OCR 1.0: inference spec and native port plan

Status (2026-10-28): **native runtime written, not run, not qualified.** The
default-off runtime (`CRISPEMBED_MINICPM_OCR=1`, plus `CRISPEMBED_QWEN35_OCR=1`
for the reused decoder kernels) is in `src/minicpm_ocr*.{h,cpp}`. It has only
been syntax-checked. Every former "verify" item is now settled by reading the
pinned Transformers 5.21.0 source and Hub files (see "Decision table"); no
`TODO(verify)` and no semantics switch is left in the code. The hosted
reference capture (`tools/minicpm-ocr/reference.py`) and the native parity
kernel (`tools/kaggle/minicpm-ocr-native`) still have to confirm the port
numerically.

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
| Positions | Learned `position_embedding` of shape [4900, 1152], which is a 70×70 grid (980/14). Variable-resolution (NaViT) inputs get position ids by nearest bucketing of each patch's fractional (h,w) coordinate into that grid; exact rule in the decision table. |
| `vit_merger` | Window-attention merger inserted after vision layer `insert_layer_id=6`. It is used **only in 16x mode**. This checkpoint runs `downsample_mode="4x"`, so it is skipped. Its weights are present (linear_1 is 4608→17216) and must be kept in the GGUF, unused. |
| Projector | `model.merger.mlp.0` (`merger_times=1`, one stage). Each 2×2 patch block is concatenated into 4608 channels in (dy, dx) row-major order, then `pre_norm` LayerNorm(4608, eps 1e-6, with bias) → `linear_1` 4608→4608 → `nn.GELU()` (erf) → `linear_2` 4608→1024. |
| LLM | Qwen3.5 text model with the same shapes as Qwen3.5-0.8B and LightOnOCR-3. It has 24 blocks of width 1024, laid out as 3×linear (gated delta-net) then 1×full attention, repeated. FFN is SwiGLU 3584. Full attention has 8 q heads, 2 kv heads and head_dim 256, with a gated q_proj (4096 = 8×256×2) and q/k RMSNorm. Linear attention has 16 k/v heads of width 128, conv kernel 4, and A_log/dt_bias. Vocabulary is 248,100, with tied embeddings and no `lm_head` tensor. RMSNorm is zero-centered. |
| RoPE | Partial: 64 of 256 dims, theta 1e7, `rope_type: default`. The processor and model pass no `position_ids`, so the text model uses `arange` expanded to four equal rows; with equal t/h/w axes the default `[11, 11, 10]` interleave is the identity. Positions are plain 1-D sequential over the whole sequence, image tokens included. |

## Preprocessing (`MiniCPMOcrImageProcessor`, preprocessor_config.json)

Parameters: `scale_resolution=448`, `max_slice_nums=9`, `patch_size=14`,
`slice_mode=true`, `use_image_id=true`, `downsample_mode="4x"` (merge factor 2,
so sizes snap to multiples of 28), and mean/std 0.5/0.5 (pixels map to [-1, 1]).
Defaults: `small_image_upsize="always"`.

1. Convert to RGB. `ratio = W*H / 448²` and `multiple = min(ceil(ratio), 9)`.
2. **Overview image** (`find_best_resize`). Scale so that the area is 448² while
   keeping the aspect ratio. Small images are upscaled in `always` mode when
   there is no slice grid. Each side is rounded with `ensure_divide` to the
   nearest multiple of 28 (Python half-to-even `round`), with a minimum of 28.
   Resample: torch CPU uint8 antialiased bicubic (see the decision table).
3. **Slices** (`get_sliced_grid`). If `multiple <= 1`, there are none. Otherwise
   consider counts `multiple-1, multiple, multiple+1` (skipping 1 and anything
   above `max_slice_nums`). For each count, enumerate the factorizations rows×cols.
   Pick the one that minimizes `|log(W/H) - log(cols/rows)|`. Rows ascend; an
   exactly equal error replaces the best only when it has more rows.
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

Image ids start at 0 for each sample. There is no separator between `</image>`
and the first `<slice>`; rows are joined by `\n` with no trailing `\n`.
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

## Decision table (source-pinned, 2026-10-28)

Sources are the Transformers **v5.21.0** release (`src/transformers/` at tag
`v5.21.0`; line numbers from the published 5.21.0 wheel, which is generated
from that tag) and the Hub files of `openbmb/MiniCPM-OCR-1.0` at revision
`6796197b`. Abbreviations: `IP` = `models/minicpm_ocr/image_processing_minicpm_ocr.py`,
`MOD` = `models/minicpm_ocr/modeling_minicpm_ocr.py`, `PROC` =
`models/minicpm_ocr/processing_minicpm_ocr.py`, `Q35` =
`models/qwen3_5/modeling_qwen3_5.py`, `BK` = `image_processing_backends.py`.

| Item | Decision | Source | Quoted snippet |
| --- | --- | --- | --- |
| Image processor class | `MiniCPMOcrImageProcessor`, a `TorchvisionBackend`. 5.21.0 has no PIL or slow variant: the model folder has no `*_fast` or `*_pil` file. | `IP:139`; Hub `preprocessor_config.json` | `class MiniCPMOcrImageProcessor(TorchvisionBackend):` |
| Size snapping | Multiple of `patch_size * 2` = 28 in "4x" mode. `ensure_divide` uses Python `round` (half to even), with a floor of 28. The area rescale truncates with `int()`: `h = int(448/sqrt(w/h))`, then `w = int(h*aspect)`. | `IP:74-75`, `IP:173-182` | `max(round(length / divisor) * divisor, divisor)`; `merge_factor = 2 if downsample_mode == "4x" else 4` |
| Upscale rule | An unsliced image is always rescaled to area 448² (`small_image_upsize="always"`). A sliced overview is only downscaled. Slices pass `allow_upscale=True`. | `IP:289`, `IP:307` | `allow_upscale = best_grid is None and small_image_upsize == "always"` |
| Slice grid and ties | Counts `multiple-1..multiple+1` (skip 1 and >9). Rows ascend. A strictly smaller error wins; an equal error wins only with more rows. All math is double. | `IP:207-234` | `elif error == min_error and num_rows > best_grid[0]:` |
| Refine size and tiles | `ensure_divide(W, cols)` / `ensure_divide(H, rows)`, then the float cell size goes into `find_best_resize(..., allow_upscale=True)`, and the result is multiplied by the grid. One resize of the original image is cut into tiles row-major. | `IP:184-205`, `IP:310-316`; `image_transforms.py:894-897` | `for i in range(0, height, patch_h): for j in range(0, width, patch_w)` |
| Resize filter | `tvF.resize(uint8 tensor, BICUBIC, antialias=True)`. torchvision keeps uint8 for bicubic on CPU, so torch's uint8 AA kernel runs: a = -0.5, int16 weights and the precision rule. This is not PIL's int32 22-bit path, so the previous "PIL-exact" default was wrong. | `BK:230-283`; torchvision v0.28.0 `transforms/v2/functional/_geometry.py:270-280`; torch v2.13.0 `aten/src/ATen/native/cpu/UpSampleKernel.cpp:1010-1034, 1342-1347, 352-359` | `return interpolation == InterpolationMode.BICUBIC or ...`; `constexpr scalar_t a = use_keys_cubic ? -0.5 : -0.75;` |
| Rescale and normalize | Fused rescale and normalize. Mean and std are multiplied by `1/rescale_factor` (255.0), which gives `(float32(v) - 127.5f) / 127.5f` in float32 after the uint8 resize. Mean and std are 0.5 (`IMAGENET_STANDARD_*`; the same values are in `preprocessor_config.json`). | `BK:325-360`; `IP:144-145` | `image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)` |
| Patch layout and packing | `unfold` gives row-major patches, and the conv runs over `[C, 14, N*14]`. Each image is its own sequence (cu_seqlens). Attention is split per image, with no mask and no padding. | `IP:236-244`; `MOD:80-93`, `MOD:205-229`, `MOD:461-470` | `lengths = cu_seqlens[1:] - cu_seqlens[:-1]` ... `attention_mask=None` |
| Position bucketing | `boundaries = torch.arange(1/70, 1.0, 1/70)` in float32 (69 values). `coords = arange(n)/n` in float32. `bucketize(right=True)`, then id = `bh*70 + bw`. | `vision_utils.py:346-356` | `torch.bucketize(h_coords, boundaries, right=True)` |
| vit_merger in 4x | Skipped (`use_vit_merger = downsample_mode != "4x"`). All 27 layers run, then post_layernorm. The weights stay in the GGUF unused. | `MOD:618-628`, `MOD:474-489` | `use_vit_merger = downsample_mode != "4x"` |
| Post-layernorm | `LayerNorm(1152, eps=1e-6)` on the last hidden state. | `MOD:417`, `MOD:489`; Hub `config.json` | `last_hidden_state = self.post_layernorm(hidden_states)` |
| 2×2 merge order | `view(mh, 2, mw, 2, D).permute(0, 2, 1, 3, 4)`: within a cell the order is (dy, dx, D); cells are row-major. | `MOD:540-546` | `.permute(0, 2, 1, 3, 4)` |
| Projector norm and GELU | `pre_norm = nn.LayerNorm(4608, eps=1e-6)` (a literal, not a config value), `act = nn.GELU()` (erf), `merger_times=1`, so `mlp.0` is the only stage. The vision MLP itself uses `gelu_pytorch_tanh`. | `MOD:500-502`; `configuration_minicpm_ocr.py:94`; safetensors header `model.merger.mlp.0.pre_norm.{weight,bias}` [4608] | `self.pre_norm = nn.LayerNorm(merged_hidden_size, eps=1e-6)` / `self.act = nn.GELU()` |
| RoPE | No `position_ids` are passed, so `arange` is expanded to four equal rows (text, t, h, w). The default `mrope_section` interleave of equal axes is the identity. Rotary dim `int(256*0.25)` = 64, theta 1e7, `rotate_half` on the first 64 dims. | `Q35:1247-1256`, `Q35:155-176`, `Q35:178-201`, `Q35:685-695`; Hub `config.json` `rope_parameters` | `position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)` |
| Placeholder text | `<image_id>0</image_id><image>` + pad×N0 + `</image>`, then rows joined by `"\n"`, each row being `(<slice>pad×Ni</slice>)×cols`. There is no separator before the first `<slice>` and no trailing newline. Ni = (h/14)·(w/14)/4. | `PROC:151-182`, `PROC:232-242` | `image_placeholder += "\n".join(slices)` |
| Chat template, thinking off | Content parts are joined by `\n`, image first. The generation prompt is `<|im_start|>assistant\n<think>\n\n</think>\n\n`, and `enable_thinking` defaults to false. | Hub `chat_template.jinja:1-3, 26, 79, 138-144` | `{{- '<think>\n\n</think>\n\n' }}` |
| Tokenizer pretokenizer | The `tokenizer.json` Split regex: plain `\p{L}+`, no `\p{M}`. The `pretokenize_regex` key in `tokenizer_config.json` is never read. AutoTokenizer maps `minicpm_ocr` to `TokenizersBackend`, which loads `tokenizer.json` as is, and the string `pretokenize_regex` appears nowhere in the 5.21.0 package. | `models/auto/tokenization_auto.py:205`; Hub `tokenizer.json` `pre_tokenizer` | `("minicpm_ocr", "TokenizersBackend" if is_tokenizers_available() else None)` |
| Stop ids | `[248044, 248046]`, greedy decoding, no other sampling fields. | Hub `generation_config.json` | `"eos_token_id": [248044, 248046]` |
| Overview image | The overview is the first part. Its token count comes from its own `target_sizes`, and it is not adaptive in "always" mode. | `IP:293-299`, `IP:343-344` | `image_patches: list[torch.Tensor] = [source_img]` |

The expected values in `tests/test_minicpm_ocr_tiles.cpp` come from a numpy
transcription of `IP` and `vision_utils.py`, line by line, with no torch and no
weights. The C++ output (`make_plan`, `bucket_coords`) is byte-identical to
that transcription for 5,913 image sizes from 8 to 3000 px and for bucket rows
n = 1..2000. For every n ≤ 2000 the float32 bucket rule equals
`floor(70*i/n)`.

**Still undetermined by the source: nothing.** The remaining risk is numerical.
The torch uint8 AA kernel was transcribed from source, and it was qualified
earlier against torch 2.10 pixels for OvisOCR2. The capture with torch 2.13.0
must show bit-equal `pixel_values`. Any difference between the AVX2 and generic
torch kernels would show up in the dump comparison.

## Prior art (cross-check only, no code copied)

| Repo | Revision | License | Relevant behavior / differences vs Transformers |
| --- | --- | --- | --- |
| ggml-org/llama.cpp `tools/mtmd` | `f4d8af608a6b39946406f9439edfb6e83ca6a923` (master, 2026-10-28) | MIT | `PROJECTOR_TYPE_MINICPM_OCR_1_0` (clip.cpp:1562, mtmd.cpp:726-740, mtmd-image.h:214-222). Matches Transformers on the `<image>`/`<slice>` layout, rows joined by `\n` with no trailing newline, the `<image_id>N</image_id>` prefix and 28-px alignment in 4x mode. Differences: (1) `get_best_grid` computes `log_ratio` in float32, keeps the first minimum while iterating cols ascending, and has no "more rows" rule across counts (mtmd-image.cpp:743-772); (2) the slice ratio is float32; (3) positions are `floor(70.0*i/n)` in double (clip.cpp:5578-5590), equal to torch's float32 bucketize for n ≤ 2000; (4) the image id is the global `n_images_added`, not the per-sample index; (5) llama.cpp uses its own bicubic resampler, not the torch uint8 int16 kernel. |
| OpenBMB/MiniCPM-OCR `docs/llamacpp_deploy.md` | `main` (2026-10-28) | Apache-2.0 | Points to upstream `ggml-org/llama.cpp`; there is no fork or branch. |
| openbmb/MiniCPM-OCR-1.0-gguf | Hub `main` | Apache-2.0 (card) | Q4_K_M to Q8_0 text GGUFs plus `mmproj-model-f16-4x.gguf` for upstream llama.cpp. |

Transformers is the parity target. llama.cpp was read only to learn its behavior.

## Reuse map (2026-10-28)

| Component | Existing code | Optimizations carried | Use here |
| --- | --- | --- | --- |
| Torch uint8 AA bicubic resize | `src/qwen35_image.cpp` (Qwen2VL/OvisOCR preprocessing, qualified against torch pixels) | Fixed-point int16 two-pass; skips no-op passes | **Reused.** Moved verbatim to `src/core/torch_bicubic_u8.h`, and `qwen35_image.cpp` now calls it. A/B harness on 39 random images: the hash of resized pixels and patches is byte-identical before and after. This touches a shared qwen35 file, so OvisOCR declaration pins need a refresh. |
| PIL-exact / float bicubic (`image_preprocess.h`, `unlimited_ocr.cpp`, `lfm2_vl_tiling.h`) | various | n/a | Not used. They do not match the torchvision uint8 path that the source selects. |
| Qwen3.5 BPE tokenizer | `qwen35_ocr::frontend_tokenizer` (`src/qwen35_tokenizer.cpp`) | Added-token splitting, NFC, byte-level BPE with merge ranks | **Reused** through a new `load_minicpm_ocr()` loader (vocab 248100, stops [248044, 248046], plain pattern). `load()`, `encode()` and `decode()` are unchanged for existing profiles. This replaces the duplicate local tokenizer. |
| Qwen3.5 decoder mapping | `qwen35_ocr::map_small_decoder` | Shared weight schedule | **Generalized** with a vocabulary-size overload. The old signature forwards 248320, so existing behavior is unchanged. This replaces the duplicate local mapper. |
| Qwen3.5 decoder kernels | `qwen35_ocr::embed_cpu` / `decoder_cpu` (`decoder_cache` KV, `converted_weight_cache` behind `CRISPEMBED_QWEN35_WEIGHT_CACHE`) | KV-cached decode, delta-net state reuse, optional converted-weight cache | Reused as is. Prefill is one batched call, and decode is cached per token. |
| SigLIP block | `smoldocling_ocr.cpp` `sd_vision_forward` pattern; `core/gelu_tanh_f32.h` | ggml graph; GELU-tanh without the F16 LUT | Adapted, as in the earlier port. There is no shared SigLIP function; each engine builds its own graph. |
| Fused vision attention | `lfm2_vl_ocr.cpp` `ggml_flash_attn_ext` (default on there after A/B) | Flash attention at F32 precision | Adapted behind the default-off `CRISPEMBED_MINICPM_OCR_VISION_FLASH` until the capture A/B is done. |
| Weight loading | `core_gguf::load_weights` (mmap/backend buffers) | mmap | Reused. |
| Gates | `core_env::on` | | Reused. |
| Debug dumps | Local `.npy` writer in `minicpm_ocr.cpp` | | Kept. No shared npy helper uses the reference names. |
| Fast ggml decoder (Unlimited-OCR-2) | `unlimited_ocr*.cpp` | ggml graph decode | Not used: it is a different architecture. The qwen35 CPU decoder is the one qualified for this shape. |
| CLI / C API / server dispatch | `crispembed.cpp`, `ocr_orchestrator.cpp`, `examples/cli` | | Reused, already wired in the base commit. |

## Native runtime (2026-10-28, unqualified)

| Piece | File | Notes |
| --- | --- | --- |
| Slicing, grid, token count, bucketed positions | `src/minicpm_ocr_tiles.h`, `tests/test_minicpm_ocr_tiles.cpp` | Header-only. Expected values come from the Python algorithm, not from the reference capture. |
| Preprocessing, SigLIP tower, 4x projector | `src/minicpm_ocr_vision.cpp` | Each image runs as its own sequence. Resampling is the shared torch uint8 AA bicubic (`src/core/torch_bicubic_u8.h`). `CRISPEMBED_MINICPM_OCR_VISION_FLASH=1` (default off) is a performance-only fused-attention gate, pending A/B. |
| Prompt, injection, qwen35 decoder, greedy decode | `src/minicpm_ocr.cpp` | Positions are 1-D, written as equal axes. Stops are 248044/248046. Tokenizer and decoder mapping are the shared `qwen35_ocr::frontend_tokenizer::load_minicpm_ocr` and `map_small_decoder(..., 248100)`. |
| Unified OCR C API and CLI | `src/crispembed.cpp`, `src/ocr_orchestrator.cpp`, `examples/cli/main.cpp` | `--ocr` dispatches on arch `minicpm_ocr`. `--ocr-pipeline` uses `--ocr-engine minicpm-ocr` (id 23), which needs `--ocr-rec`. |
| Debug dump | `CRISPEMBED_MINICPM_OCR_DUMP=<dir>` | Writes `.npy` files named like the reference arrays, plus `native_pixels_<i>.npy` and `native_meta.json`. |

No `TODO(verify)` items remain; see the decision table.

## Next concrete steps

1. Run the reference kernel (push via `tools/kaggle/kpush.py`) and freeze its report under
   `tests/results/minicpm-ocr/`.
2. Compare the capture's `pixel_values`, `input_ids` and stage tensors against
   `CRISPEMBED_MINICPM_OCR_DUMP` at the earliest divergence.
3. Finish the converter on a hosted runner and compare tensor names and shapes against meta-init.
4. Write the tiler and vision encoder, then wire up the qwen35 decoder and run
   the token-ID parity.
