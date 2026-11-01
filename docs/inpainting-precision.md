# Inpainting precision variants (F16 / Q8_0 / Q4_K): inpaint-v1

Experimental, default-off precision variants for the issue-71 MI-GAN512,
Moebius and PixelHacker engines (and the shared SDXL VAE). The F32 profiles and
their frozen original/native budgets are unchanged; a variant is measured, not
qualified, against those F32 budgets. The owner's precision acceptance is
**output cosine > 0.94 against both native F32 and the original capture**.
Results and receipts: `tests/results/migan512/`, `tests/results/moebius/`,
`tests/results/pixelhacker/`.

## Gates

| Gate | Effect |
| --- | --- |
| `CRISPEMBED_INPAINT_QUANT=1` | Accept an `inpaint-v1` GGUF (F32 files need no gate). Default: every stored tensor is dequantized to F32 at load into its original shape, so the unchanged F32 graph runs. |
| `CRISPEMBED_INPAINT_QUANT_MATMUL=1` | Keep stored F16/block matrices and consume them directly with `ggml_mul_mat`: MI-GAN pointwise convs, latent `conv(groups=1)` (1x1 and k x k through im2col) and linear layers, on the eager CPU path and on the persistent graph path (`CRISPEMBED_INPAINT_GRAPH=1`). |

Both are value-parsed `core_env::on` gates. Precision variants run the strict
graph profile only (`CRISPEMBED_INPAINT_FAST` is rejected for them).

## Quantizer policy (`tools/quantize.cpp`, `inpaint-v1`)

`crispembed-quantize IN-F32.gguf OUT.gguf f16|q8_0|q4_k` on a GGUF whose
`general.architecture` is `migan`, `moebius`, `pixelhacker` or `sdxl_vae` uses a
self-contained family path that runs **before** the generic name guards (the
generic `downsample` guard would otherwise copy PixelHacker's `downsamplers.*`
3x3 kernels verbatim after the generic 4D pre-scan had already rewritten their
header shape; OXR had to skip the same pre-scan).

* F32-pinned: vectors (bias, norms, BN statistics, scalars); every kernel with
  one input channel per group (depthwise, Moebius `pos_conv`); MI-GAN fixed
  buffers (`noise_const`, `filter_const`, `noise_strength`) and `fromrgb` /
  `torgb`; `embedding_layer`; relative-position tables; the denoisers'
  `diff_model.conv_in*` / `conv_out*` and the VAE `conv_in`/`conv_out`/
  `quant_conv`/`post_quant_conv`.
* Everything else: the requested type, stored as a **natural-row** matrix
  (`[IC*KH*KW, OC]` for a conv, a pure reshape of the contiguous OIHW bytes;
  `[IN, OUT]` for a linear). K-quants fall back along Q4_K->Q4_0, Q5_K->Q5_0,
  Q6_K->Q8_0 when the row is not 256-aligned, then to F16.
* MI-GAN, measured: a sub-Q8 request keeps `synthesis.*` at Q8_0 (below).
* `--tensor-type GLOB=TYPE` overrides apply to every non-pinned tensor.

The file records `inpaint.quantization.policy=inpaint-v1` and the requested
type. Latent variants carry an adjacent `crispembed-inpaint-precision-variant-v1`
receipt naming their source F32 conversion; the latent comparator verifies that
source with the unchanged F32 receipt/readback checks.

## GGUF tensor names

GGUF names hold at most 63 bytes (`GGML_MAX_NAME`). 540 Moebius and 384
PixelHacker state names are longer (up to 89 bytes), so the first hosted F32
replay could not open any denoiser GGUF. Names of 64+ bytes are now stored under
one deterministic abbreviation (`src/inpaint_tensor_names.h`; Python twin
`gguf_tensor_name`): `diff_model.`->`dm.`, `.transformer_blocks.`->`.tb.`,
`.attentions.`->`.at.`, `down_blocks.`->`db.`, `up_blocks.`->`ub.`,
`mid_block.`->`mb.` (max 57 bytes, collision-free). All loaders map lookups
through it and re-key tensors to original names. MI-GAN, VAE and LaMa names
are unaffected.
