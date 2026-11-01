# Latent inpainting: source attribution and licence notices

Keep this directory with source distributions and with the notices supplied
for binary distributions containing the latent inpainting implementation.
CrispEmbed's original code is covered by the repository's root licence.
The upstream materials identified below retain their respective terms.

The native files `src/latent_inpaint_unet.cpp`, `src/latent_inpaint_vae.cpp`
and `src/latent_inpaint.cpp` are new C++ adaptations of the identified original
inference architecture and operations. Changes include channel-first native
tensor storage, explicit CPU/ggml operation dispatch, fixed inference-only
profiles, GGUF loading, separate conditional/unconditional execution,
independent native random generation, and test observers. Training, optimizers,
Python framework integration and upstream deployment code are not included.
`tools/dump_latent_inpaint_reference.py` loads a separately acquired, verified
original checkout; it does not bundle that checkout or replace its operators.

## Moebius and PixelHacker

- Moebius: `hustvl/Moebius`, source revision
  `b88d462bacb9af6e7128a3b4cc4a07418bedfd61`.
  The original licence is retained unchanged in `moebius-Apache-2.0.txt`,
  including **Copyright 2024 Moebius Authors**.
  [Original licence](https://github.com/hustvl/Moebius/blob/b88d462bacb9af6e7128a3b4cc4a07418bedfd61/LICENSE).
- PixelHacker: `hustvl/PixelHacker`, source revision
  `f5567db2871598aa178fe7a34c520dd478a0b41b`.
  The original licence is retained unchanged in `pixelhacker-Apache-2.0.txt`.
  [Original licence](https://github.com/hustvl/PixelHacker/blob/f5567db2871598aa178fe7a34c520dd478a0b41b/LICENSE).

These cover the upstream denoiser structure, conditioning and inference
pipeline inspected for the native adaptations. The source contract in
`docs/latent-inpaint-source-contract.json` records the exact original files
and their SHA-256 values.

## Upstream model-operation sources

The originals themselves build on the following projects. Their unchanged
licence files are included here to preserve attribution alongside the native
adaptations:

| Project | Inference operations inspected | Retained licence |
| --- | --- | --- |
| Hugging Face Diffusers 0.30.2 and 0.38.0 | AutoencoderKL, ResNet/UNet structure, normalization, timestep embedding, attention and DDIM | `diffusers-Apache-2.0.txt` |
| timm / PyTorch Image Models | Depthwise separable convolution, BatchNorm/activation and residual order used by Moebius | `timm-Apache-2.0.txt` |
| NVIDIA Sana | GLUMBConv feed-forward block used by Moebius | `sana-Apache-2.0.txt` |
| Lambda Networks | Local/content lambda attention adapted by Moebius | `lambda-networks-MIT.txt` |
| Flash Linear Attention 0.3.2 | Gated linear attention, recurrent state and fused RMSNorm/SiLU gate used by PixelHacker | `flash-linear-attention-MIT.txt` |

Relevant original source notices are retained here as well:

```text
Copyright 2024 The HuggingFace Team. All rights reserved.
Copyright 2025 The HuggingFace Team. All rights reserved.
Copyright 2024 HuggingFace Inc.
Copyright 2025 HuggingFace Inc.
Copyright 2024 Stanford University Team and The HuggingFace Team. All rights reserved.
Copyright 2025 Stanford University Team and The HuggingFace Team. All rights reserved.
Hacked together by / Copyright 2019, Ross Wightman
Copyright 2024 NVIDIA CORPORATION & AFFILIATES
Copyright (c) 2020 Phil Wang
Copyright (c) 2023-2025 Songlin Yang
```

`upstream-notice-sources.json` supplies immutable URLs, revisions, byte counts
and SHA-256 values for the downloaded additional licence texts. The actual
inference source versions are pinned separately in the model source contract;
the licence-source manifest does not select runtime dependency versions.

## Weight terms are recorded separately

No model weights are distributed in this source package. The following
unchanged first-party model cards preserve the relevant weight declarations:

- `moebius-weights-model-card.md`: author HF revision
  `cd01f47fb648219d3fa605806c5ce00b713faa5e`, with an MIT tag.
  The [pinned Moebius source README](https://github.com/hustvl/Moebius/blob/b88d462bacb9af6e7128a3b4cc4a07418bedfd61/README.md#-license)
  explicitly applies Apache-2.0 to code and pretrained weights and permits
  commercial use. The converter records that explicit first-party grant and
  preserves the conflicting card tag; it does not assert a dual-licence choice.
- `pixelhacker-weights-model-card.md`: author HF revision
  `012fd343158936a265b8a0ee38a791a7a2841f45`, with an MIT declaration for
  that model repository. PixelHacker source code has its separate Apache-2.0
  licence above.
- `sdxl-vae-model-card.md`: Stability AI's standalone SDXL VAE model card,
  revision `6f5909a7e596173e25d4e97b07fd19cdf9611c76`, declaring MIT.
  The accepted VAE checkpoint is the separately hashed file hosted by the
  PixelHacker authors, not a byte-identity claim about the standalone upload.
  No separate `LICENSE` file is present in that standalone revision; the
  first-party card is retained without inventing a copyright notice.

The converter creates a sidecar licence/provenance directory for each actual
GGUF. Preserve that directory and its conversion receipt when redistributing
a converted model, along with the applicable upstream terms. A converted-file
digest is produced only after the full checkpoint and conversion have been
verified; no generated-weight digest or model asset is supplied here.
