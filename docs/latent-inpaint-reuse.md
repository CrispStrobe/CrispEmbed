# What Moebius and PixelHacker reuse in CrispEmbed

The experimental implementation loads one Places2 denoiser and the shared
SDXL VAE as F32 GGUF files. The retained eager path runs on the existing CPU backend and requires
the model-specific `CRISPEMBED_MOEBIUS=1` or `CRISPEMBED_PIXELHACKER=1` gate.
Both gates are off by default. No text encoder is required: the original
models use a learned, checkpoint-resident `embedding_layer.weight` with
20 vectors, split into ten conditional and ten unconditional vectors.

The additional [persistent graph path](inpainting-ggml-graphs.md) reuses the
same contracts and surrounding CLI while executing complete components on
native CPU, CUDA or Metal. It replaces per-operator allocations and host
activation loops with cached graphs, batched guidance and resident sampler
state. The table below documents the original eager baseline; the graph guide
describes the additional backend/runtime reuse and its pending qualification.

The exact supported source revisions, state dictionaries and inference
profiles are in the [source contract](latent-inpaint-source-contract.json).
The [conversion guide](latent-inpaint-conversion.md) describes the independent
original-framework capture and the evidence needed before claiming native
parity. This implementation has not yet passed those full-model gates.

## Existing components used directly

| Existing CrispEmbed component | Use in this implementation | New work around it |
| --- | --- | --- |
| `src/core/gguf_loader.*` | `core_gguf::open_metadata`, `load_weights` and `free_weights` own model metadata, CPU storage and mmap where available. | A frozen native tensor-name/shape census, exact metadata types and provenance checks, F32-only validation, and non-owning weight views in `latent_inpaint::Engine`. |
| `src/core/ggml_cpu_backend.*` | The same CPU backend and thread-count controls execute small ggml operator graphs. | Bounded patch extraction for convolution, channel-first tensor adapters, and explicit per-operator allocation in `src/latent_inpaint_ops.cpp`. |
| `ggml_mul_mat`, `ggml_conv_2d_dw_direct`, `ggml_flash_attn_ext` | Existing matrix multiplication, direct depthwise convolution and scaled dot-product attention implement the applicable numerical primitives. | Model-specific weight layouts, attention head layouts, F32 precision requests and output reshaping. This is a CPU path; an accelerator backend has not been qualified. |
| `src/core/cpu_ops.*` | `linear_cpu`, `layernorm_cpu`, `rmsnorm_cpu`, `softmax` and `dot_product` are reused where their definitions match the original operations. | GroupNorm, evaluation BatchNorm, exact-erf GELU, tensor concatenation/resizing and the model-specific attention recurrences. |
| `src/core/sha256.*`, `src/core/env_gate.h` | Actual GGUF files are hashed before loading, and the existing value-parsed environment gates control entry. | The loader compares the supplied GGUF digest with the conversion receipt, then separately checks the frozen upstream checkpoint, source, component, schema, layout and input-contract metadata. |
| `src/inpaint.cpp` / `inpaint::detail::process_tiles_rgb8_variant` | The established tile traversal, overlapping accumulation, input padding and exact preservation of unmasked RGB pixels are shared. | A 512-pixel latent-model tile callback. Tiling is an explicit option because independent diffusion tiles change image-level context. |
| `examples/cli/inpaint_cli.cpp` | Existing PNG/image handling, binary-mask checks, rectangle/text masks, DBNet selection, TSV/directory processing, protected-path checks and output provenance remain the surrounding CLI flow. | Engine selection, VAE and digest arguments, diffusion settings, fixed-512 geometry checks, and dispatch to a persistent latent context. |

The CLI reuses one loaded model context across a batch. Each frame is still
processed independently; this provides no temporal consistency or video-model
semantics. Known pixels are copied from the original RGB8 input after generated
pixels are composed under the binary fill mask. The original models' optional
Gaussian paste and compensation are disabled for the parity profile.

## Conversion and verification patterns extended

The converter is framework-free. It extends the allow-listed, inert PyTorch
ZIP metadata reader pattern from `tools/prepare_migan_inpaint.py` and the
minimal F32 GGUF conventions from `tools/migan_reference_contract.py`. The
implementation streams tensor payloads and readback hashes instead of holding
an additional multi-GB state dictionary in Python memory.

The full input checkpoint must match its frozen byte count and SHA-256 before
its pickle metadata is interpreted. Tensor reconstruction yields inert storage
and shape records; foreign pickle globals are rejected. The converter then
checks every original name, dtype, shape, storage size, offset and stride.
Only two explicit transformations are allowed: omit Moebius's 124 scalar
BatchNorm training counters, and squeeze the singleton depth axis of its
15 named positional convolutions. The shared VAE's FP16 values are widened
exactly to F32. Original F32 tensor bytes remain unchanged.

The original-reference capture extends `tools/dump_migan_reference.py`'s
module hooks, owned input fixtures, output hashes and hook-neutrality check.
Diffusion adds captured full-image and masked-image VAE posterior noise,
initial diffusion noise, per-channel offset noise, and every DDIM step.
Those explicit noise tensors are replayed by the native test. A matching seed
alone would not establish matching random inputs across different runtimes.

The native census in `src/latent_inpaint_schema.h` is generated from the frozen
original checkpoint metadata. It is compared independently with the GGUF
header; self-declared tensor metadata in a converted file is not accepted as
the schema authority. The conversion receipt binds the actual GGUF byte count
and SHA-256 and records all exported tensor hashes. Conversion success does
not mark a native model as qualified.

## Model mathematics added

`src/latent_inpaint_unet.cpp` implements the original models' inference graphs
with their original state names. Moebius needs its local and cross lambda
attention, depthwise residual blocks and GLUMBConv feed-forward blocks.
PixelHacker needs gated linear attention, including its causal recurrent
state, gate normalization and activation; its ordinary cross attention uses
the existing ggml attention primitive. Both graphs include the learned
conditioning embeddings, timestep projection, skip stack and original
down/up-sampling order. These graphs cannot be substituted by an existing
text-embedding or image-embedding model graph.

`src/latent_inpaint_vae.cpp` implements the shared AutoencoderKL encoder,
posterior moments and sampling, and decoder. Its block counts, normalization
epsilon, asymmetric downsample padding, single-head mid-block attention and
scaling factor 0.13025 are part of the pinned original-source contract.

`src/latent_inpaint.cpp` adds the original scaled-linear DDIM schedule,
classifier-free guidance, masked-image conditioning and deterministic native
noise stream. The declared 20-step profiles execute 19 timesteps after the
original strength slicing. The default guidance is 2.0 for Moebius and 4.5
for PixelHacker. Full parity uses identical explicit noise rather than trying
to reproduce PyTorch's seed stream with the native generator.

## Weight footprint and experimental status

Sizes below are calculated from the strict original tensor census, using
decimal GB. Headers and alignment add a small amount to the converted files.

| Profile | Published denoiser plus VAE files | F32 tensor payload after conversion |
| --- | ---: | ---: |
| Moebius Places2 + SDXL VAE | 1,072,692,662 bytes | 1,239,392,880 bytes, about **1.24 GB** |
| PixelHacker Places2 + SDXL VAE | 3,616,739,746 bytes | 3,783,622,284 bytes, about **3.78 GB** |

The same VAE can be shared between the two profiles. Its published checkpoint
is 167,394,306 bytes; its F32 tensor payload is 334,615,452 bytes. The increase
comes from widening FP16 weights, not an additional model component.

These values describe stored weights. They do not establish peak resident
memory, temporary activation memory, throughput, latency, or representative
image quality. The CPU implementation uses bounded operator workspaces and
reuses the GGUF loader's mmap support where possible, but those design choices
do not substitute for measuring a complete run. F16 and quantized GGUF files
are not included in the accepted conversion contract. The graph's explicit
mixed GPU profile derives selected F16 weights at load from that unchanged,
digest-verified F32 file and needs its own numerical/image qualification.

The bounded parser/converter controls can run without a framework or model.
Full builds, full checkpoint conversion, original inference and native parity
belong on hosted workers under the repository's contribution policy. The
default-off status remains appropriate until those jobs produce retained,
passing evidence under the frozen acceptance budgets.

## Licensing and retained upstream notices

| Component | First-party statement used | Recorded qualification |
| --- | --- | --- |
| Moebius code and weights | The [pinned author README](https://github.com/hustvl/Moebius/blob/b88d462bacb9af6e7128a3b4cc4a07418bedfd61/README.md#-license) explicitly grants Apache-2.0 for both code and pretrained weights, including commercial use. | The [author weight card](https://huggingface.co/hustvl/Moebius/blob/cd01f47fb648219d3fa605806c5ce00b713faa5e/README.md) instead has an MIT tag. This discrepancy is preserved. The converter records the explicit Apache-2.0 grant; it does not infer a dual-licence election from the conflicting tag. |
| PixelHacker code | The [pinned repository licence](https://github.com/hustvl/PixelHacker/blob/f5567db2871598aa178fe7a34c520dd478a0b41b/LICENSE) is Apache-2.0. | This code licence is recorded separately from the weight card. |
| PixelHacker weights | The [author weight card](https://huggingface.co/hustvl/PixelHacker/blob/012fd343158936a265b8a0ee38a791a7a2841f45/README.md) declares MIT. | The supported model is the exact Places2 checkpoint in the source contract. |
| Shared SDXL VAE weights | The author-hosted VAE is in the same MIT-tagged PixelHacker repository; the [standalone SDXL VAE card](https://huggingface.co/stabilityai/sdxl-vae/blob/6f5909a7e596173e25d4e97b07fd19cdf9611c76/README.md) also declares MIT. | The accepted payload is the exact author-hosted VAE digest, not a claim that all similarly named VAE files are interchangeable. |
| Diffusers, timm and Sana code used by the originals | Their upstream code is Apache-2.0. | Native architectural adaptations retain source attribution and licence texts. |
| Lambda Networks and Flash Linear Attention | Their upstream code is MIT. | Original notices for Phil Wang and Songlin Yang are retained. |

The source and binary redistribution notices are in
[`licenses/latent-inpaint/NOTICE.md`](../licenses/latent-inpaint/NOTICE.md),
with unchanged upstream licence texts and author weight cards. No checkpoint,
converted weight file, external framework package or generated model output
is bundled with these source changes. A hosted conversion also writes a
per-model licence/provenance directory beside its GGUF and receipt.
