---
license: mit
tags:
- stable-diffusion
- stable-diffusion-diffusers
inference: false
---
# SDXL - VAE

#### How to use with 🧨 diffusers
You can integrate this fine-tuned VAE decoder to your existing `diffusers` workflows, by including a `vae` argument to the `StableDiffusionPipeline`
```py
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

model = "stabilityai/your-stable-diffusion-model"
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
```

## Model 
[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) is a [latent diffusion model](https://arxiv.org/abs/2112.10752), where the diffusion operates in a pretrained, 
learned (and fixed) latent space of an autoencoder. 
While the bulk of the semantic composition is done by the latent diffusion model, 
we can improve _local_, high-frequency details in generated images by improving the quality of the autoencoder. 
To this end, we train the same autoencoder architecture used for the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) at a larger batch-size (256 vs 9) 
and additionally track the weights with an exponential moving average (EMA). 
The resulting autoencoder outperforms the original model in all evaluated reconstruction metrics, see the table below.


## Evaluation 
_SDXL-VAE vs original kl-f8 VAE vs f8-ft-MSE_
### COCO 2017 (256x256, val, 5000 images)
| Model    | rFID | PSNR         | SSIM          | PSIM          | Link                                                                                                 | Comments                                                                                        
|----------|------|--------------|---------------|---------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
|          |      |              |               |               |                                                                                                      |                                                                                                 |
| SDXL-VAE | 4.42 | 24.7 +/- 3.9 | 0.73 +/- 0.13 | 0.88 +/- 0.27 | https://huggingface.co/stabilityai/sdxl-vae/blob/main/sdxl_vae.safetensors                                                                                                     | as used in SDXL                                                                                 |
| original | 4.99 | 23.4 +/- 3.8 | 0.69 +/- 0.14 | 1.01 +/- 0.28 | https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                                               | as used in SD                                                                                   |
| ft-MSE   | 4.70 | 24.5 +/- 3.7 | 0.71 +/- 0.13 | 0.92 +/- 0.27 | https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt | resumed with EMA from ft-EMA, emphasis on MSE (rec. loss = MSE + 0.1 * LPIPS), smoother outputs |
