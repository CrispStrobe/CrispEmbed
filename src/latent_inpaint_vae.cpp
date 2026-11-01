#include "latent_inpaint_vae.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace latent_inpaint {
namespace {
Tensor residual(Engine & engine, const std::string & name, const Tensor & input) {
    Tensor x = engine.conv(name + ".conv1", silu(engine.group_norm(name + ".norm1", input, 32, 1e-6f)), 1, 1);
    x = engine.conv(name + ".conv2", silu(engine.group_norm(name + ".norm2", x, 32, 1e-6f)), 1, 1);
    x = add(x, engine.has(name + ".conv_shortcut.weight") ? engine.conv(name + ".conv_shortcut", input) : input);
    engine.mark(name + ".output", x);
    return x;
}

Tensor attention(Engine & engine, const std::string & name, const Tensor & input) {
    Tensor normalized = engine.group_norm(name + ".group_norm", input, 32, 1e-6f);
    Tensor q = engine.linear(name + ".to_q", normalized);
    Tensor k = engine.linear(name + ".to_k", normalized);
    Tensor v = engine.linear(name + ".to_v", normalized);
    Tensor x = engine.attention(q, k, v, 1);
    x = add(engine.linear(name + ".to_out.0", x), input);
    engine.mark(name + ".output", x);
    return x;
}

Tensor middle(Engine & engine, const std::string & prefix, const Tensor & input) {
    Tensor x = residual(engine, prefix + ".resnets.0", input);
    x = attention(engine, prefix + ".attentions.0", x);
    return residual(engine, prefix + ".resnets.1", x);
}
} // namespace

Tensor vae_encode_moments(Engine & engine, const Tensor & rgb) {
    if (rgb.c != 3 || rgb.h != 512 || rgb.w != 512 || !finite(rgb))
        throw std::runtime_error("SDXL VAE encoder requires finite RGB512 CHW");
    Tensor x = engine.conv("encoder.conv_in", rgb, 1, 1);
    for (int block = 0; block < 4; ++block) {
        const std::string name = "encoder.down_blocks." + std::to_string(block);
        for (int layer = 0; layer < 2; ++layer) x = residual(engine, name + ".resnets." + std::to_string(layer), x);
        if (block < 3) {
            // Diffusers Downsample2D(padding=0) pads only right and bottom.
            x = engine.conv(name + ".downsamplers.0.conv", pad(x, 0, 1, 0, 1), 2, 0);
        }
    }
    x = middle(engine, "encoder.mid_block", x);
    x = engine.conv("encoder.conv_out", silu(engine.group_norm("encoder.conv_norm_out", x, 32, 1e-6f)), 1, 1);
    x = engine.conv("quant_conv", x);
    engine.mark("vae.encoder.moments", x);
    return x;
}

Tensor vae_sample(const Tensor & moments, const Tensor & posterior_noise) {
    if (moments.c != 8 || posterior_noise.c != 4 || moments.h != posterior_noise.h || moments.w != posterior_noise.w ||
        !finite(moments) || !finite(posterior_noise))
        throw std::runtime_error("VAE posterior shape or values differ");
    Tensor result(4, moments.h, moments.w);
    for (size_t i = 0; i < result.v.size(); ++i) {
        const float logvar = std::clamp(moments.v[result.v.size() + i], -30.0f, 20.0f);
        result.v[i] = (moments.v[i] + std::exp(0.5f * logvar) * posterior_noise.v[i]) * vae_scaling_factor;
    }
    return result;
}

Tensor vae_decode(Engine & engine, const Tensor & latent) {
    if (latent.c != 4 || latent.h != 64 || latent.w != 64 || !finite(latent))
        throw std::runtime_error("SDXL VAE decoder requires finite unscaled4x64x64 latent");
    Tensor x = engine.conv("post_quant_conv", latent);
    x = engine.conv("decoder.conv_in", x, 1, 1);
    x = middle(engine, "decoder.mid_block", x);
    for (int block = 0; block < 4; ++block) {
        const std::string name = "decoder.up_blocks." + std::to_string(block);
        for (int layer = 0; layer < 3; ++layer) x = residual(engine, name + ".resnets." + std::to_string(layer), x);
        if (block < 3) x = engine.conv(name + ".upsamplers.0.conv", resize_nearest(x, x.h * 2, x.w * 2), 1, 1);
    }
    x = engine.conv("decoder.conv_out", silu(engine.group_norm("decoder.conv_norm_out", x, 32, 1e-6f)), 1, 1);
    engine.mark("vae.decoder.output", x);
    return x;
}
} // namespace latent_inpaint
