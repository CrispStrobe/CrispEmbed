#include "latent_inpaint_graph.h"
#include "latent_inpaint_graph_unet.h"
#include "latent_inpaint_schema.h"
#include "latent_inpaint_vae.h"
#include "inpaint_graph.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace latent_inpaint {
namespace {
using inpaint_graph::Builder;
ggml_tensor * residual(Builder & b, const std::string & name, ggml_tensor * input) {
    ggml_tensor * x = b.conv(name + ".conv1", ggml_silu(b.ctx, b.group_norm(name + ".norm1", input, 32, 1e-6f)), 1, 1);
    x = b.conv(name + ".conv2", ggml_silu(b.ctx, b.group_norm(name + ".norm2", x, 32, 1e-6f)), 1, 1);
    x = ggml_add(b.ctx, x, b.has(name + ".conv_shortcut.weight") ? b.conv(name + ".conv_shortcut", input) : input);
    return x;
}
ggml_tensor * middle(Builder & b, const std::string & name, ggml_tensor * input) {
    ggml_tensor * x = residual(b, name + ".resnets.0", input);
    const std::string attn = name + ".attentions.0";
    ggml_tensor * n = b.group_norm(attn + ".group_norm", x, 32, 1e-6f);
    ggml_tensor * y =
        b.attention(b.linear(attn + ".to_q", n), b.linear(attn + ".to_k", n), b.linear(attn + ".to_v", n));
    x = ggml_add(b.ctx, x, b.linear(attn + ".to_out.0", y));
    return residual(b, name + ".resnets.1", x);
}
ggml_tensor * encode(Builder & b, ggml_tensor * rgb) {
    ggml_tensor * x = b.conv("encoder.conv_in", rgb, 1, 1);
    for (int block = 0; block < 4; ++block) {
        const std::string name = "encoder.down_blocks." + std::to_string(block);
        for (int layer = 0; layer < 2; ++layer) x = residual(b, name + ".resnets." + std::to_string(layer), x);
        if (block < 3) x = b.conv(name + ".downsamplers.0.conv", b.pad(x, 0, 1, 0, 1), 2, 0);
    }
    x = middle(b, "encoder.mid_block", x);
    x = b.conv("encoder.conv_out", ggml_silu(b.ctx, b.group_norm("encoder.conv_norm_out", x, 32, 1e-6f)), 1, 1);
    return b.conv("quant_conv", x);
}
ggml_tensor * decode(Builder & b, ggml_tensor * latent) {
    ggml_tensor * x = b.conv("decoder.conv_in", b.conv("post_quant_conv", latent), 1, 1);
    x = middle(b, "decoder.mid_block", x);
    for (int block = 0; block < 4; ++block) {
        const std::string name = "decoder.up_blocks." + std::to_string(block);
        for (int layer = 0; layer < 3; ++layer) x = residual(b, name + ".resnets." + std::to_string(layer), x);
        if (block < 3) x = b.conv(name + ".upsamplers.0.conv", b.resize(x, int(x->ne[1]) * 2, int(x->ne[0]) * 2), 1, 1);
    }
    return b.conv("decoder.conv_out", ggml_silu(b.ctx, b.group_norm("decoder.conv_norm_out", x, 32, 1e-6f)), 1, 1);
}
ggml_tensor * batch_slice(Builder & b, ggml_tensor * x, int batch) {
    return ggml_cont(b.ctx, ggml_view_4d(b.ctx, x, x->ne[0], x->ne[1], x->ne[2], 1, x->nb[1], x->nb[2], x->nb[3],
                                         size_t(batch) * x->nb[3]));
}
} // namespace

struct GraphPipeline::Impl {
    Options options;
    std::vector<int> timesteps;
    std::vector<float> alphas;
    inpaint_graph::Device device;
    inpaint_graph::Weights denoiser, vae;
    inpaint_graph::State state;
    ggml_tensor *latent, *latent_mask, *masked_latent;
    std::unique_ptr<Builder> encoder, stepper, decoder;
    ggml_tensor *image = nullptr, *mask = nullptr, *posterior = nullptr, *noise = nullptr, *offset = nullptr;
    ggml_tensor *time = nullptr, *coefficients = nullptr, *decoded = nullptr;
    ggml_tensor *eps_u = nullptr, *eps_c = nullptr, *eps = nullptr, *next = nullptr;
    size_t step = 0;
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    Observer observer = nullptr;
    void * opaque = nullptr;
    void mark(const std::string & name, ggml_tensor * tensor, const std::vector<float> & values) {
        if (!observer) return;
        Tensor value(int(tensor->ne[2]), int(tensor->ne[1]), int(tensor->ne[0]));
        const size_t count = value.v.size();
        if (tensor->ne[3] == 2) {
            if (step != 0) return;
            std::copy_n(values.begin(), count, value.v.begin());
            observer(opaque, "unet.step.0.unconditional." + name, value);
            std::copy_n(values.begin() + count, count, value.v.begin());
            observer(opaque, "unet.step.0.conditional." + name, value);
        } else {
            value.v = values;
            observer(opaque, name, value);
        }
    }
    static void observe(void * self, const std::string & name, ggml_tensor * tensor,
                        const std::vector<float> & values) {
        static_cast<Impl *>(self)->mark(name, tensor, values);
    }
    void observer_for(Builder & b) { b.set_observer(observer ? observe : nullptr, this); }
#else
    void observer_for(Builder &) {}
#endif
    Impl(const Options & settings, const std::vector<int> & steps, const std::vector<float> & alpha)
        : options(settings), timesteps(steps), alphas(alpha), device(settings.threads),
          denoiser(device, settings.denoiser_path.c_str(), denoiser_schema(settings.model)),
          vae(device, settings.vae_path.c_str(), vae_schema()), state(device) {
        latent = state.tensor("latent.state", { 64, 64, 4, 1 });
        latent_mask = state.tensor("latent.mask", { 64, 64, 1, 1 });
        masked_latent = state.tensor("latent.masked", { 64, 64, 4, 1 });
        state.allocate();
    }
    void build_encoder() {
        encoder = std::make_unique<Builder>(device, &vae);
        Builder & b = *encoder;
        observer_for(b);
        image = b.parameter("image", { 512, 512, 3, 1 });
        mask = b.parameter("mask", { 512, 512, 1, 1 });
        posterior = b.parameter("posterior.noise", { 64, 64, 4, 2 });
        noise = b.parameter("latent.noise", { 64, 64, 4, 1 });
        offset = b.parameter("offset.noise", { 1, 1, 4, 1 });
        ggml_tensor * keep = ggml_sub(b.ctx, ggml_repeat(b.ctx, b.constant("one", { 1 }, { 1 }), mask), mask);
        ggml_tensor * masked = ggml_mul(b.ctx, image, keep);
        // Encode image and masked image together using the same resident weights.
        ggml_tensor * moments = encode(b, ggml_concat(b.ctx, image, masked, 3));
        ggml_tensor * mean = b.channel_slice(moments, 0, 4);
        ggml_tensor * logvar = ggml_clamp(b.ctx, b.channel_slice(moments, 4, 4), -30, 20);
        ggml_tensor * sampled = ggml_scale(
            b.ctx, ggml_add(b.ctx, mean, ggml_mul(b.ctx, ggml_exp(b.ctx, ggml_scale(b.ctx, logvar, 0.5f)), posterior)),
            vae_scaling_factor);
        ggml_tensor * image_latent = batch_slice(b, sampled, 0);
        ggml_tensor * masked_value = batch_slice(b, sampled, 1);
        b.mark("vae.image.moments", batch_slice(b, moments, 0));
        b.mark("vae.masked.moments", batch_slice(b, moments, 1));
        b.mark("vae.image.latents", image_latent);
        b.mark("vae.masked.latents", masked_value);
        const float a = alphas.at(size_t(timesteps.front()));
        ggml_tensor * added = ggml_add(b.ctx, noise, ggml_scale(b.ctx, offset, options.noise_offset));
        ggml_tensor * initial =
            ggml_add(b.ctx, ggml_scale(b.ctx, image_latent, std::sqrt(a)), ggml_scale(b.ctx, added, std::sqrt(1 - a)));
        b.mark("latent.initial", initial);
        b.finish({ ggml_cpy(b.ctx, initial, latent), ggml_cpy(b.ctx, masked_value, masked_latent),
                   ggml_cpy(b.ctx, b.resize(mask, 64, 64), latent_mask) },
                 "vae-encoder-batch2");
    }
    void build_stepper() {
        stepper = std::make_unique<Builder>(device, &denoiser, 131072);
        Builder & b = *stepper;
        observer_for(b);
        time = b.parameter("timestep.embedding", { 1, 1, 320, 1 });
        coefficients = b.parameter("ddim.coefficients", { 4 });
        ggml_tensor * tokens = b.weight("embedding_layer.weight"); // 3072,20
        if (tokens->type != GGML_TYPE_F32) tokens = ggml_cast(b.ctx, tokens, GGML_TYPE_F32);
        ggml_tensor * cond = ggml_view_2d(b.ctx, tokens, 3072, 10, tokens->nb[1], 0);
        ggml_tensor * uncond = ggml_view_2d(b.ctx, tokens, 3072, 10, tokens->nb[1], 10 * tokens->nb[1]);
        tokens = ggml_reshape_4d(b.ctx, ggml_concat(b.ctx, uncond, cond, 1), 3072, 10, 1, 2);
        tokens = ggml_cont(b.ctx, ggml_permute(b.ctx, tokens, 2, 0, 1, 3)); // 10,1,3072,2
        ggml_tensor * input = ggml_concat(b.ctx, ggml_concat(b.ctx, latent, latent_mask, 2), masked_latent, 2);
        input = ggml_concat(b.ctx, input, input, 3);
        ggml_tensor * prediction =
            graph_denoise(b, options.model == "moebius" ? Variant::moebius : Variant::pixelhacker, input,
                          ggml_concat(b.ctx, time, time, 3), tokens);
        eps_u = batch_slice(b, prediction, 0);
        eps_c = batch_slice(b, prediction, 1);
        eps = ggml_add(b.ctx, eps_u, ggml_scale(b.ctx, ggml_sub(b.ctx, eps_c, eps_u), options.guidance));
        auto coefficient = [&](int i) { return ggml_view_1d(b.ctx, coefficients, 1, size_t(i) * sizeof(float)); };
        // Retain the original DDIM evaluation order (predict x0, then combine)
        // instead of algebraically fusing coefficients and changing rounding.
        ggml_tensor * original =
            ggml_div(b.ctx, ggml_sub(b.ctx, latent, ggml_mul(b.ctx, eps, coefficient(0))), coefficient(1));
        next = ggml_add(b.ctx, ggml_mul(b.ctx, original, coefficient(2)), ggml_mul(b.ctx, eps, coefficient(3)));
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
        if (observer) {
            for (ggml_tensor * value : { eps_u, eps_c, eps, next }) ggml_set_output(value);
        }
#endif
        b.finish({ ggml_cpy(b.ctx, next, latent) }, "unet-cfg-ddim-batch2");
    }
    void build_decoder() {
        decoder = std::make_unique<Builder>(device, &vae);
        Builder & b = *decoder;
        observer_for(b);
        decoded = decode(b, ggml_scale(b.ctx, latent, 1.0f / vae_scaling_factor));
        b.finish({ decoded }, "vae-decoder");
    }
};

GraphPipeline::GraphPipeline(const Options & options, const std::vector<int> & timesteps,
                             const std::vector<float> & alphas)
    : impl_(new Impl(options, timesteps, alphas)) {}
GraphPipeline::~GraphPipeline() = default;

Tensor GraphPipeline::run(const Tensor & image, const Tensor & mask, const Tensor & posterior_image,
                          const Tensor & posterior_masked, const Tensor & latent_noise, const Tensor & offset_noise) {
    auto & p = *impl_;
    if (!p.encoder) {
        try {
            p.build_encoder();
        } catch (...) {
            p.encoder.reset();
            throw;
        }
    }
    p.encoder->upload(p.image, image.v);
    p.encoder->upload(p.mask, mask.v);
    std::vector<float> posterior = posterior_image.v;
    posterior.insert(posterior.end(), posterior_masked.v.begin(), posterior_masked.v.end());
    p.encoder->upload(p.posterior, posterior);
    p.encoder->upload(p.noise, latent_noise.v);
    p.encoder->upload(p.offset, offset_noise.v);
    p.encoder->compute();
    if (!p.stepper) {
        try {
            p.build_stepper();
        } catch (...) {
            p.stepper.reset();
            throw;
        }
    }
    for (p.step = 0; p.step < p.timesteps.size(); ++p.step) {
        const int t = p.timesteps[p.step], prev = t - 1000 / p.options.steps;
        const float a = p.alphas.at(size_t(t)), ap = prev >= 0 ? p.alphas.at(size_t(prev)) : 1.0f;
        p.stepper->upload(p.time, detail::timestep_embedding(t, 320).v);
        p.stepper->upload(p.coefficients, { std::sqrt(1 - a), std::sqrt(a), std::sqrt(ap), std::sqrt(1 - ap) });
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
        p.stepper->set_observer(p.step == 0 && p.observer ? Impl::observe : nullptr, &p);
#endif
        p.stepper->compute();
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
        if (p.observer) {
            p.mark("unet.step" + std::to_string(p.step) + ".unconditional", p.eps_u, p.stepper->download(p.eps_u));
            p.mark("unet.step" + std::to_string(p.step) + ".conditional", p.eps_c, p.stepper->download(p.eps_c));
            p.mark("sampler.noise." + std::to_string(p.step), p.eps, p.stepper->download(p.eps));
            p.mark("sampler.step." + std::to_string(p.step), p.next, p.stepper->download(p.next));
        }
#endif
    }
    if (!p.decoder) {
        try {
            p.build_decoder();
        } catch (...) {
            p.decoder.reset();
            throw;
        }
    }
    p.decoder->compute();
    Tensor result(3, 512, 512);
    result.v = p.decoder->download(p.decoded);
    if (!finite(result)) throw std::runtime_error("inpainting graph produced nonfinite decoded RGB");
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    p.mark("vae.decoded", p.decoded, result.v);
#endif
    return result;
}
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
void GraphPipeline::set_observer(Observer callback, void * opaque) {
    if (impl_->observer != callback || impl_->opaque != opaque) {
        impl_->encoder.reset();
        impl_->stepper.reset();
        impl_->decoder.reset();
    }
    impl_->observer = callback;
    impl_->opaque = opaque;
}
#endif
} // namespace latent_inpaint
