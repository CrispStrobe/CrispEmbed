// Architectural transcription of the original author inference modules:
//   hustvl/Moebius b88d462bacb9af6e7128a3b4cc4a07418bedfd61
//   hustvl/PixelHacker f5567db2871598aa178fe7a34c520dd478a0b41b
// See docs/latent-inpaint-unets.md for exact operations and pending evidence.
#include "latent_inpaint_unet.h"

#include "core/cpu_ops.h"
#include "core/env_gate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace latent_inpaint {
namespace {

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("latent inpaint UNet: " + message);
}

void check_tensor(const Tensor & x, const char * name) {
    require(x.c > 0 && x.c <= 8192 && x.h > 0 && x.w > 0 && x.h <= 4096 && x.w <= 4096,
            std::string(name) + " has invalid dimensions");
    require(x.spatial() <= 4096 && x.v.size() == size_t(x.c) * x.spatial(),
            std::string(name) + " has invalid storage or exceeds the fixed-512 latent limit");
    require(finite(x), std::string(name) + " contains nonfinite values");
}

void check_heads(const Tensor & q, const Tensor & k, const Tensor & v, int heads) {
    check_tensor(q, "query");
    check_tensor(k, "key");
    check_tensor(v, "value");
    require(heads > 0 && heads <= 32 && q.c % heads == 0 && v.c % heads == 0, "invalid attention head dimensions");
    require(q.c == k.c && k.spatial() == v.spatial(), "incompatible attention key/value dimensions");
}

Tensor relu(Tensor x) {
    for (float & v : x.v) v = std::max(v, 0.0f);
    return x;
}

Tensor broadcast_add(Tensor x, const Tensor & bias) {
    require(bias.c == x.c && bias.h == 1 && bias.w == 1, "invalid channel bias");
    const size_t n = x.spatial();
    for (int c = 0; c < x.c; ++c) {
        float * row = x.v.data() + size_t(c) * n;
        const float b = bias.v[size_t(c)];
        for (size_t p = 0; p < n; ++p) row[p] += b;
    }
    return x;
}

Tensor channel_half_product(const Tensor & x, bool use_gelu) {
    require(x.c % 2 == 0, "feed-forward projection cannot be split into two equal halves");
    Tensor out(x.c / 2, x.h, x.w);
    const size_t n = out.v.size();
    // GEGLU uses PyTorch's exact erf GELU. Moebius' GLUMBConv gates with SiLU.
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    for (size_t i = 0; i < n; ++i) {
        const float gate = x.v[n + i];
        const float activated =
            use_gelu ? 0.5f * gate * (1.0f + std::erf(gate * inv_sqrt2)) : gate / (1.0f + std::exp(-gate));
        out.v[i] = x.v[i] * activated;
    }
    return out;
}

Tensor depthwise_separable(Engine & e, const std::string & name, const Tensor & x) {
    // timm DepthwiseSeparableConv has its own BN/ReLU and, for equal channel
    // counts, an INTERNAL residual. DWResnetBlock2D adds a second outer skip.
    Tensor out = e.conv(name + ".conv_dw", x, 1, 1, x.c);
    out = relu(e.batch_norm(name + ".bn1", out, 1e-5f));
    out = e.conv(name + ".conv_pw", out);
    out = e.batch_norm(name + ".bn2", out, 1e-5f);
    if (x.c == out.c) out = add(out, x);
    e.mark(name, out);
    return out;
}

Tensor resnet(Engine & e, Variant variant, const std::string & name, const Tensor & x, const Tensor & time) {
    Tensor out = silu(e.group_norm(name + ".norm1", x, 32, 1e-5f));
    out =
        variant == Variant::moebius ? depthwise_separable(e, name + ".conv1", out) : e.conv(name + ".conv1", out, 1, 1);
    if (variant == Variant::pixelhacker) e.mark(name + ".conv1", out);
    const Tensor projected_time = e.linear(name + ".time_emb_proj", silu(time));
    out = broadcast_add(std::move(out), projected_time);
    out = silu(e.group_norm(name + ".norm2", out, 32, 1e-5f));
    out =
        variant == Variant::moebius ? depthwise_separable(e, name + ".conv2", out) : e.conv(name + ".conv2", out, 1, 1);
    if (variant == Variant::pixelhacker) e.mark(name + ".conv2", out);
    const Tensor skip = e.has(name + ".conv_shortcut.weight") ? e.conv(name + ".conv_shortcut", x) : x;
    out = add(out, skip); // all supported author profiles have output_scale_factor=1
    e.mark(name, out);
    return out;
}

Tensor lambda_self(Engine & e, const std::string & name, const Tensor & x) {
    const Tensor q = e.batch_norm(name + ".norm_q", e.conv(name + ".to_q", x), 1e-5f);
    const Tensor k = e.conv(name + ".to_k", x);
    const Tensor v = e.batch_norm(name + ".norm_v", e.conv(name + ".to_v", x), 1e-5f);
    constexpr int heads = 8;
    Tensor out = detail::lambda_content(q, k, v, heads);
    const int dk = q.c / heads;
    const size_t n = q.spatial();
    const Weight & pos = e.weight(name + ".pos_conv.weight");
    require(pos.shape == std::vector<int64_t>({ dk, 1, 15, 15 }), "unsupported local lambda positional kernel");
    // Conv3d uses a singleton depth kernel. Each V feature plane is therefore
    // an independent 2-D convolution with the same K filters, including bias.
    // Stream these planes to avoid materializing K*V*H*W lambda activations.
    Tensor plane(1, x.h, x.w);
    std::vector<float> positional_sum(n, 0.0f);
    for (int value = 0; value < v.c; ++value) {
        std::copy_n(v.v.data() + size_t(value) * n, n, plane.v.data());
        const Tensor positional = e.conv(name + ".pos_conv", plane, 1, 7);
        require(positional.c == dk && positional.h == x.h && positional.w == x.w,
                "local lambda positional convolution shape mismatch");
        for (int head = 0; head < heads; ++head) {
            float * dst = out.v.data() + size_t(head * v.c + value) * n;
            std::fill(positional_sum.begin(), positional_sum.end(), 0.0f);
            for (int key = 0; key < dk; ++key) {
                const float * qq = q.v.data() + size_t(head * dk + key) * n;
                const float * pp = positional.v.data() + size_t(key) * n;
                for (size_t p = 0; p < n; ++p) positional_sum[p] += qq[p] * pp[p];
            }
            for (size_t p = 0; p < n; ++p) dst[p] += positional_sum[p];
        }
    }
    e.mark(name, out);
    return out;
}

Tensor lambda_cross(Engine & e, const std::string & name, const Tensor & x, const Tensor & cond) {
    const Tensor q = e.batch_norm(name + ".norm_q", e.conv(name + ".to_q", x), 1e-5f);
    const Tensor k = e.linear(name + ".to_k", cond, false);
    const Tensor v = e.batch_norm(name + ".norm_v", e.linear(name + ".to_v", cond, false), 1e-5f);
    Tensor out = detail::lambda_content(q, k, v, 8);
    out = add(out, detail::lambda_relative(q, v, e.weight(name + ".rel_pos_emb"), 8));
    e.mark(name, out);
    return out;
}

Tensor gated_self(Engine & e, const std::string & name, const Tensor & x) {
    const Tensor q = e.linear(name + ".q_proj", x, false);
    const Tensor k = e.linear(name + ".k_proj", x, false);
    const Tensor v = e.linear(name + ".v_proj", x, false);
    const Tensor gate_k = e.linear(name + ".gk_proj.1", e.linear(name + ".gk_proj.0", x, false));
    Tensor out = detail::gla_recurrent(q, k, v, gate_k, 8, 16.0f);
    e.mark(name + ".recurrent", out);
    const Tensor gate = e.linear(name + ".g_proj", x, false);
    const Weight & norm = e.weight(name + ".g_norm_swish_gate.weight");
    constexpr int heads = 8;
    const int dv = out.c / heads;
    require(norm.shape == std::vector<int64_t>({ dv }) && gate.c == out.c && gate.spatial() == out.spatial(),
            "unsupported gated RMSNorm dimensions");
    std::vector<float> head_value(size_t(dv), 0.0f);
    const size_t n = out.spatial();
    for (int head = 0; head < heads; ++head) {
        for (size_t p = 0; p < n; ++p) {
            for (int d = 0; d < dv; ++d) head_value[size_t(d)] = out.v[size_t(head * dv + d) * n + p];
            core_cpu::rmsnorm_cpu(head_value.data(), head_value.data(), dv, norm.data, 1e-6f);
            for (int d = 0; d < dv; ++d) {
                const size_t i = size_t(head * dv + d) * n + p;
                const float g = gate.v[i];
                out.v[i] = head_value[size_t(d)] * g * (1.0f / (1.0f + std::exp(-g)));
            }
        }
    }
    e.mark(name + ".g_norm_swish_gate", out);
    out = e.linear(name + ".o_proj", out, false);
    e.mark(name, out);
    return out;
}

Tensor standard_cross(Engine & e, const std::string & name, const Tensor & x, const Tensor & cond) {
    const Tensor q = e.linear(name + ".to_q", x, false);
    const Tensor k = e.linear(name + ".to_k", cond, false);
    const Tensor v = e.linear(name + ".to_v", cond, false);
    Tensor out = e.attention(q, k, v, 8);
    out = e.linear(name + ".to_out.0", out);
    e.mark(name, out);
    return out;
}

Tensor feed_forward(Engine & e, Variant variant, const std::string & name, const Tensor & x) {
    Tensor out;
    if (variant == Variant::moebius) {
        out = silu(e.conv(name + ".inverted_conv.conv", x));
        out = e.conv(name + ".depth_conv.conv", out, 1, 1, out.c);
        out = channel_half_product(out, false);
        out = e.conv(name + ".point_conv.conv", out);
    } else {
        out = channel_half_product(e.linear(name + ".net.0.proj", x), true);
        out = e.linear(name + ".net.2", out);
    }
    e.mark(name, out);
    return out;
}

Tensor transformer(Engine & e, Variant variant, const std::string & name, const Tensor & input, const Tensor & cond) {
    Tensor x = e.group_norm(name + ".norm", input, 32, 1e-6f);
    x = e.conv(name + ".proj_in", x);
    const std::string block = name + ".transformer_blocks.0";
    Tensor n = e.layer_norm(block + ".norm1", x, 1e-5f);
    x = add(x, variant == Variant::moebius ? lambda_self(e, block + ".attn1", n) : gated_self(e, block + ".attn1", n));
    n = e.layer_norm(block + ".norm2", x, 1e-5f);
    x = add(x, variant == Variant::moebius ? lambda_cross(e, block + ".attn2", n, cond)
                                           : standard_cross(e, block + ".attn2", n, cond));
    x = add(x, feed_forward(e, variant, block + ".ff", e.layer_norm(block + ".norm3", x, 1e-5f)));
    e.mark(block, x);
    x = add(e.conv(name + ".proj_out", x), input);
    e.mark(name, x);
    return x;
}

Tensor conditioning(Engine & e, bool conditional) {
    const Weight & embedding = e.weight("embedding_layer.weight");
    require(embedding.shape == std::vector<int64_t>({ 20, 3072 }), "unsupported learned category embedding");
    Tensor tokens(3072, 1, 10);
    const int start = conditional ? 0 : 10;
    for (int t = 0; t < 10; ++t)
        for (int c = 0; c < 3072; ++c) tokens.v[size_t(c) * 10 + t] = embedding.data[size_t(start + t) * 3072 + c];
    e.mark("embedding_layer", tokens);
    tokens = e.linear("diff_model.encoder_hid_proj", tokens);
    require(tokens.c == 768 && tokens.w == 10 && tokens.h == 1, "unsupported projected category embedding");
    e.mark("diff_model.encoder_hid_proj", tokens);
    return tokens;
}

} // namespace

namespace detail {

Tensor timestep_embedding(int timestep, int channels) {
    require(timestep >= 0 && timestep < 1000 && channels > 0 && channels <= 320 && channels % 2 == 0,
            "unsupported timestep embedding");
    Tensor out(channels, 1, 1);
    const int half = channels / 2;
    // Diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0.
    const float log_period = std::log(10000.0f);
    for (int i = 0; i < half; ++i) {
        const float frequency = std::exp((-log_period * float(i)) / float(half));
        const float phase = float(timestep) * frequency;
        out.v[size_t(i)] = std::cos(phase);
        out.v[size_t(half + i)] = std::sin(phase);
    }
    return out;
}

Tensor gla_recurrent(const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & gate_logits, int heads,
                     float gate_normalizer) {
    check_heads(q, k, v, heads);
    check_tensor(gate_logits, "key gate");
    require(q.h == k.h && q.w == k.w && q.h == v.h && q.w == v.w && gate_logits.c == k.c && gate_logits.h == k.h &&
                gate_logits.w == k.w,
            "GLA requires aligned raster-ordered query/key/value/gate tokens");
    require(std::isfinite(gate_normalizer) && gate_normalizer > 0, "invalid GLA gate normalizer");
    const int dk = q.c / heads, dv = v.c / heads;
    const size_t n = q.spatial();
    Tensor out(v.c, q.h, q.w);
    const float scale = 1.0f / std::sqrt(float(dk));
    std::vector<float> state(size_t(dk) * dv, 0.0f), value(size_t(dv), 0.0f), result(size_t(dv), 0.0f);
    for (int head = 0; head < heads; ++head) {
        std::fill(state.begin(), state.end(), 0.0f);
        for (size_t token = 0; token < n; ++token) {
            for (int d = 0; d < dv; ++d) value[size_t(d)] = v.v[size_t(head * dv + d) * n + token];
            std::fill(result.begin(), result.end(), 0.0f);
            for (int d = 0; d < dk; ++d) {
                const size_t index = size_t(head * dk + d) * n + token;
                const float logit = gate_logits.v[index];
                const float log_sigmoid =
                    logit >= 0.0f ? -std::log1p(std::exp(-logit)) : logit - std::log1p(std::exp(logit));
                const float decay = std::exp(log_sigmoid / gate_normalizer);
                const float key = k.v[index], query = q.v[index] * scale;
                float * row = state.data() + size_t(d) * dv;
                for (int dvalue = 0; dvalue < dv; ++dvalue) {
                    row[dvalue] = row[dvalue] * decay + key * value[size_t(dvalue)];
                    result[size_t(dvalue)] += query * row[dvalue];
                }
            }
            for (int d = 0; d < dv; ++d) out.v[size_t(head * dv + d) * n + token] = result[size_t(d)];
        }
    }
    return out;
}

Tensor lambda_content(const Tensor & q, const Tensor & k, const Tensor & v, int heads) {
    check_tensor(q, "lambda query");
    check_tensor(k, "lambda key");
    check_tensor(v, "lambda value");
    require(heads > 0 && heads <= 32 && q.c == heads * k.c && k.spatial() == v.spatial(),
            "unsupported multi-query lambda dimensions");
    const int dk = k.c, dv = v.c;
    const size_t n = q.spatial(), m = k.spatial();
    Tensor keys = k;
    for (int d = 0; d < dk; ++d) core_cpu::softmax(keys.v.data() + size_t(d) * m, int(m));
    std::vector<float> content(size_t(dk) * dv, 0.0f);
    for (int d = 0; d < dk; ++d)
        for (int value = 0; value < dv; ++value)
            content[size_t(d) * dv + value] =
                core_cpu::dot_product(keys.v.data() + size_t(d) * m, v.v.data() + size_t(value) * m, int(m));
    Tensor out(heads * dv, q.h, q.w);
    for (int head = 0; head < heads; ++head)
        for (int value = 0; value < dv; ++value) {
            float * dst = out.v.data() + size_t(head * dv + value) * n;
            for (int d = 0; d < dk; ++d) {
                const float coefficient = content[size_t(d) * dv + value];
                const float * qq = q.v.data() + size_t(head * dk + d) * n;
                for (size_t p = 0; p < n; ++p) dst[p] += qq[p] * coefficient;
            }
        }
    return out;
}

Tensor lambda_relative(const Tensor & q, const Tensor & v, const Weight & relative, int heads) {
    check_tensor(q, "lambda query");
    check_tensor(v, "lambda value");
    require(heads > 0 && heads <= 32 && q.c % heads == 0, "invalid relative lambda head count");
    const int dk = q.c / heads, dv = v.c;
    const size_t n = q.spatial(), m = v.spatial();
    require(relative.shape == std::vector<int64_t>({ int64_t(n), int64_t(m), dk, 1 }) && relative.data != nullptr &&
                relative.size == n * m * size_t(dk),
            "relative lambda embedding does not match the fixed spatial grid");
    Tensor out(heads * dv, q.h, q.w);
    std::vector<float> positional(size_t(dk) * dv, 0.0f);
    for (size_t p = 0; p < n; ++p) {
        std::fill(positional.begin(), positional.end(), 0.0f);
        for (size_t token = 0; token < m; ++token) {
            const float * rel = relative.data + (p * m + token) * size_t(dk);
            for (int d = 0; d < dk; ++d)
                for (int value = 0; value < dv; ++value)
                    positional[size_t(d) * dv + value] += rel[d] * v.v[size_t(value) * m + token];
        }
        for (int head = 0; head < heads; ++head)
            for (int value = 0; value < dv; ++value) {
                float sum = 0.0f;
                for (int d = 0; d < dk; ++d)
                    sum += q.v[size_t(head * dk + d) * n + p] * positional[size_t(d) * dv + value];
                out.v[size_t(head * dv + value) * n + p] = sum;
            }
    }
    return out;
}

} // namespace detail

Tensor denoise(Engine & e, Variant variant, const Tensor & input9, int timestep, bool conditional) {
    require(variant == Variant::moebius || variant == Variant::pixelhacker, "unknown model variant");
    const char * gate = variant == Variant::moebius ? "CRISPEMBED_MOEBIUS" : "CRISPEMBED_PIXELHACKER";
    require(core_env::on(gate), std::string("experimental path requires ") + gate + "=1");
    check_tensor(input9, "denoiser input");
    require(input9.c == 9 && input9.h == 64 && input9.w == 64, "denoiser requires exactly 9x64x64 latent input");
    Tensor time = detail::timestep_embedding(timestep);
    e.mark("diff_model.time_proj", time);
    time = e.linear("diff_model.time_embedding.linear_1", time);
    time = e.linear("diff_model.time_embedding.linear_2", silu(std::move(time)));
    require(time.c == 1280 && time.h == 1 && time.w == 1, "unsupported time embedding dimensions");
    e.mark("diff_model.time_embedding", time);
    const Tensor cond = conditioning(e, conditional);

    Tensor x = variant == Variant::moebius ? depthwise_separable(e, "diff_model.conv_in", input9)
                                           : e.conv("diff_model.conv_in", input9, 1, 1);
    if (variant == Variant::pixelhacker) e.mark("diff_model.conv_in", x);
    const int blocks = variant == Variant::moebius ? 3 : 4;
    std::vector<Tensor> residuals;
    residuals.reserve(size_t(3 * blocks));
    residuals.push_back(x);
    for (int b = 0; b < blocks; ++b) {
        const std::string block = "diff_model.down_blocks." + std::to_string(b);
        const bool has_attention = variant == Variant::moebius || b < 3;
        for (int r = 0; r < 2; ++r) {
            x = resnet(e, variant, block + ".resnets." + std::to_string(r), x, time);
            if (has_attention) x = transformer(e, variant, block + ".attentions." + std::to_string(r), x, cond);
            residuals.push_back(x);
        }
        if (b < blocks - 1) {
            x = e.conv(block + ".downsamplers.0.conv", x, 2, 1);
            e.mark(block + ".downsamplers.0", x);
            residuals.push_back(x);
        }
        e.mark(block, x);
    }
    if (variant == Variant::pixelhacker) {
        x = resnet(e, variant, "diff_model.mid_block.resnets.0", x, time);
        x = transformer(e, variant, "diff_model.mid_block.attentions.0", x, cond);
        x = resnet(e, variant, "diff_model.mid_block.resnets.1", x, time);
        e.mark("diff_model.mid_block", x);
    }
    for (int b = 0; b < blocks; ++b) {
        const std::string block = "diff_model.up_blocks." + std::to_string(b);
        const bool has_attention = variant == Variant::moebius || b > 0;
        for (int r = 0; r < 3; ++r) {
            require(!residuals.empty(), "UNet skip stack underflow");
            Tensor combined = concat(x, residuals.back());
            residuals.pop_back();
            x = resnet(e, variant, block + ".resnets." + std::to_string(r), combined, time);
            if (has_attention) x = transformer(e, variant, block + ".attentions." + std::to_string(r), x, cond);
        }
        if (b < blocks - 1) {
            x = e.conv(block + ".upsamplers.0.conv", resize_nearest(x, x.h * 2, x.w * 2), 1, 1);
            e.mark(block + ".upsamplers.0", x);
        }
        e.mark(block, x);
    }
    require(residuals.empty(), "UNet skip stack has unconsumed tensors");
    x = e.group_norm("diff_model.conv_norm_out", x, 32, 1e-5f);
    e.mark("diff_model.conv_norm_out", x);
    x = silu(std::move(x));
    x = variant == Variant::moebius ? depthwise_separable(e, "diff_model.conv_out", x)
                                    : e.conv("diff_model.conv_out", x, 1, 1);
    require(x.c == 4 && x.h == 64 && x.w == 64 && finite(x), "invalid denoiser output");
    if (variant == Variant::pixelhacker) e.mark("diff_model.conv_out", x);
    e.mark("diff_model", x);
    return x;
}

} // namespace latent_inpaint
