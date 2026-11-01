// Device graph transcription of the same pinned modules as
// latent_inpaint_unet.cpp. No activation is downloaded or evaluated on the
// host while constructing or executing these denoiser graphs.
#include "latent_inpaint_graph_unet.h"

#include "inpaint_graph_spectral.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace latent_inpaint {
namespace {

using inpaint_graph::Builder;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("latent inpaint graph UNet: " + message);
}

void check_tensor(const ggml_tensor * x, const char * name) {
    require(x && x->type == GGML_TYPE_F32, std::string(name) + " must be an F32 tensor");
    for (int i = 0; i < 4; ++i) require(x->ne[i] > 0, std::string(name) + " has an invalid dimension");
}

bool same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < 4; ++i)
        if (a->ne[i] != b->ne[i]) return false;
    return true;
}

ggml_tensor * contiguous(Builder & b, ggml_tensor * x) {
    return ggml_is_contiguous(x) ? x : ggml_cont(b.ctx, x);
}

ggml_tensor * matrix(Builder & b, ggml_tensor * a, ggml_tensor * x) {
    // Keep strict accumulation explicit. The caller's separately gated mixed
    // profile permits the selected backend's optimized matrix precision.
    auto * out = ggml_mul_mat(b.ctx, a, x);
    ggml_mul_mat_set_prec(out, b.fast() ? GGML_PREC_DEFAULT : GGML_PREC_F32);
    return out;
}

// Convert [N,head_feature,heads,B] to the common contiguous WHCN layout.
ggml_tensor * image_from_heads(Builder & b, ggml_tensor * x, int64_t width, int64_t height) {
    require(x->ne[0] == width * height, "head output has the wrong spatial extent");
    return ggml_reshape_4d(b.ctx, contiguous(b, x), width, height, x->ne[1] * x->ne[2], x->ne[3]);
}

// Q [W,H,heads*dk,B] -> [dk,heads,N,B], for a per-pixel positional reduction.
ggml_tensor * position_queries(Builder & b, ggml_tensor * q, int heads) {
    auto * split = ggml_reshape_4d(b.ctx, contiguous(b, q), q->ne[0] * q->ne[1], q->ne[2] / heads, heads, q->ne[3]);
    return ggml_cont(b.ctx, ggml_permute(b.ctx, split, 2, 0, 1, 3));
}

// lambda [dk,dv,N,B] times Q [dk,heads,N,B] -> [W,H,heads*dv,B].
ggml_tensor * position_product(Builder & b, ggml_tensor * q, ggml_tensor * lambda, int heads) {
    auto * y = matrix(b, lambda, position_queries(b, q, heads)); // [dv,heads,N,B]
    y = ggml_permute(b.ctx, y, 1, 2, 0, 3);                      // [N,dv,heads,B]
    return image_from_heads(b, y, q->ne[0], q->ne[1]);
}

ggml_tensor * channel_half_product(Builder & b, ggml_tensor * x, bool use_gelu) {
    require(x->ne[2] % 2 == 0, "feed-forward projection must have an even channel count");
    const int channels = int(x->ne[2] / 2);
    auto * value = b.channel_slice(x, 0, channels);
    auto * gate = b.channel_slice(x, channels, channels);
    gate = use_gelu ? ggml_gelu_erf(b.ctx, gate) : ggml_silu(b.ctx, gate);
    return ggml_mul(b.ctx, value, gate);
}

ggml_tensor * depthwise_separable(Builder & b, const std::string & name, ggml_tensor * x) {
    auto * out = b.conv(name + ".conv_dw", x, 1, 1, int(x->ne[2]));
    out = ggml_relu(b.ctx, b.batch_norm(name + ".bn1", out, 1e-5f));
    out = b.conv(name + ".conv_pw", out);
    out = b.batch_norm(name + ".bn2", out, 1e-5f);
    // timm DepthwiseSeparableConv has an internal residual for equal widths.
    // This is separate from the surrounding DWResnetBlock2D skip.
    if (x->ne[2] == out->ne[2]) out = ggml_add(b.ctx, out, x);
    b.mark(name, out);
    return out;
}

ggml_tensor * resnet(Builder & b, Variant variant, const std::string & name, ggml_tensor * x,
                     ggml_tensor * activated_time) {
    auto * out = ggml_silu(b.ctx, b.group_norm(name + ".norm1", x, 32, 1e-5f));
    out =
        variant == Variant::moebius ? depthwise_separable(b, name + ".conv1", out) : b.conv(name + ".conv1", out, 1, 1);
    if (variant == Variant::pixelhacker) b.mark(name + ".conv1", out);
    auto * projected_time = b.linear(name + ".time_emb_proj", activated_time);
    require(projected_time->ne[0] == 1 && projected_time->ne[1] == 1 && projected_time->ne[2] == out->ne[2] &&
                projected_time->ne[3] == out->ne[3],
            "invalid residual time projection");
    out = ggml_add(b.ctx, out, projected_time);
    out = ggml_silu(b.ctx, b.group_norm(name + ".norm2", out, 32, 1e-5f));
    out =
        variant == Variant::moebius ? depthwise_separable(b, name + ".conv2", out) : b.conv(name + ".conv2", out, 1, 1);
    if (variant == Variant::pixelhacker) b.mark(name + ".conv2", out);
    auto * skip = b.has(name + ".conv_shortcut.weight") ? b.conv(name + ".conv_shortcut", x) : x;
    require(same_shape(out, skip), "residual shortcut has the wrong shape");
    out = ggml_add(b.ctx, out, skip);
    b.mark(name, out);
    return out;
}

ggml_tensor * lambda_self(Builder & b, const std::string & name, ggml_tensor * x) {
    auto * q = b.batch_norm(name + ".norm_q", b.conv(name + ".to_q", x), 1e-5f);
    auto * k = b.conv(name + ".to_k", x);
    auto * v = b.batch_norm(name + ".norm_v", b.conv(name + ".to_v", x), 1e-5f);
    constexpr int heads = 8;
    auto * content = graph_detail::lambda_content(b, q, k, v, heads);
    const int64_t dk = q->ne[2] / heads;
    auto * kernel = b.weight(name + ".pos_conv.weight");
    require(kernel->ne[0] == 15 && kernel->ne[1] == 15 && kernel->ne[2] == 1 && kernel->ne[3] == dk,
            "unsupported local lambda positional kernel");
    // Original Conv3d has singleton depth. Folding V channels into the batch
    // applies the same 15x15 K filters (and bias) to every value plane in one
    // backend convolution, with neither a host loop nor an activation copy.
    auto * planes = ggml_reshape_4d(b.ctx, contiguous(b, v), v->ne[0], v->ne[1], 1, v->ne[2] * v->ne[3]);
    auto * positional = b.conv(name + ".pos_conv", planes, 1, 7);
    auto * out = ggml_add(b.ctx, content, graph_detail::lambda_position(b, q, positional, heads, int(v->ne[2])));
    b.mark(name, out);
    return out;
}

ggml_tensor * lambda_cross(Builder & b, const std::string & name, ggml_tensor * x, ggml_tensor * cond) {
    auto * q = b.batch_norm(name + ".norm_q", b.conv(name + ".to_q", x), 1e-5f);
    auto * k = b.linear(name + ".to_k", cond, false);
    auto * v = b.batch_norm(name + ".norm_v", b.linear(name + ".to_v", cond, false), 1e-5f);
    auto * content = graph_detail::lambda_content(b, q, k, v, 8);
    auto * positional = graph_detail::lambda_relative(b, q, v, b.weight(name + ".rel_pos_emb"), 8);
    auto * out = ggml_add(b.ctx, content, positional);
    b.mark(name, out);
    return out;
}

ggml_tensor * gated_self(Builder & b, const std::string & name, ggml_tensor * x) {
    auto * q = b.linear(name + ".q_proj", x, false);
    auto * k = b.linear(name + ".k_proj", x, false);
    auto * v = b.linear(name + ".v_proj", x, false);
    auto * gate_k = b.linear(name + ".gk_proj.1", b.linear(name + ".gk_proj.0", x, false));
    auto * out = inpaint_graph::gla(b, q, k, v, gate_k, 8, 16.0f);
    b.mark(name + ".recurrent", out);
    auto * gate = b.linear(name + ".g_proj", x, false);
    out = graph_detail::gated_rms_norm(b, out, gate, b.weight(name + ".g_norm_swish_gate.weight"), 8);
    b.mark(name + ".g_norm_swish_gate", out);
    out = b.linear(name + ".o_proj", out, false);
    b.mark(name, out);
    return out;
}

ggml_tensor * standard_cross(Builder & b, const std::string & name, ggml_tensor * x, ggml_tensor * cond) {
    auto * q = b.linear(name + ".to_q", x, false);
    auto * k = b.linear(name + ".to_k", cond, false);
    auto * v = b.linear(name + ".to_v", cond, false);
    auto * out = b.linear(name + ".to_out.0", b.attention(q, k, v, 8));
    b.mark(name, out);
    return out;
}

ggml_tensor * feed_forward(Builder & b, Variant variant, const std::string & name, ggml_tensor * x) {
    ggml_tensor * out = nullptr;
    if (variant == Variant::moebius) {
        out = ggml_silu(b.ctx, b.conv(name + ".inverted_conv.conv", x));
        out = b.conv(name + ".depth_conv.conv", out, 1, 1, int(out->ne[2]));
        out = b.conv(name + ".point_conv.conv", channel_half_product(b, out, false));
    } else {
        out = b.linear(name + ".net.2", channel_half_product(b, b.linear(name + ".net.0.proj", x), true));
    }
    b.mark(name, out);
    return out;
}

ggml_tensor * transformer(Builder & b, Variant variant, const std::string & name, ggml_tensor * input,
                          ggml_tensor * cond) {
    auto * x = b.conv(name + ".proj_in", b.group_norm(name + ".norm", input, 32, 1e-6f));
    const std::string block = name + ".transformer_blocks.0";
    auto * n = b.layer_norm(block + ".norm1", x, 1e-5f);
    x = ggml_add(b.ctx, x,
                 variant == Variant::moebius ? lambda_self(b, block + ".attn1", n)
                                             : gated_self(b, block + ".attn1", n));
    n = b.layer_norm(block + ".norm2", x, 1e-5f);
    x = ggml_add(b.ctx, x,
                 variant == Variant::moebius ? lambda_cross(b, block + ".attn2", n, cond)
                                             : standard_cross(b, block + ".attn2", n, cond));
    x = ggml_add(b.ctx, x, feed_forward(b, variant, block + ".ff", b.layer_norm(block + ".norm3", x, 1e-5f)));
    b.mark(block, x);
    x = ggml_add(b.ctx, b.conv(name + ".proj_out", x), input);
    b.mark(name, x);
    return x;
}

} // namespace

namespace graph_detail {

ggml_tensor * lambda_content(Builder & b, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int heads) {
    check_tensor(q, "lambda query");
    check_tensor(k, "lambda key");
    check_tensor(v, "lambda value");
    require(heads > 0 && heads <= 32 && q->ne[2] % heads == 0 && q->ne[2] / heads == k->ne[2],
            "invalid lambda query/key head dimensions");
    require(k->ne[0] * k->ne[1] == v->ne[0] * v->ne[1] && q->ne[3] == k->ne[3] && k->ne[3] == v->ne[3],
            "incompatible lambda sequence or batch dimensions");
    const int64_t n = q->ne[0] * q->ne[1], m = k->ne[0] * k->ne[1], dk = k->ne[2], dv = v->ne[2], batch = q->ne[3];
    auto * keys = ggml_reshape_4d(b.ctx, contiguous(b, k), m, dk, 1, batch);
    keys = ggml_soft_max(b.ctx, keys); // spatial key axis, not channels
    auto * values = ggml_reshape_4d(b.ctx, contiguous(b, v), m, dv, 1, batch);
    auto * lambda = matrix(b, keys, values); // [dk,dv,1,B], shared by heads
    auto * queries = ggml_reshape_4d(b.ctx, contiguous(b, q), n, dk, heads, batch);
    queries = ggml_cont(b.ctx, ggml_permute(b.ctx, queries, 1, 0, 2, 3)); // [dk,N,heads,B]
    auto * out = matrix(b, lambda, queries);                              // [dv,N,heads,B]
    out = ggml_permute(b.ctx, out, 1, 0, 2, 3);
    return image_from_heads(b, out, q->ne[0], q->ne[1]);
}

ggml_tensor * lambda_position(Builder & b, ggml_tensor * q, ggml_tensor * positional, int heads, int value_channels) {
    check_tensor(q, "local lambda query");
    check_tensor(positional, "local lambda position");
    require(heads > 0 && heads <= 32 && q->ne[2] % heads == 0 && value_channels > 0,
            "invalid local lambda head dimensions");
    const int64_t n = q->ne[0] * q->ne[1], dk = q->ne[2] / heads;
    require(positional->ne[0] == q->ne[0] && positional->ne[1] == q->ne[1] && positional->ne[2] == dk &&
                positional->ne[3] == value_channels * q->ne[3],
            "invalid batched positional convolution shape");
    auto * lambda = ggml_reshape_4d(b.ctx, contiguous(b, positional), n, dk, value_channels, q->ne[3]);
    lambda = ggml_cont(b.ctx, ggml_permute(b.ctx, lambda, 2, 0, 1, 3)); // [dk,dv,N,B]
    return position_product(b, q, lambda, heads);
}

ggml_tensor * lambda_relative(Builder & b, ggml_tensor * q, ggml_tensor * v, ggml_tensor * relative, int heads) {
    check_tensor(q, "relative lambda query");
    check_tensor(v, "relative lambda value");
    require(relative && (relative->type == GGML_TYPE_F32 || (b.fast() && relative->type == GGML_TYPE_F16)),
            "relative lambda table must be F32 or explicitly selected fast-profile F16");
    require(heads > 0 && heads <= 32 && q->ne[2] % heads == 0 && q->ne[3] == v->ne[3],
            "invalid relative lambda head or batch dimensions");
    const int64_t n = q->ne[0] * q->ne[1], m = v->ne[0] * v->ne[1], dk = q->ne[2] / heads;
    require(relative->ne[0] == 1 && relative->ne[1] == dk && relative->ne[2] == m && relative->ne[3] == n,
            "learned relative lambda table does not match the query/key geometry");
    // Flatten (key feature, query position) into one matrix-output axis. This
    // avoids materializing N copies of the ten-token value sequence, while
    // broadcasting the immutable learned table across both guidance samples.
    auto * table = ggml_reshape_4d(b.ctx, contiguous(b, relative), dk, m, n, 1);
    table = ggml_cont(b.ctx, ggml_permute(b.ctx, table, 1, 0, 2, 3)); // [M,dk,N,1]
    table = ggml_reshape_4d(b.ctx, table, m, dk * n, 1, 1);
    auto * values = ggml_reshape_4d(b.ctx, contiguous(b, v), m, v->ne[2], 1, v->ne[3]);
    auto * lambda = matrix(b, table, values); // [dk*N,dv,1,B]
    lambda = ggml_reshape_4d(b.ctx, lambda, dk, n, v->ne[2], v->ne[3]);
    lambda = ggml_cont(b.ctx, ggml_permute(b.ctx, lambda, 0, 2, 1, 3)); // [dk,dv,N,B]
    return position_product(b, q, lambda, heads);
}

ggml_tensor * gated_rms_norm(Builder & b, ggml_tensor * x, ggml_tensor * gate, ggml_tensor * weight, int heads) {
    check_tensor(x, "gated RMSNorm input");
    check_tensor(gate, "gated RMSNorm gate");
    check_tensor(weight, "gated RMSNorm weight");
    require(heads > 0 && heads <= 32 && x->ne[2] % heads == 0 && same_shape(x, gate),
            "invalid gated RMSNorm head or gate dimensions");
    const int64_t n = x->ne[0] * x->ne[1], dv = x->ne[2] / heads;
    require(weight->ne[0] == dv && weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1,
            "gated RMSNorm scale must be shared across heads");
    auto * split = ggml_reshape_4d(b.ctx, contiguous(b, x), n, dv, heads, x->ne[3]);
    split = ggml_cont(b.ctx, ggml_permute(b.ctx, split, 1, 0, 2, 3)); // [dv,N,heads,B]
    split = ggml_mul(b.ctx, ggml_rms_norm(b.ctx, split, 1e-6f), weight);
    auto * out = image_from_heads(b, ggml_permute(b.ctx, split, 1, 0, 2, 3), x->ne[0], x->ne[1]);
    return ggml_mul(b.ctx, out, ggml_silu(b.ctx, gate));
}

} // namespace graph_detail

ggml_tensor * graph_denoise(Builder & b, Variant variant, ggml_tensor * input9, ggml_tensor * time320,
                            ggml_tensor * tokens) {
    require(variant == Variant::moebius || variant == Variant::pixelhacker, "unknown model variant");
    check_tensor(input9, "denoiser input");
    check_tensor(time320, "sinusoidal time embedding");
    check_tensor(tokens, "learned category tokens");
    const int64_t batch = input9->ne[3];
    require(input9->ne[0] == 64 && input9->ne[1] == 64 && input9->ne[2] == 9 && (batch == 1 || batch == 2),
            "denoiser requires a 9x64x64 latent with one or two samples");
    require(time320->ne[0] == 1 && time320->ne[1] == 1 && time320->ne[2] == 320 && time320->ne[3] == batch,
            "denoiser requires a separate 320-channel time embedding per sample");
    require(tokens->ne[0] == 10 && tokens->ne[1] == 1 && tokens->ne[2] == 3072 && tokens->ne[3] == batch,
            "denoiser requires ten 3072-channel learned tokens per sample");

    b.mark("diff_model.time_proj", time320);
    auto * time = b.linear("diff_model.time_embedding.linear_1", time320);
    time = b.linear("diff_model.time_embedding.linear_2", ggml_silu(b.ctx, time));
    require(time->ne[0] == 1 && time->ne[1] == 1 && time->ne[2] == 1280 && time->ne[3] == batch,
            "unsupported time embedding projection");
    b.mark("diff_model.time_embedding", time);
    // All residual blocks consume this identical activation. Share one graph
    // node rather than recomputing SiLU for every time projection.
    auto * activated_time = ggml_silu(b.ctx, time);
    b.mark("embedding_layer", tokens);
    auto * cond = b.linear("diff_model.encoder_hid_proj", tokens);
    require(cond->ne[0] == 10 && cond->ne[1] == 1 && cond->ne[2] == 768 && cond->ne[3] == batch,
            "unsupported learned category projection");
    b.mark("diff_model.encoder_hid_proj", cond);

    auto * x = variant == Variant::moebius ? depthwise_separable(b, "diff_model.conv_in", input9)
                                           : b.conv("diff_model.conv_in", input9, 1, 1);
    if (variant == Variant::pixelhacker) b.mark("diff_model.conv_in", x);
    const int blocks = variant == Variant::moebius ? 3 : 4;
    std::vector<ggml_tensor *> residuals;
    residuals.reserve(size_t(3 * blocks));
    residuals.push_back(x);
    for (int i = 0; i < blocks; ++i) {
        const std::string block = "diff_model.down_blocks." + std::to_string(i);
        const bool has_attention = variant == Variant::moebius || i < 3;
        for (int r = 0; r < 2; ++r) {
            x = resnet(b, variant, block + ".resnets." + std::to_string(r), x, activated_time);
            if (has_attention) x = transformer(b, variant, block + ".attentions." + std::to_string(r), x, cond);
            residuals.push_back(x);
        }
        if (i < blocks - 1) {
            x = b.conv(block + ".downsamplers.0.conv", x, 2, 1);
            b.mark(block + ".downsamplers.0", x);
            residuals.push_back(x);
        }
        b.mark(block, x);
    }
    if (variant == Variant::pixelhacker) {
        x = resnet(b, variant, "diff_model.mid_block.resnets.0", x, activated_time);
        x = transformer(b, variant, "diff_model.mid_block.attentions.0", x, cond);
        x = resnet(b, variant, "diff_model.mid_block.resnets.1", x, activated_time);
        b.mark("diff_model.mid_block", x);
    }
    for (int i = 0; i < blocks; ++i) {
        const std::string block = "diff_model.up_blocks." + std::to_string(i);
        const bool has_attention = variant == Variant::moebius || i > 0;
        for (int r = 0; r < 3; ++r) {
            require(!residuals.empty(), "UNet skip stack underflow");
            auto * skip = residuals.back();
            residuals.pop_back();
            require(x->ne[0] == skip->ne[0] && x->ne[1] == skip->ne[1] && x->ne[3] == skip->ne[3],
                    "UNet skip spatial or batch dimensions differ");
            x = resnet(b, variant, block + ".resnets." + std::to_string(r), ggml_concat(b.ctx, x, skip, 2),
                       activated_time);
            if (has_attention) x = transformer(b, variant, block + ".attentions." + std::to_string(r), x, cond);
        }
        if (i < blocks - 1) {
            x = b.conv(block + ".upsamplers.0.conv", b.resize(x, int(x->ne[1] * 2), int(x->ne[0] * 2)), 1, 1);
            b.mark(block + ".upsamplers.0", x);
        }
        b.mark(block, x);
    }
    require(residuals.empty(), "UNet skip stack contains unconsumed activations");
    x = b.group_norm("diff_model.conv_norm_out", x, 32, 1e-5f);
    b.mark("diff_model.conv_norm_out", x);
    x = ggml_silu(b.ctx, x);
    x = variant == Variant::moebius ? depthwise_separable(b, "diff_model.conv_out", x)
                                    : b.conv("diff_model.conv_out", x, 1, 1);
    require(x->ne[0] == 64 && x->ne[1] == 64 && x->ne[2] == 4 && x->ne[3] == batch, "invalid denoiser output geometry");
    if (variant == Variant::pixelhacker) b.mark("diff_model.conv_out", x);
    b.mark("diff_model", x);
    return x;
}

} // namespace latent_inpaint
