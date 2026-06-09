// layout_detect.cpp — RT-DETRv2 document layout analysis via ggml.
//
// Hardcoded architecture (no graph replay):
//   1. ResNet-50 backbone (stem + 4 stages of Bottleneck blocks)
//   2. Hybrid encoder (lateral convs + FPN + PAN + AIFI encoder)
//   3. Transformer decoder (6 layers: self-attn + deformable cross-attn + FFN)
//   4. Detection heads (bbox + class per query)
//
// All BN is pre-folded by the converter. Deformable attention uses CPU-side
// bilinear grid sampling (no ggml op).

#include "layout_detect.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

// stb_image declarations
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
    void stbi_image_free(void *retval_from_stbi_load);
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace layout_detect {

// ---------------------------------------------------------------------------
// Label names
// ---------------------------------------------------------------------------

static const char* LABEL_NAMES[] = {
    "caption", "footnote", "formula", "list_item", "page_footer",
    "page_header", "picture", "section_header", "table", "text",
    "title", "document_index", "code", "checkbox_selected",
    "checkbox_unselected", "form", "key_value_region"
};

const char* label_name(label_id id) {
    int i = (int)id;
    if (i >= 0 && i < (int)label_id::NUM_CLASSES) return LABEL_NAMES[i];
    return "unknown";
}

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------

struct conv_w { ggml_tensor *w = nullptr, *b = nullptr; };

struct bottleneck {
    conv_w branch2a;  // 1×1 reduce
    conv_w branch2b;  // 3×3
    conv_w branch2c;  // 1×1 expand
    conv_w shortcut;  // 1×1 (only when dims change)
};

struct resnet50_backbone {
    conv_w stem[3];  // conv1_1, conv1_2, conv1_3 (all 3×3)
    // Stage 0: 3 blocks (64→256), Stage 1: 4 blocks (128→512)
    // Stage 2: 6 blocks (256→1024), Stage 3: 3 blocks (512→2048)
    std::vector<bottleneck> stages[4];
};

struct encoder_block {
    // RepVGG-style block (2 conv layers + optional identity)
    conv_w conv1, conv2;
};

struct hybrid_encoder {
    conv_w input_proj[3];     // project backbone features to 256d
    conv_w lateral_convs[2];  // for FPN
    conv_w downsample_convs[2]; // for PAN
    // FPN/PAN blocks (each has 2 convs)
    encoder_block fpn_blocks[2];
    encoder_block pan_blocks[2];
    // AIFI self-attention encoder
    ggml_tensor *aifi_qkv_w = nullptr, *aifi_qkv_b = nullptr;
    ggml_tensor *aifi_out_w = nullptr, *aifi_out_b = nullptr;
    ggml_tensor *aifi_norm1_w = nullptr, *aifi_norm1_b = nullptr;
    ggml_tensor *aifi_ffn1_w = nullptr, *aifi_ffn1_b = nullptr;
    ggml_tensor *aifi_ffn2_w = nullptr, *aifi_ffn2_b = nullptr;
    ggml_tensor *aifi_norm2_w = nullptr, *aifi_norm2_b = nullptr;
    ggml_tensor *pos_embed = nullptr; // 2D positional embedding
};

struct decoder_layer {
    // Self-attention
    ggml_tensor *self_qkv_w = nullptr, *self_qkv_b = nullptr;
    ggml_tensor *self_out_w = nullptr, *self_out_b = nullptr;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    // Deformable cross-attention
    ggml_tensor *cross_value_w = nullptr, *cross_value_b = nullptr;
    ggml_tensor *cross_sampling_offsets_w = nullptr, *cross_sampling_offsets_b = nullptr;
    ggml_tensor *cross_attn_weights_w = nullptr, *cross_attn_weights_b = nullptr;
    ggml_tensor *cross_out_w = nullptr, *cross_out_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    // FFN
    ggml_tensor *ffn1_w = nullptr, *ffn1_b = nullptr;
    ggml_tensor *ffn2_w = nullptr, *ffn2_b = nullptr;
    ggml_tensor *norm3_w = nullptr, *norm3_b = nullptr;
};

struct transformer_decoder {
    // Input projection (3 scales → 256d)
    conv_w input_proj[3];
    // Query initialization
    ggml_tensor *anchors = nullptr;      // [300, 4] reference points
    ggml_tensor *valid_mask = nullptr;   // [300, N] validity mask
    // Encoder output projection
    ggml_tensor *enc_proj_w = nullptr;
    ggml_tensor *enc_norm_w = nullptr, *enc_norm_b = nullptr;
    ggml_tensor *enc_score_w = nullptr, *enc_score_b = nullptr;
    ggml_tensor *enc_bbox_w[3] = {};     // MLP layers
    ggml_tensor *enc_bbox_b[3] = {};
    // Query position head (MLP)
    ggml_tensor *qpos_w[3] = {}, *qpos_b[3] = {};
    // Decoder layers
    decoder_layer layers[6];
    // Per-layer detection heads
    ggml_tensor *dec_score_w = nullptr, *dec_score_b = nullptr;
    ggml_tensor *dec_bbox_w[6][3] = {};
    ggml_tensor *dec_bbox_b[6][3] = {};
};

struct context {
    resnet50_backbone backbone;
    hybrid_encoder encoder;
    transformer_decoder decoder;

    // Preprocessing
    float img_mean[3] = {0.485f, 0.456f, 0.406f};
    float img_std[3] = {0.229f, 0.224f, 0.225f};
    int input_h = 640, input_w = 640;
    int num_queries = 300;
    int num_classes = 17;

    // Backend
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool load(context** out, const char* path, int n_threads) {
    auto* ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;

    fprintf(stderr, "layout_detect: loading %s\n", path);

    // Load weights
    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    if (!core_gguf::load_weights(path, ctx->backend, "layout", ctx->wl)) {
        fprintf(stderr, "layout_detect: failed to load weights\n");
        delete ctx;
        *out = nullptr;
        return false;
    }

    auto get = [&](const std::string& n) -> ggml_tensor* {
        auto it = ctx->wl.tensors.find(n);
        return it != ctx->wl.tensors.end() ? it->second : nullptr;
    };

    auto load_conv = [&](conv_w& c, const std::string& prefix) {
        c.w = get(prefix + ".weight");
        c.b = get(prefix + ".bias");
        if (!c.w) {
            // Try without .conv suffix (some layers vary)
            c.w = get(prefix + ".conv.weight");
            c.b = get(prefix + ".conv.bias");
        }
    };

    // --- Backbone ---
    auto& bb = ctx->backbone;
    load_conv(bb.stem[0], "model.backbone.conv1.conv1_1.conv");
    load_conv(bb.stem[1], "model.backbone.conv1.conv1_2.conv");
    load_conv(bb.stem[2], "model.backbone.conv1.conv1_3.conv");

    int block_counts[] = {3, 4, 6, 3};
    for (int s = 0; s < 4; s++) {
        bb.stages[s].resize(block_counts[s]);
        for (int b = 0; b < block_counts[s]; b++) {
            auto pfx = "model.backbone.res_layers." + std::to_string(s) +
                       ".blocks." + std::to_string(b);
            auto& blk = bb.stages[s][b];
            load_conv(blk.branch2a, pfx + ".branch2a.conv");
            load_conv(blk.branch2b, pfx + ".branch2b.conv");
            load_conv(blk.branch2c, pfx + ".branch2c.conv");
            // Shortcut — different key patterns for different stages
            load_conv(blk.shortcut, pfx + ".short.conv");
            if (!blk.shortcut.w) load_conv(blk.shortcut, pfx + ".short.conv.conv");
            if (!blk.shortcut.w) load_conv(blk.shortcut, pfx + ".short");
        }
    }

    // --- Encoder ---
    auto& enc = ctx->encoder;
    for (int i = 0; i < 3; i++)
        load_conv(enc.input_proj[i], std::string("model.encoder.input_proj.") + std::to_string(i));
    for (int i = 0; i < 2; i++) {
        load_conv(enc.lateral_convs[i], std::string("model.encoder.lateral_convs.") + std::to_string(i));
        load_conv(enc.downsample_convs[i], std::string("model.encoder.downsample_convs.") + std::to_string(i));
    }
    // FPN/PAN blocks (each block has 2 conv layers in a RepVGG style)
    for (int i = 0; i < 2; i++) {
        auto fpn = std::string("model.encoder.fpn_blocks.") + std::to_string(i);
        auto pan = std::string("model.encoder.pan_blocks.") + std::to_string(i);
        // These are CSP-style blocks — simplified as 2 convs
        // The actual structure is more complex; we'll handle this during graph building
        enc.fpn_blocks[i].conv1.w = get(fpn + ".conv1.conv.weight");
        enc.fpn_blocks[i].conv1.b = get(fpn + ".conv1.conv.bias");
        enc.fpn_blocks[i].conv2.w = get(fpn + ".conv2.conv.weight");
        enc.fpn_blocks[i].conv2.b = get(fpn + ".conv2.conv.bias");
        enc.pan_blocks[i].conv1.w = get(pan + ".conv1.conv.weight");
        enc.pan_blocks[i].conv1.b = get(pan + ".conv1.conv.bias");
        enc.pan_blocks[i].conv2.w = get(pan + ".conv2.conv.weight");
        enc.pan_blocks[i].conv2.b = get(pan + ".conv2.conv.bias");
    }
    // AIFI encoder
    std::string aifi = "model.encoder.encoder.0.layers.0";
    enc.aifi_qkv_w = get(aifi + ".self_attn.in_proj_weight");
    enc.aifi_qkv_b = get(aifi + ".self_attn.in_proj_bias");
    enc.aifi_out_w = get(aifi + ".self_attn.out_proj.weight");
    enc.aifi_out_b = get(aifi + ".self_attn.out_proj.bias");
    enc.aifi_norm1_w = get(aifi + ".norm1.weight");
    enc.aifi_norm1_b = get(aifi + ".norm1.bias");
    enc.aifi_ffn1_w = get(aifi + ".linear1.weight");
    enc.aifi_ffn1_b = get(aifi + ".linear1.bias");
    enc.aifi_ffn2_w = get(aifi + ".linear2.weight");
    enc.aifi_ffn2_b = get(aifi + ".linear2.bias");
    enc.aifi_norm2_w = get(aifi + ".norm2.weight");
    enc.aifi_norm2_b = get(aifi + ".norm2.bias");
    enc.pos_embed = get("model.encoder.pos_embed2");

    // --- Decoder ---
    auto& dec = ctx->decoder;
    for (int i = 0; i < 3; i++)
        load_conv(dec.input_proj[i], std::string("model.decoder.input_proj.") + std::to_string(i));
    dec.anchors = get("model.decoder.anchors");
    dec.valid_mask = get("model.decoder.valid_mask");
    dec.enc_proj_w = get("model.decoder.enc_output.proj.weight");
    dec.enc_norm_w = get("model.decoder.enc_output.norm.weight");
    dec.enc_norm_b = get("model.decoder.enc_output.norm.bias");
    dec.enc_score_w = get("model.decoder.enc_score_head.weight");
    dec.enc_score_b = get("model.decoder.enc_score_head.bias");
    for (int i = 0; i < 3; i++) {
        auto k = std::string("model.decoder.enc_bbox_head.layers.") + std::to_string(i);
        dec.enc_bbox_w[i] = get(k + ".weight");
        dec.enc_bbox_b[i] = get(k + ".bias");
    }
    for (int i = 0; i < 3; i++) {
        auto k = std::string("model.decoder.query_pos_head.layers.") + std::to_string(i);
        dec.qpos_w[i] = get(k + ".weight");
        dec.qpos_b[i] = get(k + ".bias");
    }
    // Decoder layers
    for (int i = 0; i < 6; i++) {
        auto pfx = std::string("model.decoder.decoder.layers.") + std::to_string(i);
        auto& l = dec.layers[i];
        l.self_qkv_w = get(pfx + ".self_attn.in_proj_weight");
        l.self_qkv_b = get(pfx + ".self_attn.in_proj_bias");
        l.self_out_w = get(pfx + ".self_attn.out_proj.weight");
        l.self_out_b = get(pfx + ".self_attn.out_proj.bias");
        l.norm1_w = get(pfx + ".norm1.weight");
        l.norm1_b = get(pfx + ".norm1.bias");
        l.cross_value_w = get(pfx + ".cross_attn.value_proj.weight");
        l.cross_value_b = get(pfx + ".cross_attn.value_proj.bias");
        // Note: cross_attn.sampling_offsets and .attention_weights have no weight
        // tensor — they're computed from the query via linear projections stored
        // as bias-only (the weights are in the ONNX graph as Gemm nodes)
        // Try both original and shortened names (GGUF 64-char limit)
        auto short_pfx = std::string("m.dec.dec.layers.") + std::to_string(i);
        l.cross_sampling_offsets_b = get(pfx + ".cross_attn.sampling_offsets.bias");
        if (!l.cross_sampling_offsets_b)
            l.cross_sampling_offsets_b = get(short_pfx + ".cross_attn.samp_offs.bias");
        l.cross_attn_weights_b = get(pfx + ".cross_attn.attention_weights.bias");
        if (!l.cross_attn_weights_b)
            l.cross_attn_weights_b = get(short_pfx + ".cross_attn.attn_wts.bias");
        l.cross_out_w = get(pfx + ".cross_attn.output_proj.weight");
        l.cross_out_b = get(pfx + ".cross_attn.output_proj.bias");
        l.norm2_w = get(pfx + ".norm2.weight");
        l.norm2_b = get(pfx + ".norm2.bias");
        l.ffn1_w = get(pfx + ".linear1.weight");
        l.ffn1_b = get(pfx + ".linear1.bias");
        l.ffn2_w = get(pfx + ".linear2.weight");
        l.ffn2_b = get(pfx + ".linear2.bias");
        l.norm3_w = get(pfx + ".norm3.weight");
        l.norm3_b = get(pfx + ".norm3.bias");
        // Per-layer bbox head
        for (int j = 0; j < 3; j++) {
            auto bk = std::string("model.decoder.dec_bbox_head.") + std::to_string(i) +
                      ".layers." + std::to_string(j);
            dec.dec_bbox_w[i][j] = get(bk + ".weight");
            dec.dec_bbox_b[i][j] = get(bk + ".bias");
        }
    }
    dec.dec_score_w = get("model.decoder.dec_score_head.5.weight");
    dec.dec_score_b = get("model.decoder.dec_score_head.5.bias");

    // Verify critical tensors
    int missing = 0;
    if (!bb.stem[0].w) { fprintf(stderr, "  MISS: stem conv1_1\n"); missing++; }
    if (!enc.aifi_qkv_w) { fprintf(stderr, "  MISS: AIFI QKV\n"); missing++; }
    if (!dec.anchors) { fprintf(stderr, "  MISS: decoder anchors\n"); missing++; }
    if (!dec.layers[0].self_qkv_w) { fprintf(stderr, "  MISS: decoder layer 0 self QKV\n"); missing++; }

    fprintf(stderr, "layout_detect: loaded %zu tensors (%d missing)\n",
            ctx->wl.tensors.size(), missing);
    return missing == 0;
}

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

static ggml_tensor* prep_conv(ggml_context* g, ggml_tensor* w, int IC, int KH, int KW) {
    if (!w) return nullptr;
    if (ggml_n_dims(w) == 2) {
        if (ggml_is_quantized(w->type)) w = ggml_cast(g, w, GGML_TYPE_F32);
        int64_t OC = w->ne[1];
        w = ggml_reshape_4d(g, w, KW, KH, IC, OC);
    }
    if (w->type != GGML_TYPE_F16) w = ggml_cast(g, w, GGML_TYPE_F16);
    return w;
}

static ggml_tensor* conv_relu(ggml_context* g, ggml_tensor* x, const conv_w& c,
                               int IC, int KH, int KW, int stride, int pad, bool relu = true) {
    if (!c.w) return x;
    auto* w = prep_conv(g, c.w, IC, KH, KW);
    x = ggml_conv_2d(g, w, x, stride, stride, pad, pad, 1, 1);
    if (c.b) {
        int OC = (int)c.b->ne[0];
        x = ggml_add(g, x, ggml_reshape_3d(g, c.b, 1, 1, OC));
    }
    if (relu) x = ggml_relu(g, x);
    return x;
}

static ggml_tensor* linear(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, b);
    return x;
}

static ggml_tensor* layer_norm(ggml_context* g, ggml_tensor* x,
                                ggml_tensor* w, ggml_tensor* b, float eps = 1e-5f) {
    if (!w) return x;
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, w);
    if (b) x = ggml_add(g, x, b);
    return x;
}

// ---------------------------------------------------------------------------
// ResNet-50 backbone
// ---------------------------------------------------------------------------

// Returns 3 feature maps: C3 (stride 8), C4 (stride 16), C5 (stride 32)
static void backbone_forward(ggml_context* g, const resnet50_backbone& bb,
                              ggml_tensor* input,
                              ggml_tensor** c3, ggml_tensor** c4, ggml_tensor** c5) {
    // Stem: 3 × (3×3 conv, stride 2 for first, stride 1 for rest) + maxpool
    auto* x = conv_relu(g, input, bb.stem[0], 3, 3, 3, 2, 1);
    x = conv_relu(g, x, bb.stem[1], 32, 3, 3, 1, 1);
    x = conv_relu(g, x, bb.stem[2], 32, 3, 3, 1, 1);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    // x is now stride 4, 64 channels

    // Helper: Bottleneck block
    auto bottleneck_fwd = [&](ggml_tensor* inp, const bottleneck& blk,
                               int in_ch, int mid_ch, int out_ch, int stride) {
        auto* identity = inp;
        // branch2a: 1×1, reduce
        auto* out = conv_relu(g, inp, blk.branch2a, in_ch, 1, 1, stride, 0);
        // branch2b: 3×3
        out = conv_relu(g, out, blk.branch2b, mid_ch, 3, 3, 1, 1);
        // branch2c: 1×1, expand, NO relu
        out = conv_relu(g, out, blk.branch2c, mid_ch, 1, 1, 1, 0, false);
        // Shortcut
        if (blk.shortcut.w)
            identity = conv_relu(g, inp, blk.shortcut, in_ch, 1, 1, stride, 0, false);
        // Residual + ReLU
        return ggml_relu(g, ggml_add(g, out, identity));
    };

    // Stage 0: 3 blocks, 64→256, stride 1
    int channels[] = {64, 256, 512, 1024, 2048};
    int mid_channels[] = {64, 128, 256, 512};
    int strides[] = {1, 2, 2, 2};

    for (int s = 0; s < 4; s++) {
        int in_ch = (s == 0) ? 64 : channels[s];
        for (int b = 0; b < (int)bb.stages[s].size(); b++) {
            int blk_in = (b == 0) ? in_ch : channels[s+1];
            int stride = (b == 0) ? strides[s] : 1;
            x = bottleneck_fwd(x, bb.stages[s][b], blk_in, mid_channels[s],
                               channels[s+1], stride);
        }
        // Save outputs for FPN
        if (s == 1) *c3 = x;  // stride 8, 512 ch
        if (s == 2) *c4 = x;  // stride 16, 1024 ch
        if (s == 3) *c5 = x;  // stride 32, 2048 ch
    }
}

// ---------------------------------------------------------------------------
// Forward pass (placeholder — backbone only for now)
// ---------------------------------------------------------------------------

// TODO: Full forward pass implementing:
//   1. Backbone (done above)
//   2. Hybrid encoder: project → FPN → PAN → AIFI
//   3. Transformer decoder: self-attn → deformable cross-attn → FFN × 6
//   4. Detection heads: bbox + class per query
//
// The deformable cross-attention requires CPU-side bilinear grid sampling:
//   - For each of 300 queries × 8 heads × 3 scales × 4 points:
//     sample from the multi-scale feature map at learned offset positions
//   - Weighted sum → cross-attention output

std::vector<region> detect(context* ctx, const float* pixels,
                            int orig_h, int orig_w,
                            float score_threshold) {
    if (!ctx || !pixels) return {};

    // For now, run ONNX Runtime as reference to verify backbone
    // TODO: implement full ggml graph
    fprintf(stderr, "layout_detect: full RT-DETRv2 graph not yet implemented\n");
    fprintf(stderr, "  backbone + encoder + decoder need ~1500 lines of ggml ops\n");
    fprintf(stderr, "  deformable cross-attn needs CPU-side bilinear grid sampling\n");
    return {};
}

std::vector<region> detect_file(context* ctx, const char* path,
                                 float score_threshold) {
    if (!ctx || !path) return {};

    int img_w, img_h, img_c;
    stbi_uc* raw = stbi_load(path, &img_w, &img_h, &img_c, 3);
    if (!raw) {
        fprintf(stderr, "layout_detect: cannot load image: %s\n", path);
        return {};
    }

    // Resize to 640×640, normalize, CHW
    std::vector<float> pixels(3 * ctx->input_h * ctx->input_w, 0.0f);
    float scale_x = (float)ctx->input_w / img_w;
    float scale_y = (float)ctx->input_h / img_h;

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < ctx->input_h; y++) {
            float src_y = y / scale_y;
            int sy0 = std::min((int)src_y, img_h - 1);
            for (int x = 0; x < ctx->input_w; x++) {
                float src_x = x / scale_x;
                int sx0 = std::min((int)src_x, img_w - 1);
                float val = raw[(sy0 * img_w + sx0) * 3 + c] / 255.0f;
                val = (val - ctx->img_mean[c]) / ctx->img_std[c];
                pixels[c * ctx->input_h * ctx->input_w + y * ctx->input_w + x] = val;
            }
        }
    }
    stbi_image_free(raw);

    auto results = detect(ctx, pixels.data(), img_h, img_w, score_threshold);

    // Scale coordinates back to original image
    for (auto& r : results) {
        r.x1 /= scale_x; r.x2 /= scale_x;
        r.y1 /= scale_y; r.y2 /= scale_y;
    }

    return results;
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

} // namespace layout_detect
