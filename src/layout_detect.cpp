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
#include "crispembed_diff.h"

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
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Debug logging gated on LAYOUT_DEBUG env var
static bool layout_debug() {
    static int val = -1;
    if (val < 0) val = (getenv("LAYOUT_DEBUG") != nullptr) ? 1 : 0;
    return val != 0;
}
#define LDBG(...) do { if (layout_debug()) fprintf(stderr, __VA_ARGS__); } while(0)

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

struct csp_block {
    conv_w conv1;             // 1×1, IC=2*C → C (split)
    conv_w conv2;             // 1×1, IC=2*C → C (merge)
    conv_w bottlenecks[3];    // 3×3 convs
};

struct hybrid_encoder {
    conv_w input_proj[3];     // project backbone features to 256d
    conv_w lateral_convs[2];  // for FPN
    conv_w downsample_convs[2]; // for PAN
    // FPN/PAN CSP blocks
    csp_block fpn_blocks[2];
    csp_block pan_blocks[2];
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
    // Deformable cross-attention (value proj + offset/weight projections + output)
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
    ggml_backend_sched_t sched = nullptr;
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
        if (!c.b) c.b = get(prefix + ".weight_bias"); // BN-folded bias
        if (!c.w) {
            // Try without .conv suffix (some layers vary)
            c.w = get(prefix + ".conv.weight");
            c.b = get(prefix + ".conv.bias");
            if (!c.b) c.b = get(prefix + ".conv.weight_bias");
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
    // FPN/PAN CSP blocks
    auto load_csp = [&](csp_block& csp, const std::string& prefix) {
        load_conv(csp.conv1, prefix + ".conv1.conv");
        load_conv(csp.conv2, prefix + ".conv2.conv");
        for (int j = 0; j < 3; j++)
            load_conv(csp.bottlenecks[j], prefix + ".bottlenecks." + std::to_string(j) + ".conv");
    };
    for (int i = 0; i < 2; i++) {
        load_csp(enc.fpn_blocks[i], std::string("model.encoder.fpn_blocks.") + std::to_string(i));
        load_csp(enc.pan_blocks[i], std::string("model.encoder.pan_blocks.") + std::to_string(i));
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
    if (!dec.enc_proj_w) dec.enc_proj_w = get("m.dec.enc_output.proj.weight");
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
        // Sampling offsets and attention weights — both weight and bias
        l.cross_sampling_offsets_w = get(pfx + ".cross_attn.sampling_offsets.weight");
        if (!l.cross_sampling_offsets_w)
            l.cross_sampling_offsets_w = get(short_pfx + ".cross_attn.samp_offs.weight");
        l.cross_sampling_offsets_b = get(pfx + ".cross_attn.sampling_offsets.bias");
        if (!l.cross_sampling_offsets_b)
            l.cross_sampling_offsets_b = get(short_pfx + ".cross_attn.samp_offs.bias");
        l.cross_attn_weights_w = get(pfx + ".cross_attn.attn_weights.weight");
        if (!l.cross_attn_weights_w)
            l.cross_attn_weights_w = get(short_pfx + ".cross_attn.attn_wts.weight");
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
    if (!bb.stem[0].w) { LDBG("  MISS: stem conv1_1\n"); missing++; }
    if (!enc.aifi_qkv_w) { LDBG("  MISS: AIFI QKV\n"); missing++; }
    if (!dec.anchors) { LDBG("  MISS: decoder anchors\n"); missing++; }
    if (!dec.layers[0].self_qkv_w) { LDBG("  MISS: decoder layer 0 self QKV\n"); missing++; }

    // Create backend scheduler
    int sched_max = 4096;
    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, sched_max, false, false);
    if (!ctx->sched) {
        fprintf(stderr, "layout_detect: failed to create scheduler\n");
        return false;
    }

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
        if (w->type != GGML_TYPE_F32) {
            w = ggml_cont(g, ggml_cast(g, w, GGML_TYPE_F32));
        }
        int64_t OC = w->ne[1];
        w = ggml_reshape_4d(g, w, KW, KH, IC, OC);
    }
    // Use F32 weights for ggml_conv_2d_direct (no F16 precision loss)
    if (w->type != GGML_TYPE_F32) {
        w = ggml_cast(g, w, GGML_TYPE_F32);
    }
    return w;
}

static ggml_tensor* conv_relu(ggml_context* g, ggml_tensor* x, const conv_w& c,
                               int IC, int KH, int KW, int stride, int pad, bool relu = true) {
    if (!c.w) return x;
    auto* w = prep_conv(g, c.w, IC, KH, KW);
    x = ggml_conv_2d_direct(g, w, x, stride, stride, pad, pad, 1, 1);
    if (c.b) {
        int OC = (int)c.b->ne[0];
        x = ggml_add(g, x, ggml_reshape_3d(g, c.b, 1, 1, OC));
    }
    if (relu) x = ggml_relu(g, x);
    return x;
}

// Conv + optional bias + SiLU activation
static ggml_tensor* conv_silu(ggml_context* g, ggml_tensor* x, const conv_w& c,
                                int IC, int KH, int KW, int stride, int pad) {
    if (!c.w) return x;
    auto* w = prep_conv(g, c.w, IC, KH, KW);
    x = ggml_conv_2d_direct(g, w, x, stride, stride, pad, pad, 1, 1);
    if (c.b) {
        int OC = (int)c.b->ne[0];
        x = ggml_add(g, x, ggml_reshape_3d(g, c.b, 1, 1, OC));
    }
    // SiLU: x * sigmoid(x)
    x = ggml_mul(g, x, ggml_sigmoid(g, x));
    return x;
}

static ggml_tensor* linear(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    // ggml_mul_mat(a, b): contracts over a->ne[0], result ne[0] = a->ne[1]
    // If w->ne[0] != x->ne[0], the weight needs transposing
    if (w->ne[0] != x->ne[0] && w->ne[1] == x->ne[0]) {
        w = ggml_cont(g, ggml_transpose(g, w));
    }
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
        // branch2a: 1×1, reduce (NO stride — ResNet-D puts stride on branch2b)
        auto* out = conv_relu(g, inp, blk.branch2a, in_ch, 1, 1, 1, 0);
        // branch2b: 3×3 (stride applied HERE in ResNet-D)
        out = conv_relu(g, out, blk.branch2b, mid_ch, 3, 3, stride, 1);
        // branch2c: 1×1, expand, NO relu
        out = conv_relu(g, out, blk.branch2c, mid_ch, 1, 1, 1, 0, false);
        // Shortcut (ResNet-D: avgpool + 1×1 conv for downsampling)
        if (blk.shortcut.w) {
            if (stride > 1) {
                identity = ggml_pool_2d(g, inp, GGML_OP_POOL_AVG, stride, stride,
                                         stride, stride, 0, 0);
            }
            identity = conv_relu(g, identity, blk.shortcut, in_ch, 1, 1, 1, 0, false);
        }
        // Residual + ReLU
        return ggml_relu(g, ggml_add(g, out, identity));
    };

    // Stage 0: 3 blocks, 64→256, stride 1
    int channels[] = {64, 256, 512, 1024, 2048};
    int mid_channels[] = {64, 128, 256, 512};
    int strides[] = {1, 2, 2, 2};

    // Mark stem output for debug
    {
        char name[32];
        snprintf(name, sizeof(name), "bb_stem");
        ggml_set_name(x, name);
        ggml_set_output(x);
    }

    for (int s = 0; s < 4; s++) {
        int in_ch = (s == 0) ? 64 : channels[s];
        for (int b = 0; b < (int)bb.stages[s].size(); b++) {
            int blk_in = (b == 0) ? in_ch : channels[s+1];
            int stride = (b == 0) ? strides[s] : 1;
            x = bottleneck_fwd(x, bb.stages[s][b], blk_in, mid_channels[s],
                               channels[s+1], stride);
            // Mark each block output for diff comparison
            char name[32];
            snprintf(name, sizeof(name), "bb_s%d_b%d", s, b);
            ggml_set_name(x, name);
            ggml_set_output(x);
        }
        // Save outputs for FPN
        if (s == 1) *c3 = x;  // stride 8, 512 ch
        if (s == 2) *c4 = x;  // stride 16, 1024 ch
        if (s == 3) *c5 = x;  // stride 32, 2048 ch
    }
}

// ---------------------------------------------------------------------------
// Hybrid encoder forward
// ---------------------------------------------------------------------------

// Multi-head self-attention for AIFI encoder (single layer, 8 heads, 256d)
static ggml_tensor* aifi_self_attn(ggml_context* g, ggml_tensor* x,
                                    const hybrid_encoder& enc) {
    // x: [256, N] where N = spatial tokens
    auto* residual = x;

    // QKV projection: split in_proj into Q, K, V
    auto* qkv = linear(g, x, enc.aifi_qkv_w, enc.aifi_qkv_b); // [768, N]
    LDBG("  AIFI: qkv=[%lld,%lld]\n", (long long)qkv->ne[0], (long long)qkv->ne[1]);
    int D = 256, N_heads = 8, head_dim = D / N_heads; // 32

    // Split QKV → Q[256,N], K[256,N], V[256,N]
    auto* q = ggml_cont(g, ggml_view_2d(g, qkv, D, qkv->ne[1], qkv->nb[1], 0));
    auto* k = ggml_cont(g, ggml_view_2d(g, qkv, D, qkv->ne[1], qkv->nb[1], D * sizeof(float)));
    auto* v = ggml_cont(g, ggml_view_2d(g, qkv, D, qkv->ne[1], qkv->nb[1], 2 * D * sizeof(float)));

    // Reshape for multi-head: [head_dim, N, N_heads]
    int64_t N = qkv->ne[1];
    q = ggml_reshape_3d(g, q, head_dim, N, N_heads);
    k = ggml_reshape_3d(g, k, head_dim, N, N_heads);
    v = ggml_reshape_3d(g, v, head_dim, N, N_heads);

    // Permute to [head_dim, N, N_heads] → compatible with flash_attn
    // ggml flash_attn_ext expects q[D,N,H], k[D,N,H], v[D,N,H]
    auto* attn = ggml_flash_attn_ext(g, q, k, v, nullptr, 1.0f / sqrtf(head_dim), 0, 0);
    // attn: [head_dim, N, N_heads]

    // Reshape back to [D, N]
    attn = ggml_cont(g, ggml_reshape_2d(g, attn, D, N));

    // Output projection
    attn = linear(g, attn, enc.aifi_out_w, enc.aifi_out_b);

    // Residual + LayerNorm
    x = ggml_add(g, residual, attn);
    x = layer_norm(g, x, enc.aifi_norm1_w, enc.aifi_norm1_b);

    // FFN: linear1 → ReLU → linear2
    residual = x;
    auto* ffn = linear(g, x, enc.aifi_ffn1_w, enc.aifi_ffn1_b);
    ffn = ggml_relu(g, ffn);
    ffn = linear(g, ffn, enc.aifi_ffn2_w, enc.aifi_ffn2_b);

    x = ggml_add(g, residual, ffn);
    x = layer_norm(g, x, enc.aifi_norm2_w, enc.aifi_norm2_b);

    return x;
}

// Encoder: project backbone features → FPN → PAN → AIFI on top scale
// Returns 3 multi-scale features: [256, H*W] for each scale
static void encoder_forward(ggml_context* g, const hybrid_encoder& enc,
                             ggml_tensor* c3, ggml_tensor* c4, ggml_tensor* c5,
                             ggml_tensor** s3, ggml_tensor** s4, ggml_tensor** s5) {
    // Input projection: reduce channels to 256
    int c3_ch = (int)c3->ne[2], c4_ch = (int)c4->ne[2], c5_ch = (int)c5->ne[2];
    LDBG("  encoder input_proj: c3_ch=%d c4_ch=%d c5_ch=%d\n", c3_ch, c4_ch, c5_ch);
    if (enc.input_proj[0].w)
        LDBG("  input_proj[0].w: [%lld,%lld] ndim=%d\n",
                (long long)enc.input_proj[0].w->ne[0], (long long)enc.input_proj[0].w->ne[1],
                ggml_n_dims(enc.input_proj[0].w));
    auto* p3 = conv_relu(g, c3, enc.input_proj[0], c3_ch, 1, 1, 1, 0, false);
    ggml_set_name(p3, "ip3"); ggml_set_output(p3);
    auto* p4 = conv_relu(g, c4, enc.input_proj[1], c4_ch, 1, 1, 1, 0, false);
    ggml_set_name(p4, "ip4"); ggml_set_output(p4);
    auto* p5 = conv_relu(g, c5, enc.input_proj[2], c5_ch, 1, 1, 1, 0, false);
    ggml_set_name(p5, "ip5"); ggml_set_output(p5);

    // AIFI self-attention on S5 (smallest scale, 20×20 = 400 tokens)
    // p5 is [W, H, C] in ggml. We need [D, N] = [C, W*H] for attention.
    // Permute [W,H,C] → [C,W,H] then reshape [C, W*H]
    int s5_w = (int)p5->ne[0], s5_h = (int)p5->ne[1], s5_c = (int)p5->ne[2];
    LDBG("  AIFI: p5=[%lld,%lld,%lld] s5_w=%d s5_h=%d s5_c=%d\n",
            (long long)p5->ne[0], (long long)p5->ne[1], (long long)p5->ne[2],
            s5_w, s5_h, s5_c);
    auto* p5_perm = ggml_cont(g, ggml_permute(g, p5, 1, 2, 0, 3));  // ne: [C, W, H]
    LDBG("  AIFI: p5_perm=[%lld,%lld,%lld,%lld] nelems=%lld\n",
            (long long)p5_perm->ne[0], (long long)p5_perm->ne[1],
            (long long)p5_perm->ne[2], (long long)p5_perm->ne[3],
            (long long)ggml_nelements(p5_perm));
    LDBG("  AIFI: reshape to [%d, %d] = %d\n", s5_c, s5_w*s5_h, s5_c * s5_w * s5_h);
    auto* p5_flat = ggml_reshape_2d(g, p5_perm, s5_c, s5_w * s5_h); // ne: [C, W*H]
    LDBG("  AIFI: p5_flat done\n");
    // Transpose to [256, N] — ggml reshape gives [C, H*W], need [C, N] which is same
    // AIFI: add(pos_embed) → self-attn → residual+norm1 → FFN(GELU_erf) → residual+norm2
    {
        // ONNX applies pos_embed to Q/K only, not V.  Split the in_proj_weight
        // into separate Q_w[256,256], K_w[256,256], V_w[256,256] and project
        // Q,K from (x + pos) and V from x.
        auto* x_pos = p5_flat;
        if (enc.pos_embed) {
            auto* pe = ggml_reshape_2d(g, enc.pos_embed, s5_c, s5_w * s5_h);
            x_pos = ggml_add(g, p5_flat, pe);
        }

        int D_a = 256, N_tok = s5_w * s5_h, heads = 8, hd = D_a / heads;

        // Split in_proj_weight [768, 256] → Q_w, K_w, V_w each [256, 256]
        // Split in_proj_bias [768] → Q_b, K_b, V_b each [256]
        auto* qkv_w = enc.aifi_qkv_w;  // ne[0]=256(in), ne[1]=768(out) in ggml
        auto* qkv_b = enc.aifi_qkv_b;  // ne[0]=768

        auto* q_w = ggml_cont(g, ggml_view_2d(g, qkv_w, D_a, D_a, qkv_w->nb[1], 0));
        auto* k_w = ggml_cont(g, ggml_view_2d(g, qkv_w, D_a, D_a, qkv_w->nb[1], (size_t)D_a * qkv_w->nb[1]));
        auto* v_w = ggml_cont(g, ggml_view_2d(g, qkv_w, D_a, D_a, qkv_w->nb[1], (size_t)2 * D_a * qkv_w->nb[1]));

        auto* q_b = ggml_cont(g, ggml_view_1d(g, qkv_b, D_a, 0));
        auto* k_b = ggml_cont(g, ggml_view_1d(g, qkv_b, D_a, D_a*sizeof(float)));
        auto* v_b = ggml_cont(g, ggml_view_1d(g, qkv_b, D_a, 2*D_a*sizeof(float)));

        // Q, K from x+pos; V from x (no pos_embed)
        auto* q = linear(g, x_pos, q_w, q_b);   // [256, N_tok]
        auto* k = linear(g, x_pos, k_w, k_b);   // [256, N_tok]
        auto* v = linear(g, p5_flat, v_w, v_b);  // [256, N_tok] — no pos!
        ggml_set_name(q, "aifi_q"); ggml_set_output(q);
        ggml_set_name(k, "aifi_k"); ggml_set_output(k);
        ggml_set_name(v, "aifi_v"); ggml_set_output(v);

        // Reshape for multi-head: [D=256, N] → [hd=32, heads=8, N] → permute to [hd, N, heads]
        q = ggml_reshape_3d(g, q, hd, heads, N_tok);
        q = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));  // [hd, N, heads]
        k = ggml_reshape_3d(g, k, hd, heads, N_tok);
        k = ggml_cont(g, ggml_permute(g, k, 0, 2, 1, 3));
        v = ggml_reshape_3d(g, v, hd, heads, N_tok);
        v = ggml_cont(g, ggml_permute(g, v, 0, 2, 1, 3));

        // Manual scaled dot-product attention
        q = ggml_scale(g, q, 1.0f / sqrtf((float)hd));
        auto* scores = ggml_mul_mat(g, k, q);  // [N, N, heads]
        scores = ggml_soft_max(g, scores);
        ggml_set_name(scores, "aifi_softmax"); ggml_set_output(scores);
        auto* v_t = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));  // [N, hd, heads]
        auto* attn_raw = ggml_mul_mat(g, v_t, scores);  // [hd, N, heads]
        ggml_set_name(attn_raw, "aifi_attn_raw"); ggml_set_output(attn_raw);

        // Correct head interleaving: [hd, N, heads] → [hd, heads, N] → [D, N]
        auto* attn = ggml_cont(g, ggml_permute(g, attn_raw, 0, 2, 1, 3));  // [hd, heads, N]
        attn = ggml_reshape_2d(g, attn, D_a, N_tok);  // [256, 400]

        // Output projection
        attn = linear(g, attn, enc.aifi_out_w, enc.aifi_out_b);
        ggml_set_name(attn, "aifi_attn_out"); ggml_set_output(attn);

        // Residual from original input (before pos_embed) + norm1
        auto* aifi_resid = ggml_add(g, p5_flat, attn);
        ggml_set_name(aifi_resid, "aifi_resid"); ggml_set_output(aifi_resid);
        p5_flat = aifi_resid;
        p5_flat = layer_norm(g, p5_flat, enc.aifi_norm1_w, enc.aifi_norm1_b);
        ggml_set_name(p5_flat, "aifi_norm1"); ggml_set_output(p5_flat);

        // FFN: linear1 → GELU → linear2
        auto* residual = p5_flat;
        auto* ffn = linear(g, p5_flat, enc.aifi_ffn1_w, enc.aifi_ffn1_b);
        ffn = ggml_gelu_erf(g, ffn);
        ffn = linear(g, ffn, enc.aifi_ffn2_w, enc.aifi_ffn2_b);
        p5_flat = ggml_add(g, residual, ffn);
        p5_flat = layer_norm(g, p5_flat, enc.aifi_norm2_w, enc.aifi_norm2_b);
    }
    LDBG("  AIFI done: p5_flat=[%lld,%lld]\n", (long long)p5_flat->ne[0], (long long)p5_flat->ne[1]);
    // Reshape back to spatial: [C, W*H] → [C, W, H] → permute to [W, H, C]
    auto* p5_spatial = ggml_reshape_3d(g, p5_flat, s5_c, s5_w, s5_h);
    p5 = ggml_cont(g, ggml_permute(g, p5_spatial, 2, 0, 1, 3)); // [W, H, C]
    LDBG("  AIFI reshape back: p5=[%lld,%lld,%lld]\n",
            (long long)p5->ne[0], (long long)p5->ne[1], (long long)p5->ne[2]);

    // FPN top-down pathway
    LDBG("  FPN: p3=[%lld,%lld,%lld] p4=[%lld,%lld,%lld] p5=[%lld,%lld,%lld]\n",
            (long long)p3->ne[0],(long long)p3->ne[1],(long long)p3->ne[2],
            (long long)p4->ne[0],(long long)p4->ne[1],(long long)p4->ne[2],
            (long long)p5->ne[0],(long long)p5->ne[1],(long long)p5->ne[2]);
    // FPN CSP block helper:
    // cat = concat(upsample/downsample, lateral)  [512ch]
    // branch_a = conv1(cat) → SiLU → bottleneck0 → SiLU → ... → SiLU  [256ch]
    // branch_b = conv2(cat) → SiLU  [256ch]
    // output = Add(branch_a, branch_b)  [256ch]
    static int csp_idx = 0;
    auto run_csp = [&](ggml_tensor* cat_input, const csp_block& blk) -> ggml_tensor* {
        int cat_ch = (int)cat_input->ne[2];
        // Branch A: conv1 → SiLU → bottlenecks → SiLU
        auto* a = conv_silu(g, cat_input, blk.conv1, cat_ch, 1, 1, 1, 0);
        {char n[32]; snprintf(n,32,"csp%d_conv1",csp_idx); ggml_set_name(a,n); ggml_set_output(a);}
        for (int j = 0; j < 3; j++) {
            if (blk.bottlenecks[j].w) {
                a = conv_silu(g, a, blk.bottlenecks[j], 256, 3, 3, 1, 1);
                char n[32]; snprintf(n,32,"csp%d_bn%d",csp_idx,j); ggml_set_name(a,n); ggml_set_output(a);
            }
        }
        // Branch B: conv2 → SiLU
        auto* b = conv_silu(g, cat_input, blk.conv2, cat_ch, 1, 1, 1, 0);
        {char n[32]; snprintf(n,32,"csp%d_conv2",csp_idx); ggml_set_name(b,n); ggml_set_output(b);}
        // Add branches
        auto* out = ggml_add(g, a, b);
        {char n[32]; snprintf(n,32,"csp%d_out",csp_idx); ggml_set_name(out,n); ggml_set_output(out);}
        csp_idx++;
        return out;
    };

    // Mark AIFI output for comparison
    ggml_set_name(p5, "aifi_out"); ggml_set_output(p5);

    // S5 → lateral (SiLU) → upsample → concat with S4 → CSP
    auto* lat5 = conv_silu(g, p5, enc.lateral_convs[0], 256, 1, 1, 1, 0);
    ggml_set_name(lat5, "lat5_silu"); ggml_set_output(lat5);
    auto* up5 = ggml_interpolate(g, lat5, (int)p4->ne[0], (int)p4->ne[1],
                                  (int)lat5->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto* cat4 = ggml_concat(g, up5, p4, 2);  // [W, H, 512]
    p4 = run_csp(cat4, enc.fpn_blocks[0]);
    ggml_set_name(p4, "csp0"); ggml_set_output(p4);

    // S4 → lateral (SiLU) → upsample → concat with S3 → CSP
    auto* lat4 = conv_silu(g, p4, enc.lateral_convs[1], 256, 1, 1, 1, 0);
    auto* up4 = ggml_interpolate(g, lat4, (int)p3->ne[0], (int)p3->ne[1],
                                  (int)lat4->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto* cat3 = ggml_concat(g, up4, p3, 2);  // [W, H, 512]
    p3 = run_csp(cat3, enc.fpn_blocks[1]);

    // PAN bottom-up: S3 → downsample → concat with lat4 → CSP
    // RT-DETRv2 CCFM: PAN uses the lateral-processed features (inner_outs),
    // not the raw FPN outputs.  inner_outs = [fpn_p3, lat4, lat5].
    auto* down3 = conv_silu(g, p3, enc.downsample_convs[0], 256, 3, 3, 2, 1);
    auto* cat4p = ggml_concat(g, down3, lat4, 2);  // [W, H, 512]
    p4 = run_csp(cat4p, enc.pan_blocks[0]);

    // S4 → downsample → concat with lat5 → CSP
    auto* down4 = conv_silu(g, p4, enc.downsample_convs[1], 256, 3, 3, 2, 1);
    auto* cat5p = ggml_concat(g, down4, lat5, 2);  // [W, H, 512]
    p5 = run_csp(cat5p, enc.pan_blocks[1]);

    *s3 = p3; *s4 = p4; *s5 = p5;
}

// ---------------------------------------------------------------------------
// Bilinear grid sampling (CPU-side, for deformable attention)
// ---------------------------------------------------------------------------

// Sample a 2D feature map at fractional position (px, py) using bilinear interpolation.
// feat: [C, H, W] row-major. Returns C-dimensional vector.
static void bilinear_sample(const float* feat, int W, int H, int C,
                             float px, float py, float* out) {
    // Clamp to valid range
    px = std::max(0.0f, std::min(px, (float)(W - 1)));
    py = std::max(0.0f, std::min(py, (float)(H - 1)));

    int x0 = (int)px, y0 = (int)py;
    int x1 = std::min(x0 + 1, W - 1);
    int y1 = std::min(y0 + 1, H - 1);
    float fx = px - x0, fy = py - y0;

    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    for (int c = 0; c < C; c++) {
        out[c] = w00 * feat[c * H * W + y0 * W + x0]
               + w01 * feat[c * H * W + y0 * W + x1]
               + w10 * feat[c * H * W + y1 * W + x0]
               + w11 * feat[c * H * W + y1 * W + x1];
    }
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

std::vector<region> detect(context* ctx, const float* pixels,
                            int orig_h, int orig_w,
                            float score_threshold) {
    if (!ctx || !pixels) return {};

    const int H = ctx->input_h, W = ctx->input_w;  // 640
    const int D = 256, N_heads = 8, N_queries = 300;
    const int N_levels = 3, N_points = 4;

    // --- Phase 1: Backbone + Encoder (ggml graph) ---
    // This produces multi-scale features via ggml graph compute
    int max_nodes = 4096;
    size_t buf_size = ggml_tensor_overhead() * (max_nodes + 500)
                    + ggml_graph_overhead_custom(max_nodes, false)
                    + 256 * 1024 * 1024; // 256 MB extra for tensor descriptors
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params ip = { buf_size, buf.data(), true };
    ggml_context* g = ggml_init(ip);

    // Input: [W, H, 3] in ggml layout
    ggml_tensor* input = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, 3);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    // Backbone
    // Verify stem weight is the folded version
    {
        float wv[3];
        ggml_backend_tensor_get(ctx->backbone.stem[0].w, wv, 0, 3*sizeof(float));
        fprintf(stderr, "  stem[0].w first3: %.6f %.6f %.6f (expected: 0.002018)\n", wv[0], wv[1], wv[2]);
    }
    fprintf(stderr, "layout_detect: building backbone + encoder graph...\n");
    ggml_tensor *c3, *c4, *c5;
    backbone_forward(g, ctx->backbone, input, &c3, &c4, &c5);

    // Encoder
    ggml_tensor *s3, *s4, *s5;
    encoder_forward(g, ctx->encoder, c3, c4, c5, &s3, &s4, &s5);

    // Mark backbone + encoder outputs
    ggml_set_name(c3, "c3"); ggml_set_output(c3);
    ggml_set_name(c4, "c4"); ggml_set_output(c4);
    ggml_set_name(c5, "c5"); ggml_set_output(c5);
    ggml_set_name(s3, "s3"); ggml_set_output(s3);
    ggml_set_name(s4, "s4"); ggml_set_output(s4);
    ggml_set_name(s5, "s5"); ggml_set_output(s5);

    // Build + compute backbone+encoder graph
    ggml_cgraph* gf = ggml_new_graph_custom(g, max_nodes, false);
    // Expand all outputs (backbone debug + encoder outputs)
    ggml_build_forward_expand(gf, s3);
    ggml_build_forward_expand(gf, s4);
    ggml_build_forward_expand(gf, s5);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "layout_detect: backbone graph alloc failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return {};
    }

    // Set input pixels
    ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "input");
    if (inp_t) ggml_backend_tensor_set(inp_t, pixels, 0, 3 * H * W * sizeof(float));

    // Compute
    fprintf(stderr, "layout_detect: computing backbone + encoder...\n");
    ggml_backend_graph_compute(ctx->backend, gf);

    // Per-block backbone comparison
    {
        // Read stem
        auto read_range = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t) { fprintf(stderr, "  %s: NOT FOUND\n", name); return; }
            int n = ggml_nelements(t);
            std::vector<float> data(n);
            ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
            float fmin = 1e9, fmax = -1e9;
            for (auto v : data) { fmin = std::min(fmin, v); fmax = std::max(fmax, v); }
            fprintf(stderr, "  %s: [%lld,%lld,%lld] range=[%.4f, %.4f]\n",
                    name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], fmin, fmax);
        };
        read_range("bb_stem");
        read_range("ip3");
        read_range("ip4");
        read_range("ip5");
        read_range("aifi_out");
        read_range("lat5_silu");
        // CSP block 0 internals
        read_range("aifi_qkv");
        read_range("aifi_softmax");
        read_range("aifi_attn_raw");
        read_range("aifi_attn_out");
        read_range("aifi_norm1");
        read_range("aifi_resid");
        read_range("csp0_conv1");
        read_range("csp0_bn0");
        read_range("csp0_bn1");
        read_range("csp0_bn2");
        read_range("csp0_conv2");
        read_range("csp0_out");
        int block_counts[] = {3, 4, 6, 3};
        for (int s = 0; s < 4; s++) {
            for (int b = 0; b < block_counts[s]; b++) {
                char name[32];
                snprintf(name, sizeof(name), "bb_s%d_b%d", s, b);
                read_range(name);
            }
        }
    }
    // Read encoder outputs to CPU buffers
    // s3: [W3, H3, 256], s4: [W4, H4, 256], s5: [W5, H5, 256]
    struct scale_feat {
        std::vector<float> data; // CHW layout: [C, H, W]
        int W, H, C;
    };
    scale_feat feats[3];
    const char* feat_names[] = {"s3", "s4", "s5"};
    int total_tokens = 0;
    for (int i = 0; i < 3; i++) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, feat_names[i]);
        feats[i].W = (int)t->ne[0];
        feats[i].H = (int)t->ne[1];
        feats[i].C = (int)t->ne[2];
        int n = feats[i].W * feats[i].H * feats[i].C;
        feats[i].data.resize(n);
        ggml_backend_tensor_get(t, feats[i].data.data(), 0, n * sizeof(float));
        total_tokens += feats[i].W * feats[i].H;
        // Print stats for comparison with Python reference
        float fmin = 1e9, fmax = -1e9;
        for (auto v : feats[i].data) { fmin = std::min(fmin, v); fmax = std::max(fmax, v); }
        fprintf(stderr, "  s%d: %dx%d x %d range=[%.4f, %.4f]\n",
                i+3, feats[i].W, feats[i].H, feats[i].C, fmin, fmax);
    }

    ggml_gallocr_free(alloc);
    ggml_free(g);

    // --- Phase 2: Decoder (CPU-side) ---

    // cpu_linear for Gemm(transB=1) convention: y = W @ x + b
    // W stored as [in_d, out_d] in GGUF, accessed as W[i + o * in_d]
    auto cpu_linear = [](const float* x, float* y, int in_d, int out_d, int N,
                          ggml_tensor* w_t, ggml_tensor* b_t) {
        std::vector<float> W(out_d * in_d), b(out_d, 0.0f);
        if (w_t) ggml_backend_tensor_get(w_t, W.data(), 0, out_d * in_d * sizeof(float));
        if (b_t) ggml_backend_tensor_get(b_t, b.data(), 0, out_d * sizeof(float));
        for (int n = 0; n < N; n++) {
            for (int o = 0; o < out_d; o++) {
                float sum = b[o];
                for (int i = 0; i < in_d; i++)
                    sum += W[i + o * in_d] * x[i * N + n];
                y[o * N + n] = sum;
            }
        }
    };

    // cpu_matmul for ONNX MatMul(x, W) convention: y[o] = sum_i x[i] * W[i, o]
    // W stored as [in_d, out_d] in GGUF, accessed as W[i * out_d + o]
    auto cpu_matmul = [](const float* x, float* y, int in_d, int out_d, int N,
                          ggml_tensor* w_t, ggml_tensor* b_t) {
        std::vector<float> W(out_d * in_d), b(out_d, 0.0f);
        if (w_t) ggml_backend_tensor_get(w_t, W.data(), 0, out_d * in_d * sizeof(float));
        if (b_t) ggml_backend_tensor_get(b_t, b.data(), 0, out_d * sizeof(float));
        for (int n = 0; n < N; n++) {
            for (int o = 0; o < out_d; o++) {
                float sum = b[o];
                for (int i = 0; i < in_d; i++)
                    sum += W[i * out_d + o] * x[i * N + n];
                y[o * N + n] = sum;
            }
        }
    };

    auto cpu_layernorm = [](float* x, int D, int N, ggml_tensor* w_t, ggml_tensor* b_t) {
        std::vector<float> w(D), b(D);
        if (w_t) ggml_backend_tensor_get(w_t, w.data(), 0, D * sizeof(float));
        if (b_t) ggml_backend_tensor_get(b_t, b.data(), 0, D * sizeof(float));
        for (int n = 0; n < N; n++) {
            float mean = 0, var = 0;
            for (int d = 0; d < D; d++) mean += x[d * N + n];
            mean /= D;
            for (int d = 0; d < D; d++) {
                float diff = x[d * N + n] - mean;
                var += diff * diff;
            }
            var = 1.0f / sqrtf(var / D + 1e-5f);
            for (int d = 0; d < D; d++) {
                x[d * N + n] = (x[d * N + n] - mean) * var * w[d] + b[d];
            }
        }
    };


    // Helper lambdas need to be defined before use
    // (moved memory construction after lambda definitions below)
    // N_total = 80*80 + 40*40 + 20*20 = 8400
    std::vector<float> memory(D * total_tokens);
    int level_starts[3], level_sizes[3];
    int offset = 0;
    for (int lv = 0; lv < 3; lv++) {
        level_starts[lv] = offset;
        level_sizes[lv] = feats[lv].W * feats[lv].H;
        int N_lv = level_sizes[lv];

        // Convert ggml [W, H, C] to [C, N] column-major
        std::vector<float> feat_col(D * N_lv);
        for (int c = 0; c < D; c++)
            for (int s = 0; s < N_lv; s++)
                feat_col[c * N_lv + s] = feats[lv].data[s + c * N_lv];

        // Apply decoder input_proj (1×1 conv = linear projection)
        if (ctx->decoder.input_proj[lv].w) {
            std::vector<float> proj(D * N_lv);
            cpu_linear(feat_col.data(), proj.data(), D, D, N_lv,
                       ctx->decoder.input_proj[lv].w, ctx->decoder.input_proj[lv].b);
            // Copy projected features to memory
            for (int c = 0; c < D; c++)
                for (int s = 0; s < N_lv; s++)
                    memory[c * total_tokens + offset + s] = proj[c * N_lv + s];
        } else {
            // No projection — use raw features
            for (int c = 0; c < D; c++)
                for (int s = 0; s < N_lv; s++)
                    memory[c * total_tokens + offset + s] = feat_col[c * N_lv + s];
        }
        offset += N_lv;
    }

    // Initialize queries from anchors
    // anchors: [300, 4] — reference points (cx, cy, w, h) in [0, 1]
    std::vector<float> anchors(N_queries * 4);
    if (ctx->decoder.anchors)
        ggml_backend_tensor_get(ctx->decoder.anchors, anchors.data(), 0, N_queries * 4 * sizeof(float));

    // Query embeddings: project encoder output to get initial queries
    // Helper: CPU-side matrix multiply y = W @ x + b
    // W: [out, in], x: [in, N], y: [out, N]

    // CPU LayerNorm

    // --- Query initialization: select top-K from encoder output ---
    // 1. Project memory through enc_output: mask → linear + norm
    // Apply valid_mask to memory before projection (zeros out invalid positions)
    std::vector<float> masked_memory(D * total_tokens);
    {
        std::vector<float> mask(total_tokens, 1.0f);
        if (ctx->decoder.valid_mask) {
            // valid_mask is [8400] or [8400, 1] in GGUF
            int mask_n = (int)ggml_nelements(ctx->decoder.valid_mask);
            if (mask_n >= total_tokens)
                ggml_backend_tensor_get(ctx->decoder.valid_mask, mask.data(), 0,
                                         total_tokens * sizeof(float));
        }
        for (int d = 0; d < D; d++)
            for (int n = 0; n < total_tokens; n++)
                masked_memory[d * total_tokens + n] = memory[d * total_tokens + n] * mask[n];
    }

    std::vector<float> enc_proj(D * total_tokens);
    cpu_matmul(masked_memory.data(), enc_proj.data(), D, D, total_tokens,
               ctx->decoder.enc_proj_w, nullptr);
    // Add enc_output.proj.bias if exists
    {
        ggml_tensor* proj_b = ctx->wl.tensors.count("model.decoder.enc_output.proj.bias")
            ? ctx->wl.tensors.at("model.decoder.enc_output.proj.bias") : nullptr;
        if (proj_b) {
            std::vector<float> bias(D);
            ggml_backend_tensor_get(proj_b, bias.data(), 0, D * sizeof(float));
            for (int n = 0; n < total_tokens; n++)
                for (int d = 0; d < D; d++)
                    enc_proj[d * total_tokens + n] += bias[d];
        }
    }
    cpu_layernorm(enc_proj.data(), D, total_tokens,
                  ctx->decoder.enc_norm_w, ctx->decoder.enc_norm_b);

    // Debug: dump raw memory for parity comparison
    if (getenv("LAYOUT_DEBUG")) {
        // Dump memory as [N_total, D] row-major
        std::vector<float> mem_row(D * total_tokens);
        for (int n = 0; n < total_tokens; n++)
            for (int d = 0; d < D; d++)
                mem_row[n * D + d] = memory[d * total_tokens + n];
        FILE* mfp = fopen("/tmp/cpp_raw_memory.bin", "wb");
        if (mfp) { fwrite(mem_row.data(), sizeof(float), D * total_tokens, mfp); fclose(mfp); }
        // Print per-level ranges
        for (int lv = 0; lv < 3; lv++) {
            float lmin=1e9, lmax=-1e9;
            for (int i = level_starts[lv]; i < level_starts[lv] + level_sizes[lv]; i++)
                for (int d = 0; d < D; d++) {
                    float v = memory[d * total_tokens + i];
                    lmin = std::min(lmin, v); lmax = std::max(lmax, v);
                }
            fprintf(stderr, "  memory s%d (%d tokens): range=[%.4f, %.4f]\n",
                    lv+3, level_sizes[lv], lmin, lmax);
        }
        fprintf(stderr, "  memory[s3,0,:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                mem_row[0], mem_row[1], mem_row[2], mem_row[3],
                mem_row[4], mem_row[5], mem_row[6], mem_row[7]);
        fprintf(stderr, "  memory[s4,0,:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                mem_row[6400*D+0], mem_row[6400*D+1], mem_row[6400*D+2], mem_row[6400*D+3],
                mem_row[6400*D+4], mem_row[6400*D+5], mem_row[6400*D+6], mem_row[6400*D+7]);
    }

    // Debug: dump arrays for parity comparison
    if (getenv("LAYOUT_DEBUG")) {
        // Dump memory [D, total_tokens] col-major → [total_tokens, D] row-major
        auto dump_col_major = [&](const std::vector<float>& data, int dim, int N, const char* path) {
            std::vector<float> row(dim * N);
            for (int n = 0; n < N; n++)
                for (int d = 0; d < dim; d++)
                    row[n * dim + d] = data[d * N + n];
            FILE* fp = fopen(path, "wb");
            if (fp) { fwrite(row.data(), sizeof(float), dim * N, fp); fclose(fp); }
        };
        dump_col_major(memory, D, total_tokens, "/tmp/cpp_memory.bin");
        dump_col_major(enc_proj, D, total_tokens, "/tmp/cpp_enc_proj.bin");
        fprintf(stderr, "  enc_proj: range=[%.4f, %.4f]\n",
                *std::min_element(enc_proj.begin(), enc_proj.end()),
                *std::max_element(enc_proj.begin(), enc_proj.end()));
        fprintf(stderr, "  enc_proj[0,:8]: ");
        for (int d = 0; d < 8; d++) fprintf(stderr, "%.6f ", enc_proj[d * total_tokens + 0]);
        fprintf(stderr, "\n");
    }

    // 2. Score each token
    std::vector<float> enc_scores(ctx->num_classes * total_tokens);
    cpu_matmul(enc_proj.data(), enc_scores.data(), D, ctx->num_classes, total_tokens,
               ctx->decoder.enc_score_w, ctx->decoder.enc_score_b);

    // 3. Find top-K tokens by max class score
    std::vector<std::pair<float, int>> token_scores(total_tokens);
    for (int n = 0; n < total_tokens; n++) {
        float max_s = -1e9f;
        for (int c = 0; c < ctx->num_classes; c++)
            max_s = std::max(max_s, enc_scores[c * total_tokens + n]);
        token_scores[n] = {max_s, n};
    }
    std::partial_sort(token_scores.begin(), token_scores.begin() + N_queries,
                      token_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // 4. Initialize queries from selected tokens
    std::vector<float> queries(D * N_queries, 0.0f);
    for (int q = 0; q < N_queries; q++) {
        int token_idx = token_scores[q].second;
        for (int d = 0; d < D; d++)
            queries[d * N_queries + q] = enc_proj[d * total_tokens + token_idx];
    }

    // 5. Initialize reference points via enc_bbox_head MLP on selected tokens
    // Python: ref_points_unact = anchors[selected] + bbox_head(enc_proj[selected])
    //         ref_points = sigmoid(ref_points_unact)
    std::vector<float> ref_points(N_queries * 4);
    {
        // Load full anchors [8400, 4] to pick the right ones for selected tokens
        std::vector<float> all_anchors(total_tokens * 4, 0.0f);
        if (ctx->decoder.anchors)
            ggml_backend_tensor_get(ctx->decoder.anchors, all_anchors.data(), 0,
                                     total_tokens * 4 * sizeof(float));

        std::vector<float> tmp(D * N_queries);
        std::copy(queries.begin(), queries.end(), tmp.begin());
        for (int j = 0; j < 3; j++) {
            int out_d = (j < 2) ? D : 4;
            std::vector<float> out(out_d * N_queries);
            cpu_matmul(tmp.data(), out.data(), (j == 0 ? D : D), out_d, N_queries,
                       ctx->decoder.enc_bbox_w[j], ctx->decoder.enc_bbox_b[j]);
            if (j < 2) {
                for (auto& v : out) v = std::max(0.0f, v); // ReLU
                tmp = out;
            } else {
                // Add anchors for selected tokens, then sigmoid
                // out is [4, N_queries], anchors are [total_tokens, 4] row-major
                for (int q = 0; q < N_queries; q++) {
                    int token_idx = token_scores[q].second;
                    for (int d = 0; d < 4; d++) {
                        float bbox_val = out[d * N_queries + q];
                        float anchor_val = all_anchors[token_idx * 4 + d];
                        float unact = bbox_val + anchor_val;
                        ref_points[q * 4 + d] = 1.0f / (1.0f + expf(-unact));
                    }
                }
            }
        }
    }

    // Helper: compute query position encoding from reference points via query_pos_head MLP
    // 2-layer MLP: ref_points [4] → linear0 [512] → ReLU → linear1 [256] → pos_enc
    auto compute_pos_enc = [&](std::vector<float>& pos_enc_out) {
        std::vector<float> rp_col(4 * N_queries);
        for (int q = 0; q < N_queries; q++)
            for (int d = 0; d < 4; d++)
                rp_col[d * N_queries + q] = ref_points[q * 4 + d];
        int hidden = 2 * D; // 512
        std::vector<float> tmp(hidden * N_queries);
        // Layer 0: anonymous val_1947 [4,512] — MatMul convention (no Transpose in ONNX)
        cpu_matmul(rp_col.data(), tmp.data(), 4, hidden, N_queries,
                   ctx->decoder.qpos_w[0], ctx->decoder.qpos_b[0]);
        for (auto& v : tmp) v = std::max(0.0f, v);
        // Layer 1: named weight [256,512] — Gemm convention (ONNX has Transpose before MatMul)
        cpu_linear(tmp.data(), pos_enc_out.data(), hidden, D, N_queries,
                   ctx->decoder.qpos_w[1], ctx->decoder.qpos_b[1]);
    };

    fprintf(stderr, "layout_detect: query init done (top-K score: %.3f..%.3f)\n",
            token_scores[0].first, token_scores[N_queries-1].first);
    fprintf(stderr, "layout_detect: running decoder (6 layers, %d queries)...\n", N_queries);

    // Load diff reference if available
    crispembed_diff::Ref dec_ref;
    bool has_dec_ref = dec_ref.load("/tmp/dec0-diff-ref.gguf");

    auto diff_report = [&](const char* label, const char* ref_name,
                           const float* data, size_t n_elem) {
        if (!has_dec_ref) return;
        auto r = dec_ref.compare(ref_name, data, n_elem);
        if (r.found)
            fprintf(stderr, "    DIFF %s: cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n",
                    label, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
    };

    // Compare initial encoder projection and memory
    diff_report("enc_proj", "layer_norm_2", enc_proj.data(), D * total_tokens);
    diff_report("memory", "concat_4", memory.data(), D * total_tokens);

    // Load per-layer ref_points reference
    crispembed_diff::Ref rp_ref;
    bool has_rp_ref = rp_ref.load("/tmp/refpoints-per-layer.gguf");

    // Per-layer ref_points names: sigmoid, sigmoid_1, ..., sigmoid_6
    // Per-layer pos_enc names: linear_12, linear_27, linear_42, linear_57, linear_72, linear_87
    static const char* pos_enc_names[] = {
        "linear_12", "linear_27", "linear_42", "linear_57", "linear_72", "linear_87"
    };

    for (int li = 0; li < 6; li++) {
        const auto& layer = ctx->decoder.layers[li];

        // Compare ref_points before this layer
        if (has_rp_ref) {
            char rp_name[32];
            snprintf(rp_name, 32, "sigmoid%s", li == 0 ? "" : (std::string("_") + std::to_string(li)).c_str());
            auto rp_r = rp_ref.compare(rp_name, ref_points.data(), N_queries * 4);
            if (rp_r.found)
                fprintf(stderr, "    DIFF dec%d_refpts(%s): cos_min=%.6f max_abs=%.2e %s\n",
                        li, rp_name, rp_r.cos_min, rp_r.max_abs, rp_r.is_pass() ? "PASS" : "FAIL");
        }

        // Recompute pos_enc each layer from current ref_points
        std::vector<float> pos_enc(D * N_queries);
        compute_pos_enc(pos_enc);

        // Compare pos_enc for this layer
        if (has_rp_ref) {
            auto pe_r = rp_ref.compare(pos_enc_names[li], pos_enc.data(), D * N_queries);
            if (pe_r.found)
                fprintf(stderr, "    DIFF dec%d_posenc: cos_min=%.6f max_abs=%.2e %s\n",
                        li, pe_r.cos_min, pe_r.max_abs, pe_r.is_pass() ? "PASS" : "FAIL");
        }

        if (li == 0) {
            diff_report("queries_init", "gather_2", queries.data(), D * N_queries);
            diff_report("pos_enc", "linear_12", pos_enc.data(), D * N_queries);
            // Also dump ref_points for comparison
            if (has_dec_ref) {
                auto [rp_ref, rp_n] = dec_ref.get_f32("sigmoid");
                if (rp_ref && rp_n >= (size_t)(N_queries * 4)) {
                    float cos = 0, na = 0, nb = 0;
                    for (int i = 0; i < N_queries * 4; i++) {
                        cos += ref_points[i] * rp_ref[i];
                        na += ref_points[i] * ref_points[i];
                        nb += rp_ref[i] * rp_ref[i];
                    }
                    cos /= (sqrtf(na) * sqrtf(nb) + 1e-10f);
                    fprintf(stderr, "    DIFF ref_points: cos=%.6f cpp[0]=%.6f py[0]=%.6f\n",
                            cos, ref_points[0], rp_ref[0]);
                }
            }
        }

        // --- Self-attention ---
        // RT-DETRv2: Q = Q_w @ (query + pos), K = K_w @ (query + pos), V = V_w @ query
        std::vector<float> residual = queries; // [D, N_queries]
        int hd = D / N_heads;

        // (query + pos) for Q and K
        std::vector<float> qp_sa(D * N_queries);
        for (int i = 0; i < D * N_queries; i++) qp_sa[i] = queries[i] + pos_enc[i];

        // Split in_proj_weight [3*D, D] into Q_w, K_w, V_w
        // cpu_linear uses W stored as [ne0=in_d, ne1=out_d], indexing W[i + o*in_d]
        // Q part: rows 0..D-1, K part: rows D..2D-1, V part: rows 2D..3D-1
        std::vector<float> q_proj(D * N_queries), k_proj(D * N_queries), v_proj(D * N_queries);

        // Extract weight/bias sub-tensors for Q, K, V
        // self_qkv_w has shape [ne0=D, ne1=3*D]
        // self_qkv_b has shape [3*D]
        auto w_data = [](ggml_tensor* t) -> const float* {
            // This is a weight tensor, data is in backend
            static std::vector<float> buf;
            size_t n = ggml_nelements(t);
            buf.resize(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
            return buf.data();
        };

        // For cpu_linear: y[o] = sum_i W[i + o*in_d] * x[i]
        // W is read from the ggml tensor with indexing data[i + o*D]
        // For the Q portion (out rows 0..D-1): same as reading from the full weight
        // For K portion (out rows D..2D-1): offset by D*D elements
        // For V portion (out rows 2D..3D-1): offset by 2*D*D elements
        {
            // Read the full weight and bias
            int total_out = 3 * D;
            std::vector<float> full_w(D * total_out);
            ggml_backend_tensor_get(layer.self_qkv_w, full_w.data(), 0, full_w.size() * sizeof(float));
            std::vector<float> full_b(total_out);
            ggml_backend_tensor_get(layer.self_qkv_b, full_b.data(), 0, full_b.size() * sizeof(float));

            // ONNX: Split(in_proj_weight) → Transpose(perm=[1,0]) → MatMul
            // The Transpose cancels out the MatMul convention, so net effect is
            // Gemm convention: Q[o] = sum_i W_split[o, i] * input[i]
            // W_split[o, i] = data[o * D + i] (Gemm convention)
            for (int n = 0; n < N_queries; n++) {
                for (int o = 0; o < D; o++) {
                    float sum = full_b[o];
                    for (int i = 0; i < D; i++)
                        sum += full_w[o * D + i] * qp_sa[i * N_queries + n];
                    q_proj[o * N_queries + n] = sum;
                }
            }
            for (int n = 0; n < N_queries; n++) {
                for (int o = 0; o < D; o++) {
                    float sum = full_b[D + o];
                    for (int i = 0; i < D; i++)
                        sum += full_w[(D + o) * D + i] * qp_sa[i * N_queries + n];
                    k_proj[o * N_queries + n] = sum;
                }
            }
            // V = W_v @ query (no pos!)
            for (int n = 0; n < N_queries; n++) {
                for (int o = 0; o < D; o++) {
                    float sum = full_b[2 * D + o];
                    for (int i = 0; i < D; i++)
                        sum += full_w[(2 * D + o) * D + i] * queries[i * N_queries + n];
                    v_proj[o * N_queries + n] = sum;
                }
            }
        }

        // Multi-head self-attention
        std::vector<float> attn_out(D * N_queries, 0.0f);
        for (int h = 0; h < N_heads; h++) {
            std::vector<float> scores(N_queries * N_queries, 0.0f);
            for (int qi = 0; qi < N_queries; qi++) {
                for (int ki = 0; ki < N_queries; ki++) {
                    float dot = 0;
                    for (int d = 0; d < hd; d++) {
                        int idx = (h * hd + d) * N_queries;
                        dot += q_proj[idx + qi] * k_proj[idx + ki];
                    }
                    scores[qi * N_queries + ki] = dot / sqrtf((float)hd);
                }
            }
            for (int qi = 0; qi < N_queries; qi++) {
                float max_s = -1e9f;
                for (int ki = 0; ki < N_queries; ki++)
                    max_s = std::max(max_s, scores[qi * N_queries + ki]);
                float sum = 0;
                for (int ki = 0; ki < N_queries; ki++) {
                    scores[qi * N_queries + ki] = expf(scores[qi * N_queries + ki] - max_s);
                    sum += scores[qi * N_queries + ki];
                }
                for (int ki = 0; ki < N_queries; ki++)
                    scores[qi * N_queries + ki] /= sum;
            }
            for (int qi = 0; qi < N_queries; qi++) {
                for (int d = 0; d < hd; d++) {
                    float sum = 0;
                    for (int ki = 0; ki < N_queries; ki++)
                        sum += scores[qi * N_queries + ki] * v_proj[(h * hd + d) * N_queries + ki];
                    attn_out[(h * hd + d) * N_queries + qi] = sum;
                }
            }
        }
        // Output projection
        std::vector<float> sa_out(D * N_queries);
        cpu_linear(attn_out.data(), sa_out.data(), D, D, N_queries,
                   layer.self_out_w, layer.self_out_b);

        // Residual + norm
        for (int i = 0; i < D * N_queries; i++) queries[i] = residual[i] + sa_out[i];
        if (li == 0) {
            diff_report("Q", "linear_13", q_proj.data(), D * N_queries);
            diff_report("K", "linear_14", k_proj.data(), D * N_queries);
            diff_report("V", "linear_15", v_proj.data(), D * N_queries);
            diff_report("sa_out_proj", "linear_16", sa_out.data(), D * N_queries);
            diff_report("sa_resid", "add_2545", queries.data(), D * N_queries);
        }
        cpu_layernorm(queries.data(), D, N_queries, layer.norm1_w, layer.norm1_b);
        {
            char ref_name[32]; snprintf(ref_name, 32, "layer_norm_%d", 3 + li*3);
            char label[32]; snprintf(label, 32, "dec%d_norm1", li);
            diff_report(label, ref_name, queries.data(), D * N_queries);
        }
        { float mn=1e9,mx=-1e9; for(auto v:queries){mn=std::min(mn,v);mx=std::max(mx,v);} fprintf(stderr,"  dec%d_norm1: [%.4f,%.4f]\n",li,mn,mx); }

        // --- Deformable cross-attention ---
        residual = queries;

        // Value projection: [D, N_total] → [D, N_total]
        std::vector<float> values(D * total_tokens);
        cpu_matmul(memory.data(), values.data(), D, D, total_tokens,
                   layer.cross_value_w, layer.cross_value_b);
        if (li == 0 && getenv("LAYOUT_DEBUG")) {
            float vmin=1e9,vmax=-1e9;
            for (auto v:values){vmin=std::min(vmin,v);vmax=std::max(vmax,v);}
            fprintf(stderr, "  dec0 values: [%.4f, %.4f] (Python: [-66.7, 76.0])\n", vmin, vmax);
            // Dump values for comparison
            std::vector<float> val_row(D * total_tokens);
            for (int n = 0; n < total_tokens; n++)
                for (int d = 0; d < D; d++)
                    val_row[n * D + d] = values[d * total_tokens + n];
            FILE* fp = fopen("/tmp/cpp_values.bin", "wb");
            if (fp) { fwrite(val_row.data(), sizeof(float), D * total_tokens, fp); fclose(fp); }
            diff_report("values", "linear_17", values.data(), D * total_tokens);
        }

        // query + position encoding for cross-attention projections
        std::vector<float> qp(D * N_queries);
        for (int i = 0; i < D * N_queries; i++) qp[i] = queries[i] + pos_enc[i];

        // Sampling offsets: (query+pos) → linear → [192, N_queries]
        int n_offsets = N_heads * N_levels * N_points * 2; // 192
        std::vector<float> offsets(n_offsets * N_queries);
        cpu_matmul(qp.data(), offsets.data(), D, n_offsets, N_queries,
                   layer.cross_sampling_offsets_w, layer.cross_sampling_offsets_b);

        // Attention weights: (query+pos) → linear → [96, N_queries]
        int n_weights = N_heads * N_levels * N_points; // 96
        std::vector<float> attn_weights(n_weights * N_queries);
        cpu_matmul(qp.data(), attn_weights.data(), D, n_weights, N_queries,
                   layer.cross_attn_weights_w, layer.cross_attn_weights_b);

        // Softmax attention weights per head per query (over levels * points)
        for (int q = 0; q < N_queries; q++) {
            for (int h = 0; h < N_heads; h++) {
                float max_w = -1e9f;
                for (int lp = 0; lp < N_levels * N_points; lp++) {
                    int idx = (h * N_levels * N_points + lp) * N_queries + q;
                    max_w = std::max(max_w, attn_weights[idx]);
                }
                float sum = 0;
                for (int lp = 0; lp < N_levels * N_points; lp++) {
                    int idx = (h * N_levels * N_points + lp) * N_queries + q;
                    attn_weights[idx] = expf(attn_weights[idx] - max_w);
                    sum += attn_weights[idx];
                }
                for (int lp = 0; lp < N_levels * N_points; lp++) {
                    int idx = (h * N_levels * N_points + lp) * N_queries + q;
                    attn_weights[idx] /= sum;
                }
            }
        }

        // Sample features at offset positions and compute weighted sum
        int value_hd = D / N_heads; // 32
        std::vector<float> cross_out(D * N_queries, 0.0f);

        for (int q = 0; q < N_queries; q++) {
            float ref_cx = ref_points[q * 4 + 0]; // cx in [0, 1]
            float ref_cy = ref_points[q * 4 + 1]; // cy in [0, 1]
            float ref_w  = ref_points[q * 4 + 2]; // w in [0, 1]
            float ref_h  = ref_points[q * 4 + 3]; // h in [0, 1]

            for (int h = 0; h < N_heads; h++) {
                for (int lv = 0; lv < N_levels; lv++) {
                    int fW = feats[lv].W, fH = feats[lv].H;
                    for (int pt = 0; pt < N_points; pt++) {
                        int off_idx = (h * N_levels * N_points + lv * N_points + pt);
                        float dx = offsets[(off_idx * 2 + 0) * N_queries + q];
                        float dy = offsets[(off_idx * 2 + 1) * N_queries + q];

                        // RT-DETRv2 offset processing:
                        // loc = ref_xy + offset / n_points * ref_wh * 0.5
                        // grid = loc * 2.0 - 1.0
                        // pixel = (grid + 1) * size / 2 - 0.5 = loc * size - 0.5
                        // (align_corners=False, padding_mode=zeros)
                        float loc_x = ref_cx + dx * 0.25f * ref_w * 0.5f;
                        float loc_y = ref_cy + dy * 0.25f * ref_h * 0.5f;

                        // align_corners=False: pixel = loc * size - 0.5
                        float sx = loc_x * fW - 0.5f;
                        float sy = loc_y * fH - 0.5f;

                        // Bilinear with zero padding (padding_mode='zeros')
                        int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
                        int x1 = x0 + 1, y1 = y0 + 1;
                        float fx = sx - x0, fy = sy - y0;

                        float w00 = (1-fx)*(1-fy), w01 = fx*(1-fy);
                        float w10 = (1-fx)*fy, w11 = fx*fy;

                        // Zero out weights for out-of-bounds pixels
                        bool v00 = (x0 >= 0 && x0 < fW && y0 >= 0 && y0 < fH);
                        bool v01 = (x1 >= 0 && x1 < fW && y0 >= 0 && y0 < fH);
                        bool v10 = (x0 >= 0 && x0 < fW && y1 >= 0 && y1 < fH);
                        bool v11 = (x1 >= 0 && x1 < fW && y1 >= 0 && y1 < fH);

                        float attn_w = attn_weights[off_idx * N_queries + q];

                        for (int d = 0; d < value_hd; d++) {
                            int vd = h * value_hd + d;
                            int base = vd * total_tokens + level_starts[lv];
                            float v = 0;
                            if (v00) v += w00 * values[base + y0 * fW + x0];
                            if (v01) v += w01 * values[base + y0 * fW + x1];
                            if (v10) v += w10 * values[base + y1 * fW + x0];
                            if (v11) v += w11 * values[base + y1 * fW + x1];
                            cross_out[vd * N_queries + q] += attn_w * v;
                        }
                    }
                }
            }
        }

        // Cross-attention output projection
        std::vector<float> ca_out(D * N_queries);
        cpu_matmul(cross_out.data(), ca_out.data(), D, D, N_queries,
                   layer.cross_out_w, layer.cross_out_b);

        if (li == 0) {
            diff_report("offsets", "linear_18", offsets.data(), n_offsets * N_queries);
            diff_report("attn_wts", "linear_19", attn_weights.data(), n_weights * N_queries);
            diff_report("ca_out_proj", "linear_20", ca_out.data(), D * N_queries);
        }

        // Residual + norm
        for (int i = 0; i < D * N_queries; i++) queries[i] = residual[i] + ca_out[i];
        if (li == 0) diff_report("ca_resid", "add_2801", queries.data(), D * N_queries);
        cpu_layernorm(queries.data(), D, N_queries, layer.norm2_w, layer.norm2_b);
        {
            char ref_name[32]; snprintf(ref_name, 32, "layer_norm_%d", 4 + li*3);
            char label[32]; snprintf(label, 32, "dec%d_norm2", li);
            diff_report(label, ref_name, queries.data(), D * N_queries);
        }

        { float mn=1e9,mx=-1e9; for(auto v:queries){mn=std::min(mn,v);mx=std::max(mx,v);} fprintf(stderr,"  dec%d_norm2: [%.4f,%.4f]\n",li,mn,mx); }
        // --- FFN ---
        residual = queries;
        std::vector<float> ffn1(1024 * N_queries);
        cpu_matmul(queries.data(), ffn1.data(), D, 1024, N_queries,
                   layer.ffn1_w, layer.ffn1_b);
        // ReLU
        for (auto& v : ffn1) v = std::max(0.0f, v);
        std::vector<float> ffn2(D * N_queries);
        cpu_matmul(ffn1.data(), ffn2.data(), 1024, D, N_queries,
                   layer.ffn2_w, layer.ffn2_b);

        // Residual + norm
        for (int i = 0; i < D * N_queries; i++) queries[i] = residual[i] + ffn2[i];
        cpu_layernorm(queries.data(), D, N_queries, layer.norm3_w, layer.norm3_b);
        {
            char ref_name[32]; snprintf(ref_name, 32, "layer_norm_%d", 5 + li*3);
            char label[32]; snprintf(label, 32, "dec%d_norm3", li);
            diff_report(label, ref_name, queries.data(), D * N_queries);
        }

        { float mn=1e9,mx=-1e9; for(auto v:queries){mn=std::min(mn,v);mx=std::max(mx,v);} fprintf(stderr,"  dec%d_norm3: [%.4f,%.4f]\n",li,mn,mx); }
        // Update reference points via bbox head (iterative refinement)
        std::vector<float> bbox_delta(4 * N_queries);
        std::vector<float> tmp(D * N_queries);
        // MLP: 256 → 256 → 256 → 4
        std::copy(queries.begin(), queries.end(), tmp.begin());
        for (int j = 0; j < 3; j++) {
            std::vector<float> out((j < 2 ? D : 4) * N_queries);
            cpu_matmul(tmp.data(), out.data(), (j == 0 ? D : D), (j < 2 ? D : 4), N_queries,
                       ctx->decoder.dec_bbox_w[li][j], ctx->decoder.dec_bbox_b[li][j]);
            if (j < 2) {
                for (auto& v : out) v = std::max(0.0f, v); // ReLU
                tmp = out;
            } else {
                bbox_delta = out;
            }
        }

        // Debug: print bbox_delta and ref_points before refinement
        if (li <= 1 && getenv("LAYOUT_DEBUG")) {
            float dmin=1e9,dmax=-1e9;
            for (auto v : bbox_delta) { dmin=std::min(dmin,v); dmax=std::max(dmax,v); }
            fprintf(stderr, "  dec%d bbox_delta: [%.4f, %.4f]\n", li, dmin, dmax);
            fprintf(stderr, "  dec%d bbox_delta[0,:4]: %.6f %.6f %.6f %.6f\n", li,
                    bbox_delta[0*N_queries+0], bbox_delta[1*N_queries+0],
                    bbox_delta[2*N_queries+0], bbox_delta[3*N_queries+0]);
            fprintf(stderr, "  dec%d ref_points[0] before: %.6f %.6f %.6f %.6f\n", li,
                    ref_points[0], ref_points[1], ref_points[2], ref_points[3]);
        }

        // Refine reference points: sigmoid(inverse_sigmoid(ref) + delta)
        for (int q = 0; q < N_queries; q++) {
            for (int d = 0; d < 4; d++) {
                float r = ref_points[q * 4 + d];
                // inverse sigmoid: log(x1/x2) where x1=clamp(r,eps), x2=clamp(1-r,eps)
                // RT-DETRv2 uses eps=1e-5
                float x1 = std::max(r, 1e-5f);
                float x2 = std::max(1.0f - r, 1e-5f);
                float inv_sig = logf(x1 / x2);
                float refined = 1.0f / (1.0f + expf(-(inv_sig + bbox_delta[d * N_queries + q])));
                ref_points[q * 4 + d] = refined;
            }
        }

        if (li <= 1 && getenv("LAYOUT_DEBUG")) {
            fprintf(stderr, "  dec%d ref_points[0] after:  %.6f %.6f %.6f %.6f\n", li,
                    ref_points[0], ref_points[1], ref_points[2], ref_points[3]);
        }

        LDBG("  decoder layer %d done\n", li);
    }

    // --- Phase 3: Detection heads ---
    // Class scores from final decoder output
    std::vector<float> class_scores(ctx->num_classes * N_queries);
    cpu_matmul(queries.data(), class_scores.data(), D, ctx->num_classes, N_queries,
               ctx->decoder.dec_score_w, ctx->decoder.dec_score_b);
    // Sigmoid
    for (auto& v : class_scores) v = 1.0f / (1.0f + expf(-v));

    // Compare per-query max scores with Python reference
    {
        // Python scores are [300] — max class score per query (after sigmoid)
        std::vector<float> max_per_q(N_queries);
        for (int q = 0; q < N_queries; q++) {
            float mx = 0;
            for (int c = 0; c < ctx->num_classes; c++)
                mx = std::max(mx, class_scores[c * N_queries + q]);
            max_per_q[q] = mx;
        }
        diff_report("final_scores", "scores", max_per_q.data(), N_queries);
    }

    // Collect results
    std::vector<region> results;
    for (int q = 0; q < N_queries; q++) {
        // Find best class
        int best_cls = 0;
        float best_score = 0;
        for (int c = 0; c < ctx->num_classes; c++) {
            float s = class_scores[c * N_queries + q];
            if (s > best_score) { best_score = s; best_cls = c; }
        }

        if (best_score < score_threshold) continue;

        // Decode bbox: ref_points are (cx, cy, w, h) in [0, 1]
        float cx = ref_points[q * 4 + 0];
        float cy = ref_points[q * 4 + 1];
        float bw = ref_points[q * 4 + 2];
        float bh = ref_points[q * 4 + 3];

        // Convert to (x1, y1, x2, y2) in pixel coordinates
        region r;
        r.x1 = (cx - bw / 2) * orig_w;
        r.y1 = (cy - bh / 2) * orig_h;
        r.x2 = (cx + bw / 2) * orig_w;
        r.y2 = (cy + bh / 2) * orig_h;
        r.score = best_score;
        r.label = (label_id)best_cls;
        r.label_name = LABEL_NAMES[best_cls];

        results.push_back(r);
    }

    // Sort by score descending
    std::sort(results.begin(), results.end(),
              [](const region& a, const region& b) { return a.score > b.score; });

    // Debug: print top scores
    if (results.empty()) {
        float max_score = 0;
        for (auto& v : class_scores) max_score = std::max(max_score, v);
        fprintf(stderr, "layout_detect: max class score after sigmoid: %.6f\n", max_score);
    }
    fprintf(stderr, "layout_detect: %zu detections (score > %.2f)\n",
            results.size(), score_threshold);
    return results;
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
                // Model expects [0, 1] input — normalization is baked into the model weights
                float val = raw[(sy0 * img_w + sx0) * 3 + c] / 255.0f;
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
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

} // namespace layout_detect
