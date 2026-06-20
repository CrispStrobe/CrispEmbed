// ocr_detect.cpp — DBNet text detection via ggml.
//
// Architecture: ResNet-18 backbone + FPNC neck + DBHead (prob branch only).
// All BatchNorm is pre-folded into Conv by the converter, so runtime is
// just Conv → ReLU → repeat, plus ConvTranspose2d in the head.
//
// Weight tensor naming convention (from convert-dbnet-to-gguf.py):
//   det.backbone.stem.conv.{weight,bias}
//   det.backbone.stage{S}.block{B}.conv{1,2}.{weight,bias}
//   det.backbone.stage{S}.block{B}.downsample.{weight,bias}
//   det.neck.lateral{i}.weight
//   det.neck.smooth{i}.{weight,bias}
//   det.head.conv1.{weight,bias}
//   det.head.deconv1.{weight,bias}
//   det.head.deconv2.{weight,bias}

#include "ocr_detect.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

// stb_image declarations (implementation lives in image_preprocess.cpp)
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
    void stbi_image_free(void *retval_from_stbi_load);
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ocr_detect {

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------

struct conv_layer {
    ggml_tensor *w = nullptr;  // [KW, KH, IC, OC] in ggml layout
    ggml_tensor *b = nullptr;  // [OC]
};

struct basic_block {
    conv_layer conv1;  // 3×3
    conv_layer conv2;  // 3×3
    conv_layer downsample;  // 1×1 (only if stride > 1 or channel change)
};

struct context {
    // Backbone: ResNet-18
    conv_layer stem;  // 7×7, stride 2
    basic_block stages[4][2];  // 4 stages × 2 blocks

    // Neck: FPNC
    conv_layer lateral[4];  // 1×1 convs
    conv_layer smooth[4];   // 3×3 convs
    conv_layer neck_output; // optional 3×3 reduce conv

    // Head: probability branch
    conv_layer head_conv1;   // 3×3 (256→64)
    conv_layer head_deconv1; // ConvTranspose2d (64→64, k=2, s=2)
    conv_layer head_deconv2; // ConvTranspose2d (64→1, k=2, s=2)

    // Preprocessing constants
    float img_mean[3] = {123.675f, 116.28f, 103.53f};
    float img_std[3]  = {58.395f, 57.12f, 57.375f};
    int pad_divisor = 32;

    // Post-processing defaults
    float prob_thresh = 0.3f;
    float box_thresh  = 0.5f;
    float unclip_ratio = 1.5f;
    int min_area = 10;

    // Backend
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;

    // Last prob map (for debugging)
    std::vector<float> last_prob_map;
    int last_prob_h = 0, last_prob_w = 0;

    bool bench = false;
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool load(context** out, const char* path, int n_threads) {
    auto* ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;

    // Read metadata
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g) { fprintf(stderr, "ocr_detect: cannot open %s\n", path); return false; }

    auto f32_val = [&](const char* k, float d) -> float {
        int64_t i = gguf_find_key(g, k);
        return i >= 0 ? gguf_get_val_f32(g, i) : d;
    };
    auto u32_val = [&](const char* k, int d) -> int {
        int64_t i = gguf_find_key(g, k);
        return i >= 0 ? (int)gguf_get_val_u32(g, i) : d;
    };

    ctx->pad_divisor = u32_val("dbnet.pad_divisor", 32);
    ctx->prob_thresh = f32_val("dbnet.prob_threshold", 0.3f);
    ctx->box_thresh  = f32_val("dbnet.box_threshold", 0.5f);
    ctx->unclip_ratio = f32_val("dbnet.unclip_ratio", 1.5f);
    ctx->min_area = u32_val("dbnet.min_area", 10);

    // Read image mean/std arrays if present
    auto read_f32_array = [&](const char* k, float* dst, int n) {
        int64_t i = gguf_find_key(g, k);
        if (i < 0) return;
        int arr_n = (int)gguf_get_arr_n(g, i);
        const float* data = (const float*)gguf_get_arr_data(g, i);
        for (int j = 0; j < std::min(arr_n, n); j++) {
            dst[j] = data[j];
        }
    };
    read_f32_array("dbnet.image_mean", ctx->img_mean, 3);
    read_f32_array("dbnet.image_std", ctx->img_std, 3);

    core_gguf::free_metadata(g);

    fprintf(stderr, "ocr_detect: loading %s\n", path);
    fprintf(stderr, "  prob_thresh=%.2f box_thresh=%.2f unclip=%.2f\n",
            ctx->prob_thresh, ctx->box_thresh, ctx->unclip_ratio);

    // Load weights — prefer GPU backend when available
    bool force_cpu = (getenv("OCR_DETECT_FORCE_CPU") && atoi(getenv("OCR_DETECT_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    if (!core_gguf::load_weights(path, ctx->backend, "det", ctx->wl)) {
        fprintf(stderr, "ocr_detect: failed to load weights\n");
        return false;
    }

    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = ctx->wl.tensors.find(name);
        return it != ctx->wl.tensors.end() ? it->second : nullptr;
    };

    auto load_conv = [&](conv_layer& cl, const std::string& prefix) {
        // Tensor names in GGUF include "det." prefix from converter
        cl.w = get("det." + prefix + ".weight");
        cl.b = get("det." + prefix + ".bias");
    };

    // Stem
    load_conv(ctx->stem, "backbone.stem.conv");

    // Stages
    for (int s = 0; s < 4; s++) {
        for (int b = 0; b < 2; b++) {
            auto pfx = "backbone.stage" + std::to_string(s) + ".block" + std::to_string(b);
            load_conv(ctx->stages[s][b].conv1, pfx + ".conv1");
            load_conv(ctx->stages[s][b].conv2, pfx + ".conv2");
            load_conv(ctx->stages[s][b].downsample, pfx + ".downsample");
        }
    }

    // Neck
    for (int i = 0; i < 4; i++) {
        load_conv(ctx->lateral[i], "neck.lateral" + std::to_string(i));
        load_conv(ctx->smooth[i], "neck.smooth" + std::to_string(i));
    }
    load_conv(ctx->neck_output, "neck.output");

    // Head
    load_conv(ctx->head_conv1, "head.conv1");
    load_conv(ctx->head_deconv1, "head.deconv1");
    load_conv(ctx->head_deconv2, "head.deconv2");

    // Verify critical weights
    if (!ctx->stem.w) { fprintf(stderr, "ocr_detect: missing stem conv\n"); return false; }
    if (!ctx->head_deconv2.w) { fprintf(stderr, "ocr_detect: missing head deconv2\n"); return false; }

    fprintf(stderr, "ocr_detect: loaded %zu tensors\n", ctx->wl.tensors.size());
    ctx->bench = (std::getenv("CRISPEMBED_OCR_DETECT_BENCH") != nullptr);
    return true;
}

// ---------------------------------------------------------------------------
// Graph building helpers
// ---------------------------------------------------------------------------

// Prepare a conv weight for ggml_conv_2d: handle 2D-flattened quantized
// weights and cast to F16 (required by ggml_conv_2d).
static ggml_tensor* prep_conv_weight(ggml_context* g, ggml_tensor* w,
                                      int IC, int KH, int KW) {
    if (!w) return nullptr;
    if (ggml_n_dims(w) == 2) {
        // Flattened: [IC*KH*KW, OC] — dequant + reshape
        if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
            w = ggml_cont(g, ggml_cast(g, w, GGML_TYPE_F32));
        }
        int64_t OC = w->ne[1];
        w = ggml_reshape_4d(g, w, KW, KH, IC, OC);
    }
    if (w->type != GGML_TYPE_F16) {
        w = ggml_cast(g, w, GGML_TYPE_F16);
    }
    return w;
}

// Conv2D + optional bias + ReLU
static ggml_tensor* conv_bias_relu(ggml_context* g, ggml_tensor* x,
                                    const conv_layer& cl,
                                    int IC, int KH, int KW,
                                    int stride, int pad,
                                    bool relu = true) {
    ggml_tensor* w = prep_conv_weight(g, cl.w, IC, KH, KW);
    x = ggml_conv_2d(g, w, x, stride, stride, pad, pad, 1, 1);

    if (cl.b) {
        int OC = (int)cl.b->ne[0];
        ggml_tensor* bias = ggml_reshape_3d(g, cl.b, 1, 1, OC);
        x = ggml_add(g, x, bias);
    }

    if (relu) {
        x = ggml_relu(g, x);
    }

    return x;
}

// ResNet BasicBlock: conv1(3×3) + relu + conv2(3×3) + shortcut + relu
static ggml_tensor* basic_block_fwd(ggml_context* g, ggml_tensor* x,
                                     const basic_block& blk,
                                     int in_ch, int out_ch, int stride) {
    ggml_tensor* identity = x;

    // conv1: 3×3, stride, pad=1
    ggml_tensor* out = conv_bias_relu(g, x, blk.conv1, in_ch, 3, 3,
                                       stride, 1, /*relu=*/true);

    // conv2: 3×3, stride=1, pad=1, NO relu (applied after residual)
    out = conv_bias_relu(g, out, blk.conv2, out_ch, 3, 3,
                          1, 1, /*relu=*/false);

    // Downsample shortcut if present
    if (blk.downsample.w) {
        identity = conv_bias_relu(g, x, blk.downsample, in_ch, 1, 1,
                                   stride, 0, /*relu=*/false);
    }

    // Residual + ReLU
    out = ggml_add(g, out, identity);
    out = ggml_relu(g, out);
    return out;
}

// Bilinear upsample 2D feature map to target size.
// ggml has ggml_upscale which does nearest-neighbor. For bilinear, we'd
// need ggml_interpolate or manual implementation. For now, use upscale
// (nearest) which is acceptable for detection quality.
static ggml_tensor* upsample_to(ggml_context* g, ggml_tensor* x,
                                 int target_w, int target_h) {
    // x is [W, H, C] in ggml layout
    int cur_w = (int)x->ne[0];
    int cur_h = (int)x->ne[1];
    if (cur_w == target_w && cur_h == target_h) return x;

    // Use ggml_interpolate (bilinear to match PyTorch's F.interpolate)
    int C = (int)x->ne[2];
    return ggml_interpolate(g, x, target_w, target_h, C, 1,
                            GGML_SCALE_MODE_BILINEAR);
}

// Prepare a ConvTranspose2d weight for ggml_conv_transpose_2d_p0.
// PyTorch layout: (IC, OC, KH, KW) → flattened (IC, OC*KH*KW)
// ggml expects: [KW, KH, OC, IC] (ne[3]=IC, ne[2]=OC)
static ggml_tensor* prep_deconv_weight(ggml_context* g, ggml_tensor* w,
                                        int OC, int KH, int KW) {
    if (!w) return nullptr;
    if (ggml_n_dims(w) == 2) {
        // Flattened: ne[0]=OC*KH*KW, ne[1]=IC
        if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
            w = ggml_cont(g, ggml_cast(g, w, GGML_TYPE_F32));
        }
        int64_t IC = w->ne[1];
        w = ggml_reshape_4d(g, w, KW, KH, OC, IC);
    }
    if (w->type != GGML_TYPE_F16) {
        w = ggml_cast(g, w, GGML_TYPE_F16);
    }
    return w;
}

// ConvTranspose2d: implemented as ggml_conv_transpose_2d_p0
// OC = output channels of the deconv (= ne[2] of the kernel in ggml)
static ggml_tensor* conv_transpose_bias_relu(ggml_context* g, ggml_tensor* x,
                                              const conv_layer& cl,
                                              int OC, int KH, int KW,
                                              int stride, bool relu) {
    ggml_tensor* w = prep_deconv_weight(g, cl.w, OC, KH, KW);
    // ggml_conv_transpose_2d_p0(kernel, input, stride)
    // kernel: [KW, KH, OC, IC], input: [W, H, IC, 1]
    x = ggml_conv_transpose_2d_p0(g, w, x, stride);

    if (cl.b) {
        int OC = (int)cl.b->ne[0];
        ggml_tensor* bias = ggml_reshape_3d(g, cl.b, 1, 1, OC);
        x = ggml_add(g, x, bias);
    }

    if (relu) {
        x = ggml_relu(g, x);
    }

    return x;
}

// Sigmoid: 1 / (1 + exp(-x))
static ggml_tensor* sigmoid_op(ggml_context* g, ggml_tensor* x) {
    return ggml_sigmoid(g, x);
}

// ---------------------------------------------------------------------------
// Forward pass: ResNet-18 + FPNC + DBHead
// ---------------------------------------------------------------------------

// Run the full detection graph and return the probability map.
// pixels: [3, H, W] CHW float32, already preprocessed.
// Returns prob_map as [H, W] in [0, 1].
static std::vector<float> forward(context* ctx, const float* pixels, int H, int W) {
    // Estimate graph size: ~200 nodes for ResNet-18 + FPNC + head
    int max_nodes = 512;
    size_t buf_size = ggml_tensor_overhead() * (max_nodes + 100)
                    + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context* g = ggml_init(p);

    // Input: [W, H, 3] in ggml layout (column-major: ne[0]=W, ne[1]=H, ne[2]=3)
    ggml_tensor* x = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, 3);
    ggml_set_name(x, "input");
    ggml_set_input(x);

    // --- Backbone: ResNet-18 ---

    // Stem: conv1(7×7, s2, p3) + relu + maxpool(3, s2, p1)
    x = conv_bias_relu(g, x, ctx->stem, 3, 7, 7, 2, 3, true);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);

    // Stage outputs for FPN
    ggml_tensor* stage_out[4];
    int stage_ch[4] = {64, 128, 256, 512};

    // Stage 0: 2 blocks, 64ch, no stride change
    x = basic_block_fwd(g, x, ctx->stages[0][0], 64, 64, 1);
    x = basic_block_fwd(g, x, ctx->stages[0][1], 64, 64, 1);
    stage_out[0] = x;

    // Stage 1: 2 blocks, 128ch, first block stride 2
    x = basic_block_fwd(g, x, ctx->stages[1][0], 64, 128, 2);
    x = basic_block_fwd(g, x, ctx->stages[1][1], 128, 128, 1);
    stage_out[1] = x;

    // Stage 2: 2 blocks, 256ch, first block stride 2
    x = basic_block_fwd(g, x, ctx->stages[2][0], 128, 256, 2);
    x = basic_block_fwd(g, x, ctx->stages[2][1], 256, 256, 1);
    stage_out[2] = x;

    // Stage 3: 2 blocks, 512ch, first block stride 2
    x = basic_block_fwd(g, x, ctx->stages[3][0], 256, 512, 2);
    x = basic_block_fwd(g, x, ctx->stages[3][1], 512, 512, 1);
    stage_out[3] = x;

    // --- Neck: FPNC ---

    // Lateral 1×1 convs (no bias, no relu in FPNC default)
    ggml_tensor* lat[4];
    for (int i = 0; i < 4; i++) {
        ggml_tensor* w = prep_conv_weight(g, ctx->lateral[i].w,
                                           stage_ch[i], 1, 1);
        lat[i] = ggml_conv_2d(g, w, stage_out[i], 1, 1, 0, 0, 1, 1);
        // Add bias if present
        if (ctx->lateral[i].b) {
            int OC = (int)ctx->lateral[i].b->ne[0];
            ggml_tensor* bias = ggml_reshape_3d(g, ctx->lateral[i].b, 1, 1, OC);
            lat[i] = ggml_add(g, lat[i], bias);
        }
    }

    // Top-down: add upsampled higher level to lower level
    for (int i = 3; i > 0; i--) {
        int target_w = (int)lat[i-1]->ne[0];
        int target_h = (int)lat[i-1]->ne[1];
        ggml_tensor* up = upsample_to(g, lat[i], target_w, target_h);
        lat[i-1] = ggml_add(g, lat[i-1], up);
    }

    // Smooth 3×3 convs (256→64, no relu in FPNC default)
    ggml_tensor* smoothed[4];
    for (int i = 0; i < 4; i++) {
        ggml_tensor* w = prep_conv_weight(g, ctx->smooth[i].w, 256, 3, 3);
        smoothed[i] = ggml_conv_2d(g, w, lat[i], 1, 1, 1, 1, 1, 1);
        if (ctx->smooth[i].b) {
            int OC = (int)ctx->smooth[i].b->ne[0];
            ggml_tensor* bias = ggml_reshape_3d(g, ctx->smooth[i].b, 1, 1, OC);
            smoothed[i] = ggml_add(g, smoothed[i], bias);
        }
    }

    // Upsample all to stage 0 resolution and concatenate
    int target_w = (int)smoothed[0]->ne[0];
    int target_h = (int)smoothed[0]->ne[1];
    for (int i = 1; i < 4; i++) {
        smoothed[i] = upsample_to(g, smoothed[i], target_w, target_h);
    }
    // Concat along channel dim: 4 × 64 = 256
    ggml_tensor* fused = ggml_concat(g, smoothed[0], smoothed[1], 2);
    fused = ggml_concat(g, fused, smoothed[2], 2);
    fused = ggml_concat(g, fused, smoothed[3], 2);

    // Optional output conv (if present in model)
    if (ctx->neck_output.w) {
        int fused_ch = (int)fused->ne[2];
        fused = conv_bias_relu(g, fused, ctx->neck_output, fused_ch, 3, 3,
                                1, 1, true);
    }

    // --- Head: probability branch ---

    // conv1: 3×3 (256→64) + relu (BN folded)
    int fused_ch = (int)fused->ne[2];
    x = conv_bias_relu(g, fused, ctx->head_conv1, fused_ch, 3, 3,
                        1, 1, true);

    // deconv1: ConvTranspose2d (64→64, k=2, s=2) + relu (BN folded)
    // OC=64 (output channels of the deconv)
    x = conv_transpose_bias_relu(g, x, ctx->head_deconv1, 64, 2, 2, 2, true);

    // deconv2: ConvTranspose2d (64→1, k=2, s=2) — no relu
    // OC=1 (output channels of the deconv)
    x = conv_transpose_bias_relu(g, x, ctx->head_deconv2, 1, 2, 2, 2, false);

    // Sigmoid → probability map
    x = sigmoid_op(g, x);

    ggml_set_name(x, "prob_map");
    ggml_set_output(x);

    // Build and compute graph
    ggml_cgraph* gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "ocr_detect: graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return {};
    }

    // Set input
    ggml_tensor* inp = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(inp, pixels, 0, 3 * H * W * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf);

    // Read probability map
    ggml_tensor* prob = ggml_graph_get_tensor(gf, "prob_map");
    int prob_w = (int)prob->ne[0];
    int prob_h = (int)prob->ne[1];
    // prob shape: [W, H, 1] in ggml — read as flat [W*H]
    int prob_total = prob_w * prob_h;
    std::vector<float> prob_map(prob_total);
    ggml_backend_tensor_get(prob, prob_map.data(), 0,
                            prob_total * sizeof(float));

    // Store for debugging
    ctx->last_prob_map = prob_map;
    ctx->last_prob_h = prob_h;
    ctx->last_prob_w = prob_w;

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return prob_map;
}

// ---------------------------------------------------------------------------
// Post-processing: prob map → bounding boxes
// ---------------------------------------------------------------------------

// Contour tracing: extract the boundary pixels of a connected component
// using Moore neighborhood tracing (8-connected boundary following).
static std::vector<std::pair<int,int>> trace_contour(
    const uint8_t * binary, int w, int h,
    const int * labels, int label, int start_x, int start_y)
{
    std::vector<std::pair<int,int>> contour;
    // Moore neighborhood: 8 directions, starting from right
    const int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    int cx = start_x, cy = start_y;
    int dir = 0; // start looking right

    auto is_fg = [&](int x, int y) -> bool {
        if (x < 0 || x >= w || y < 0 || y >= h) return false;
        return labels[x + y * w] == label;
    };

    // Find start: leftmost pixel in topmost row of component
    // (already provided as start_x, start_y)
    int max_steps = w * h * 2; // safety limit
    do {
        contour.push_back({cx, cy});
        // Look for next boundary pixel
        bool found = false;
        for (int i = 0; i < 8; i++) {
            int d = (dir + 6 + i) % 8; // start from dir-2 (backtrack)
            int nx = cx + dx[d], ny = cy + dy[d];
            if (is_fg(nx, ny)) {
                cx = nx; cy = ny;
                dir = d;
                found = true;
                break;
            }
        }
        if (!found) break;
        if (--max_steps <= 0) break;
    } while (cx != start_x || cy != start_y || contour.size() < 3);

    return contour;
}

// Compute minimum-area bounding rectangle for a set of points.
// Uses the rotating calipers algorithm on the convex hull.
// Returns center (cx,cy), size (w,h), angle in degrees.
struct min_rect {
    float cx, cy, w, h, angle;
};

static min_rect min_area_rect(const std::vector<std::pair<int,int>> & pts) {
    min_rect r = {0, 0, 0, 0, 0};
    if (pts.size() < 3) {
        if (!pts.empty()) {
            int mnx = pts[0].first, mxx = mnx, mny = pts[0].second, mxy = mny;
            for (auto & p : pts) {
                if (p.first < mnx) mnx = p.first;
                if (p.first > mxx) mxx = p.first;
                if (p.second < mny) mny = p.second;
                if (p.second > mxy) mxy = p.second;
            }
            r.cx = (mnx + mxx) / 2.0f;
            r.cy = (mny + mxy) / 2.0f;
            r.w = (float)(mxx - mnx + 1);
            r.h = (float)(mxy - mny + 1);
        }
        return r;
    }

    // Convex hull (Andrew's monotone chain)
    auto pts_sorted = pts;
    std::sort(pts_sorted.begin(), pts_sorted.end());
    std::vector<std::pair<int,int>> hull;
    // Lower hull
    for (auto & p : pts_sorted) {
        while (hull.size() >= 2) {
            auto & a = hull[hull.size()-2];
            auto & b = hull[hull.size()-1];
            long cross = (long)(b.first-a.first)*(p.second-a.second)
                       - (long)(b.second-a.second)*(p.first-a.first);
            if (cross <= 0) hull.pop_back(); else break;
        }
        hull.push_back(p);
    }
    // Upper hull
    int lower_size = (int)hull.size();
    for (int i = (int)pts_sorted.size()-2; i >= 0; i--) {
        auto & p = pts_sorted[i];
        while ((int)hull.size() > lower_size) {
            auto & a = hull[hull.size()-2];
            auto & b = hull[hull.size()-1];
            long cross = (long)(b.first-a.first)*(p.second-a.second)
                       - (long)(b.second-a.second)*(p.first-a.first);
            if (cross <= 0) hull.pop_back(); else break;
        }
        hull.push_back(p);
    }
    hull.pop_back(); // remove duplicate last point

    if (hull.size() < 3) {
        // Degenerate — use axis-aligned bbox
        int mnx = hull[0].first, mxx = mnx, mny = hull[0].second, mxy = mny;
        for (auto & p : hull) {
            if (p.first < mnx) mnx = p.first;
            if (p.first > mxx) mxx = p.first;
            if (p.second < mny) mny = p.second;
            if (p.second > mxy) mxy = p.second;
        }
        r.cx = (mnx + mxx) / 2.0f; r.cy = (mny + mxy) / 2.0f;
        r.w = (float)(mxx - mnx + 1); r.h = (float)(mxy - mny + 1);
        return r;
    }

    // Rotating calipers: try each hull edge as base, find min-area rectangle
    float min_area = 1e30f;
    int n = (int)hull.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float ex = (float)(hull[j].first - hull[i].first);
        float ey = (float)(hull[j].second - hull[i].second);
        float len = sqrtf(ex*ex + ey*ey);
        if (len < 1e-6f) continue;
        ex /= len; ey /= len;
        // Project all hull points onto this edge direction and its perpendicular
        float min_proj = 1e30f, max_proj = -1e30f;
        float min_perp = 1e30f, max_perp = -1e30f;
        for (auto & p : hull) {
            float dx = (float)(p.first - hull[i].first);
            float dy = (float)(p.second - hull[i].second);
            float proj = dx * ex + dy * ey;
            float perp = -dx * ey + dy * ex;
            if (proj < min_proj) min_proj = proj;
            if (proj > max_proj) max_proj = proj;
            if (perp < min_perp) min_perp = perp;
            if (perp > max_perp) max_perp = perp;
        }
        float area = (max_proj - min_proj) * (max_perp - min_perp);
        if (area < min_area) {
            min_area = area;
            r.w = max_proj - min_proj;
            r.h = max_perp - min_perp;
            // Center in original coords
            float mid_proj = (min_proj + max_proj) / 2;
            float mid_perp = (min_perp + max_perp) / 2;
            r.cx = hull[i].first + mid_proj * ex - mid_perp * ey;
            r.cy = hull[i].second + mid_proj * ey + mid_perp * ex;
            r.angle = atan2f(ey, ex) * 180.0f / 3.14159265f;
        }
    }
    // Normalize: ensure w >= h (swap if needed, adjust angle)
    if (r.w < r.h) {
        std::swap(r.w, r.h);
        r.angle += 90.0f;
    }
    if (r.angle > 180.0f) r.angle -= 360.0f;
    if (r.angle < -180.0f) r.angle += 360.0f;
    return r;
}

// Score a box by averaging prob_map values inside the polygon.
// More accurate than scoring the entire bounding box (which includes
// background pixels between text and bbox edges).
static float score_polygon(const float * prob_map, int map_w, int map_h,
                            const std::vector<std::pair<int,int>> & contour,
                            int min_x, int min_y, int max_x, int max_y) {
    if (contour.empty()) return 0;
    float sum = 0;
    int count = 0;
    // Simple approach: for each pixel in the bbox, check if it's inside
    // the polygon using ray-casting
    for (int y = min_y; y <= max_y && y < map_h; y++) {
        for (int x = min_x; x <= max_x && x < map_w; x++) {
            // Ray casting: count crossings to the right
            int crossings = 0;
            int n = (int)contour.size();
            for (int i = 0, j = n-1; i < n; j = i++) {
                int yi = contour[i].second, yj = contour[j].second;
                int xi = contour[i].first, xj = contour[j].first;
                if ((yi <= y && yj > y) || (yj <= y && yi > y)) {
                    float t = (float)(y - yi) / (yj - yi);
                    if (x < xi + t * (xj - xi)) crossings++;
                }
            }
            if (crossings & 1) {
                sum += prob_map[x + y * map_w];
                count++;
            }
        }
    }
    return count > 0 ? sum / count : 0;
}

// Connected-component labeling + contour-based bounding box extraction.
static std::vector<text_box> extract_boxes(const float* prob_map,
                                            int map_w, int map_h,
                                            float prob_threshold,
                                            float box_threshold,
                                            float unclip_ratio,
                                            int min_area,
                                            float scale_x, float scale_y) {
    std::vector<text_box> results;

    // Binarize
    std::vector<uint8_t> binary(map_w * map_h, 0);
    for (int i = 0; i < map_w * map_h; i++) {
        // prob_map is [W, H] in ggml layout (column-major: prob_map[x + y*W])
        if (prob_map[i] > prob_threshold) binary[i] = 1;
    }

    // Flood-fill connected components
    std::vector<int> labels(map_w * map_h, 0);
    int next_label = 1;
    std::vector<int> stack;

    for (int y = 0; y < map_h; y++) {
        for (int x = 0; x < map_w; x++) {
            int idx = x + y * map_w;
            if (binary[idx] && labels[idx] == 0) {
                // BFS flood fill
                int label = next_label++;
                stack.clear();
                stack.push_back(idx);
                labels[idx] = label;

                float sum_prob = 0;
                int count = 0;
                int min_x = x, max_x = x, min_y = y, max_y = y;

                while (!stack.empty()) {
                    int cur = stack.back();
                    stack.pop_back();
                    int cx = cur % map_w;
                    int cy = cur / map_w;

                    sum_prob += prob_map[cur];
                    count++;
                    if (cx < min_x) min_x = cx;
                    if (cx > max_x) max_x = cx;
                    if (cy < min_y) min_y = cy;
                    if (cy > max_y) max_y = cy;

                    // 4-connected neighbors
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};
                    for (int d = 0; d < 4; d++) {
                        int nx = cx + dx[d], ny = cy + dy[d];
                        if (nx >= 0 && nx < map_w && ny >= 0 && ny < map_h) {
                            int ni = nx + ny * map_w;
                            if (binary[ni] && labels[ni] == 0) {
                                labels[ni] = label;
                                stack.push_back(ni);
                            }
                        }
                    }
                }

                // Filter by area
                if (count < min_area) continue;

                // Filter by mean score
                float mean_score = sum_prob / count;
                if (mean_score < box_threshold) continue;

                // Trace contour of this component
                auto contour = trace_contour(binary.data(), map_w, map_h,
                                              labels.data(), label, min_x, min_y);

                // Score against probability map (polygon interior only)
                float poly_score = score_polygon(prob_map, map_w, map_h,
                                                  contour, min_x, min_y, max_x, max_y);
                if (poly_score < box_threshold) continue;

                // Compute minimum-area rotated rectangle
                min_rect mr = min_area_rect(contour);

                // Apply unclip expansion
                float bw = mr.w, bh = mr.h;
                float perimeter = 2 * (bw + bh);
                float area = (float)count;
                float offset = area * unclip_ratio / perimeter;
                mr.w += 2 * offset;
                mr.h += 2 * offset;

                // Clamp to map bounds
                float half_w = mr.w / 2, half_h = mr.h / 2;
                float ang_rad = mr.angle * 3.14159265f / 180.0f;
                float cos_a = cosf(ang_rad), sin_a = sinf(ang_rad);

                text_box tb;
                // Compute axis-aligned bounding box of the rotated rectangle
                float abs_cw = fabsf(cos_a * half_w) + fabsf(sin_a * half_h);
                float abs_ch = fabsf(sin_a * half_w) + fabsf(cos_a * half_h);
                tb.x = std::max(0.0f, (mr.cx - abs_cw)) * scale_x;
                tb.y = std::max(0.0f, (mr.cy - abs_ch)) * scale_y;
                tb.w = std::min(2 * abs_cw, (float)map_w - (mr.cx - abs_cw)) * scale_x;
                tb.h = std::min(2 * abs_ch, (float)map_h - (mr.cy - abs_ch)) * scale_y;
                tb.score = poly_score;
                tb.angle = mr.angle;

                // Quad corners of the rotated rectangle
                float corners[4][2] = {
                    {-half_w, -half_h}, { half_w, -half_h},
                    { half_w,  half_h}, {-half_w,  half_h}
                };
                for (int c = 0; c < 4; c++) {
                    tb.qx[c] = (mr.cx + corners[c][0]*cos_a - corners[c][1]*sin_a) * scale_x;
                    tb.qy[c] = (mr.cy + corners[c][0]*sin_a + corners[c][1]*cos_a) * scale_y;
                }

                results.push_back(tb);
            }
        }
    }

    // Sort by y then x (reading order)
    std::sort(results.begin(), results.end(), [](const text_box& a, const text_box& b) {
        if (std::abs(a.y - b.y) > std::min(a.h, b.h) * 0.5f) return a.y < b.y;
        return a.x < b.x;
    });

    return results;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<text_box> detect(context* ctx, const float* pixels, int H, int W,
                              float prob_threshold, float box_threshold,
                              float unclip_ratio) {
    if (!ctx || !pixels) return {};

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> prob_map = forward(ctx, pixels, H, W);
    if (bench) fprintf(stderr, "[ocr-detect-bench] graph compute: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
    if (prob_map.empty()) return {};

    // The prob map may be at a different resolution than the input
    // (due to head deconvs restoring to input size).
    // Scale factor: prob_map coords → input pixel coords
    float scale_x = (float)W / ctx->last_prob_w;
    float scale_y = (float)H / ctx->last_prob_h;

    t0 = std::chrono::steady_clock::now();
    auto result = extract_boxes(prob_map.data(), ctx->last_prob_w, ctx->last_prob_h,
                         prob_threshold, box_threshold, unclip_ratio,
                         ctx->min_area, scale_x, scale_y);
    if (bench) fprintf(stderr, "[ocr-detect-bench] postprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[ocr-detect-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    return result;
}

std::vector<text_box> detect_file(context* ctx, const char* path,
                                   float prob_threshold, float box_threshold,
                                   float unclip_ratio, int target_short_side) {
    if (!ctx || !path) return {};

    // Load image
    int img_w, img_h, img_c;
    unsigned char* raw = stbi_load(path, &img_w, &img_h, &img_c, 3);
    if (!raw) {
        fprintf(stderr, "ocr_detect: cannot load image: %s\n", path);
        return {};
    }

    // Resize (keep aspect ratio, short side = target)
    float scale = (float)target_short_side / std::min(img_w, img_h);
    int new_w = (int)(img_w * scale);
    int new_h = (int)(img_h * scale);

    // Pad to multiple of pad_divisor
    int pad_w = (ctx->pad_divisor - new_w % ctx->pad_divisor) % ctx->pad_divisor;
    int pad_h = (ctx->pad_divisor - new_h % ctx->pad_divisor) % ctx->pad_divisor;
    int padded_w = new_w + pad_w;
    int padded_h = new_h + pad_h;

    // Allocate CHW buffer
    std::vector<float> pixels(3 * padded_h * padded_w, 0.0f);

    // Simple bilinear resize + normalize
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < new_h; y++) {
            float src_y = y / scale;
            int sy0 = (int)src_y;
            int sy1 = std::min(sy0 + 1, img_h - 1);
            float fy = src_y - sy0;

            for (int x = 0; x < new_w; x++) {
                float src_x = x / scale;
                int sx0 = (int)src_x;
                int sx1 = std::min(sx0 + 1, img_w - 1);
                float fx = src_x - sx0;

                float v00 = raw[(sy0 * img_w + sx0) * 3 + c];
                float v01 = raw[(sy0 * img_w + sx1) * 3 + c];
                float v10 = raw[(sy1 * img_w + sx0) * 3 + c];
                float v11 = raw[(sy1 * img_w + sx1) * 3 + c];
                float val = v00 * (1-fx) * (1-fy) + v01 * fx * (1-fy)
                          + v10 * (1-fx) * fy + v11 * fx * fy;

                // Normalize: (pixel - mean) / std
                val = (val - ctx->img_mean[c]) / ctx->img_std[c];

                // Store in CHW layout: [c * H * W + y * W + x]
                pixels[c * padded_h * padded_w + y * padded_w + x] = val;
            }
        }
    }

    stbi_image_free(raw);

    fprintf(stderr, "ocr_detect: %s %dx%d → %dx%d (padded %dx%d)\n",
            path, img_w, img_h, new_w, new_h, padded_w, padded_h);

    // Run detection
    auto boxes = detect(ctx, pixels.data(), padded_h, padded_w,
                        prob_threshold, box_threshold, unclip_ratio);

    // Scale coordinates back to original image space
    float inv_scale = 1.0f / scale;
    for (auto& b : boxes) {
        b.x *= inv_scale;
        b.y *= inv_scale;
        b.w *= inv_scale;
        b.h *= inv_scale;
        for (int i = 0; i < 4; i++) {
            b.qx[i] *= inv_scale;
            b.qy[i] *= inv_scale;
        }
    }

    return boxes;
}

const float* get_prob_map(const context* ctx, int* out_h, int* out_w) {
    if (!ctx || ctx->last_prob_map.empty()) return nullptr;
    if (out_h) *out_h = ctx->last_prob_h;
    if (out_w) *out_w = ctx->last_prob_w;
    return ctx->last_prob_map.data();
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

} // namespace ocr_detect
