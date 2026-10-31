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
#include "core/gpu_backend_pref.h"
#include "gguf.h"
#include "core/env_gate.h"
#include "core/opencv_lanczos4_u8.h"

// stb_image declarations
extern "C" {
typedef unsigned char stbi_uc;
stbi_uc * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// Debug logging gated on LAYOUT_DEBUG env var
static bool layout_debug() {
    static int val = -1;
    if (val < 0) val = (getenv("LAYOUT_DEBUG") != nullptr) ? 1 : 0;
    return val != 0;
}
#define LDBG(...)                                                                                                      \
    do {                                                                                                               \
        if (layout_debug()) fprintf(stderr, __VA_ARGS__);                                                              \
    } while (0)

namespace layout_detect {

// ---------------------------------------------------------------------------
// Label names
// ---------------------------------------------------------------------------

static const char * LABEL_NAMES[] = { "caption",
                                      "footnote",
                                      "formula",
                                      "list_item",
                                      "page_footer",
                                      "page_header",
                                      "picture",
                                      "section_header",
                                      "table",
                                      "text",
                                      "title",
                                      "document_index",
                                      "code",
                                      "checkbox_selected",
                                      "checkbox_unselected",
                                      "form",
                                      "key_value_region" };

const char * label_name(label_id id) {
    int i = (int)id;
    if (i >= 0 && i < (int)label_id::NUM_CLASSES) return LABEL_NAMES[i];
    return "unknown";
}

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------

struct conv_w {
    ggml_tensor *w = nullptr, *b = nullptr;
};

struct bottleneck {
    conv_w branch2a; // 1×1 reduce
    conv_w branch2b; // 3×3
    conv_w branch2c; // 1×1 expand
    conv_w shortcut; // 1×1 (only when dims change)
};

struct resnet50_backbone {
    conv_w stem[3]; // conv1_1, conv1_2, conv1_3 (all 3×3)
    // Stage 0: 3 blocks (64→256), Stage 1: 4 blocks (128→512)
    // Stage 2: 6 blocks (256→1024), Stage 3: 3 blocks (512→2048)
    std::vector<bottleneck> stages[4];
};

struct csp_block {
    conv_w conv1;          // 1×1, IC=2*C → C (split)
    conv_w conv2;          // 1×1, IC=2*C → C (merge)
    conv_w bottlenecks[3]; // 3×3 convs
};

struct hybrid_encoder {
    conv_w input_proj[3];       // project backbone features to 256d
    conv_w lateral_convs[2];    // for FPN
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
    ggml_tensor * pos_embed = nullptr; // 2D positional embedding
};

struct decoder_layer {
    // Self-attention
    ggml_tensor *self_qkv_w = nullptr, *self_qkv_b = nullptr;
    ggml_tensor *self_out_w = nullptr, *self_out_b = nullptr;
    // Lazily-filled caches of the self-attn weights, dequantized + transposed
    // into the layout the per-layer ggml graph uploads. Immutable after first
    // fill; before this cache the block re-read and re-transposed ~0.85 MB of
    // weights per layer on EVERY detect call.
    std::vector<float> sa_qkv_w_t, sa_qkv_b_v, sa_out_w_t, sa_out_b_v;
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
    ggml_tensor * anchors = nullptr;    // [300, 4] reference points
    ggml_tensor * valid_mask = nullptr; // [300, N] validity mask
    // Encoder output projection
    ggml_tensor * enc_proj_w = nullptr;
    ggml_tensor *enc_norm_w = nullptr, *enc_norm_b = nullptr;
    ggml_tensor *enc_score_w = nullptr, *enc_score_b = nullptr;
    ggml_tensor * enc_bbox_w[3] = {}; // MLP layers
    ggml_tensor * enc_bbox_b[3] = {};
    // Query position head (MLP)
    ggml_tensor *qpos_w[2] = {}, *qpos_b[2] = {};
    // Decoder layers
    decoder_layer layers[6];
    // Per-layer detection heads
    ggml_tensor *dec_score_w = nullptr, *dec_score_b = nullptr;
    ggml_tensor * dec_bbox_w[6][3] = {};
    ggml_tensor * dec_bbox_b[6][3] = {};
};

struct context {
    resnet50_backbone backbone;
    hybrid_encoder encoder;
    transformer_decoder decoder;

    // Preprocessing
    float img_mean[3] = { 0.485f, 0.456f, 0.406f };
    float img_std[3] = { 0.229f, 0.224f, 0.225f };
    int input_h = 640, input_w = 640;
    int num_queries = 300;
    int num_classes = 17;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr; // CPU fallback when primary is GPU
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t galloc = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 1;
    bool bench = false;

    // Architecture selection (GGUF layout.arch) and D-FINE state.
    arch_id arch = arch_id::rt_detr_v2;
    struct dfine_model * dfine = nullptr;
    std::string error;
    timings last;
    stage_map * capture = nullptr;
};

static bool dfine_load(context * ctx, const char * path, gguf_context * meta);

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool load(context ** out, const char * path, int n_threads) {
    if (!out) return false;
    *out = nullptr;
    // Architecture dispatch: GGUF `layout.arch` (absent = RT-DETRv2 heron).
    std::string arch_name = "rt_detr_v2", license_gate;
    if (gguf_context * meta = path ? core_gguf::open_metadata(path) : nullptr) {
        arch_name = core_gguf::kv_str(meta, "layout.arch", "rt_detr_v2");
        license_gate = core_gguf::kv_str(meta, "layout.license_gate", "");
        if (arch_name == "dfine") {
            if (license_gate != "oxr") {
                fprintf(stderr, "layout_detect: unsupported D-FINE GGUF (license gate '%s')\n", license_gate.c_str());
                core_gguf::free_metadata(meta);
                return false;
            }
            // OXR License gate (Apache-2.0 plus additional terms, licenses/oxr-LICENSE),
            // identical to src/oxr_ocr.cpp: every surface loading these weights.
            if (!core_env::on("CRISPEMBED_OXR") || !core_env::on("CRISPEMBED_ACCEPT_OXR_LICENSE")) {
                fprintf(stderr,
                        "layout_detect: the OXR-1.0 layout model requires CRISPEMBED_OXR=1 and\n"
                        "  CRISPEMBED_ACCEPT_OXR_LICENSE=1. OXR weights are under the OXR License (Apache-2.0 with\n"
                        "  additional terms): commercial use needs a SoMark license above USD 2M total funding or\n"
                        "  annual revenue; online services must credit SoMark OXR; no hosted API competing with\n"
                        "  SoMark's, at any size. See licenses/oxr-LICENSE.\n");
                core_gguf::free_metadata(meta);
                return false;
            }
            auto * ctx = new context();
            ctx->n_threads = std::max(1, n_threads);
            ctx->bench = core_env::on("CRISPEMBED_LAYOUT_DETECT_BENCH");
            const bool ok = dfine_load(ctx, path, meta);
            core_gguf::free_metadata(meta);
            if (!ok) {
                free(ctx);
                return false;
            }
            *out = ctx;
            return true;
        }
        core_gguf::free_metadata(meta);
        if (arch_name != "rt_detr_v2") {
            fprintf(stderr, "layout_detect: unsupported layout.arch '%s'\n", arch_name.c_str());
            return false;
        }
    }
    auto * ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;
    ctx->bench = core_env::on("CRISPEMBED_LAYOUT_DETECT_BENCH");

    LDBG("layout_detect: loading %s\n", path);

    // Load weights — prefer GPU backend when available
    bool force_cpu = (getenv("LAYOUT_DETECT_FORCE_CPU") && atoi(getenv("LAYOUT_DETECT_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    if (!core_gguf::load_weights(path, ctx->backend, "layout", ctx->wl)) {
        fprintf(stderr, "layout_detect: failed to load weights\n");
        delete ctx;
        *out = nullptr;
        return false;
    }

    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = ctx->wl.tensors.find(n);
        return it != ctx->wl.tensors.end() ? it->second : nullptr;
    };

    auto load_conv = [&](conv_w & c, const std::string & prefix) {
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
    auto & bb = ctx->backbone;
    load_conv(bb.stem[0], "model.backbone.conv1.conv1_1.conv");
    load_conv(bb.stem[1], "model.backbone.conv1.conv1_2.conv");
    load_conv(bb.stem[2], "model.backbone.conv1.conv1_3.conv");

    int block_counts[] = { 3, 4, 6, 3 };
    for (int s = 0; s < 4; s++) {
        bb.stages[s].resize(block_counts[s]);
        for (int b = 0; b < block_counts[s]; b++) {
            auto pfx = "model.backbone.res_layers." + std::to_string(s) + ".blocks." + std::to_string(b);
            auto & blk = bb.stages[s][b];
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
    auto & enc = ctx->encoder;
    for (int i = 0; i < 3; i++)
        load_conv(enc.input_proj[i], std::string("model.encoder.input_proj.") + std::to_string(i));
    for (int i = 0; i < 2; i++) {
        load_conv(enc.lateral_convs[i], std::string("model.encoder.lateral_convs.") + std::to_string(i));
        load_conv(enc.downsample_convs[i], std::string("model.encoder.downsample_convs.") + std::to_string(i));
    }
    // FPN/PAN CSP blocks
    auto load_csp = [&](csp_block & csp, const std::string & prefix) {
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
    auto & dec = ctx->decoder;
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
    for (int i = 0; i < 2; i++) {
        auto k = std::string("model.decoder.query_pos_head.layers.") + std::to_string(i);
        dec.qpos_w[i] = get(k + ".weight");
        dec.qpos_b[i] = get(k + ".bias");
    }
    // Decoder layers
    for (int i = 0; i < 6; i++) {
        auto pfx = std::string("model.decoder.decoder.layers.") + std::to_string(i);
        auto & l = dec.layers[i];
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
        if (!l.cross_sampling_offsets_w) l.cross_sampling_offsets_w = get(short_pfx + ".cross_attn.samp_offs.weight");
        l.cross_sampling_offsets_b = get(pfx + ".cross_attn.sampling_offsets.bias");
        if (!l.cross_sampling_offsets_b) l.cross_sampling_offsets_b = get(short_pfx + ".cross_attn.samp_offs.bias");
        l.cross_attn_weights_w = get(pfx + ".cross_attn.attn_weights.weight");
        if (!l.cross_attn_weights_w) l.cross_attn_weights_w = get(short_pfx + ".cross_attn.attn_wts.weight");
        l.cross_attn_weights_b = get(pfx + ".cross_attn.attention_weights.bias");
        if (!l.cross_attn_weights_b) l.cross_attn_weights_b = get(short_pfx + ".cross_attn.attn_wts.bias");
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
            auto bk = std::string("model.decoder.dec_bbox_head.") + std::to_string(i) + ".layers." + std::to_string(j);
            dec.dec_bbox_w[i][j] = get(bk + ".weight");
            dec.dec_bbox_b[i][j] = get(bk + ".bias");
        }
    }
    dec.dec_score_w = get("model.decoder.dec_score_head.5.weight");
    dec.dec_score_b = get("model.decoder.dec_score_head.5.bias");

    // Verify critical tensors
    int missing = 0;
    if (!bb.stem[0].w) {
        LDBG("  MISS: stem conv1_1\n");
        missing++;
    }
    if (!enc.aifi_qkv_w) {
        LDBG("  MISS: AIFI QKV\n");
        missing++;
    }
    if (!dec.anchors) {
        LDBG("  MISS: decoder anchors\n");
        missing++;
    }
    if (!dec.layers[0].self_qkv_w) {
        LDBG("  MISS: decoder layer 0 self QKV\n");
        missing++;
    }

    // Create backend scheduler — append CPU fallback when primary is GPU
    // (ggml_backend_sched_new asserts last backend is CPU)
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    int sched_max = 4096;
    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx->backend);
    if (ctx->backend_cpu) backends.push_back(ctx->backend_cpu);
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), sched_max, false, false);
    if (!ctx->sched) {
        fprintf(stderr, "layout_detect: failed to create scheduler\n");
        return false;
    }

    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));

    LDBG("layout_detect: loaded %zu tensors (%d missing)\n", ctx->wl.tensors.size(), missing);
    return missing == 0;
}

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

static std::vector<float> tensor_to_f32(ggml_tensor * t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else {
        size_t raw_size = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_size);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_size);
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(raw.data(), out.data(), n);
    }
    return out;
}

// Full (unmasked) multi-head attention for the layout encoder/decoder.
// Inputs Q,K,V are [head_dim, N, heads]; returns [head_dim, heads, N] — the SAME
// layout ggml_flash_attn_ext produces, so callers reshape straight to [D, N].
//
// Default is a manual masked-attention path (mul_mat + soft_max_ext + mul_mat),
// which is correct on EVERY backend/arch. `ggml_flash_attn_ext` is gated behind
// LAYOUT_DETECT_FLASH=1 because on **CUDA Pascal (sm_60)** there is no flash
// kernel and `ggml_cuda_flash_attn_ext` calls `GGML_ABORT("fatal error")`
// (ggml-cuda/fattn.cu:602) — the layout graph runs on a single CUDA backend that
// bypasses the scheduler's `supports_op` CPU fallback, so it aborts instead of
// falling back. Manual attention is a negligible cost here (Phase-2 is dominated
// by the cpu_linear projections, not this attention). Numerically ~identical to
// flash on backends that support it (verify: LAYOUT_DETECT_FLASH=1 A/B).
static ggml_tensor * layout_attn(ggml_context * g, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V, float scale) {
    static const bool use_flash = (std::getenv("LAYOUT_DETECT_FLASH") != nullptr);
    if (use_flash) {
        return ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
    }
    ggml_tensor * scores = ggml_mul_mat(g, K, Q);                    // [N_k, N_q, heads]
    scores = ggml_soft_max_ext(g, scores, nullptr, scale, 0.0f);     // softmax over keys
    ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3)); // [N, head_dim, heads]
    ggml_tensor * attn = ggml_mul_mat(g, Vt, scores);                // [head_dim, N_q, heads]
    return ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));          // [head_dim, heads, N_q]
}

static ggml_tensor * prep_conv(ggml_context * g, ggml_tensor * w, int IC, int KH, int KW) {
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

// Backbone conv dispatch. Default: ggml_conv_2d (im2col + F32 GEMM). The
// metal-prof probe showed the backbone graph is ~99.6% GPU-execute (~11.7s
// gpu_us / 44ms encode on layout-heron @ 640^2) — conv_2d_direct is a poor Metal
// kernel for these shapes, while mul_mm (GEMM) is highly optimized; the im2col
// path is ~9.8x faster (Phase-1 11.4s → 1.2s). prep_conv casts the kernel to F32,
// so the fork's ggml_conv_2d picks an F32 im2col and the GEMM stays F32 — output
// is cos≈1 vs direct (same detections; only FP reduction order differs, ≤0.001
// score / ≤0.1px bbox jitter, well within the test_layout_diff cos≥0.99 gate).
// LAYOUT_CONV_DIRECT=1 restores the old direct path (regression fallback).
static bool layout_conv_direct() {
    static int v = -1;
    if (v < 0) v = (getenv("LAYOUT_CONV_DIRECT") != nullptr) ? 1 : 0;
    return v == 1;
}
// LAYOUT_CONV_F16=1 (opt-in): run the conv GEMMs in F16. The fork's
// ggml_conv_2d forces an F32 im2col whenever the activation is F32 (a CPU
// vec_dot compatibility patch), and Metal's im2col kernel only reads an F32
// source — so the F16 shape has to be composed manually: F32 activation into
// ggml_im2col with an F16 DESTINATION type (the standard upstream path,
// supported on Metal), then an F16xF16 mul_mat, which halves im2col bytes and
// lands on Metal's F16 simdgroup mul_mm. mul_mat output is F32 either way, so
// only the conv-internal precision changes. Quality is gated on region output
// before any default talk — see PERFORMANCE.md.
static bool layout_conv_f16() {
    static int v = -1;
    if (v < 0) v = core_env::on("LAYOUT_CONV_F16") ? 1 : 0;
    return v == 1;
}
static ggml_tensor * conv2d_dispatch(ggml_context * g, ggml_tensor * w, ggml_tensor * x, int stride, int pad) {
    if (layout_conv_direct()) return ggml_conv_2d_direct(g, w, x, stride, stride, pad, pad, 1, 1);
    if (layout_conv_f16()) {
        ggml_tensor * w16 = w->type == GGML_TYPE_F16 ? w : ggml_cast(g, w, GGML_TYPE_F16);
        // [N, OH, OW, IC*KH*KW] in F16, from the F32 activation directly.
        ggml_tensor * im = ggml_im2col(g, w16, x, stride, stride, pad, pad, 1, 1, true, GGML_TYPE_F16);
        ggml_tensor * r = ggml_mul_mat(g, ggml_reshape_2d(g, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
                                       ggml_reshape_2d(g, w16, w16->ne[0] * w16->ne[1] * w16->ne[2], w16->ne[3]));
        // Same output layout as ggml_conv_2d: [OW, OH, OC, N].
        r = ggml_reshape_4d(g, r, im->ne[1], im->ne[2], im->ne[3], w16->ne[3]);
        return ggml_cont(g, ggml_permute(g, r, 0, 1, 3, 2));
    }
    return ggml_conv_2d(g, w, x, stride, stride, pad, pad, 1, 1);
}

static ggml_tensor * conv_relu(ggml_context * g, ggml_tensor * x, const conv_w & c, int IC, int KH, int KW, int stride,
                               int pad, bool relu = true) {
    if (!c.w) return x;
    auto * w = prep_conv(g, c.w, IC, KH, KW);
    x = conv2d_dispatch(g, w, x, stride, pad);
    if (c.b) {
        auto * b = c.b;
        if (b->type != GGML_TYPE_F32) b = ggml_cast(g, b, GGML_TYPE_F32);
        int OC = (int)b->ne[0];
        x = ggml_add(g, x, ggml_reshape_3d(g, b, 1, 1, OC));
    }
    if (relu) x = ggml_relu(g, x);
    return x;
}

// Conv + optional bias + SiLU activation
static ggml_tensor * conv_silu(ggml_context * g, ggml_tensor * x, const conv_w & c, int IC, int KH, int KW, int stride,
                               int pad) {
    if (!c.w) return x;
    auto * w = prep_conv(g, c.w, IC, KH, KW);
    x = conv2d_dispatch(g, w, x, stride, pad);
    if (c.b) {
        auto * b = c.b;
        if (b->type != GGML_TYPE_F32) b = ggml_cast(g, b, GGML_TYPE_F32);
        int OC = (int)b->ne[0];
        x = ggml_add(g, x, ggml_reshape_3d(g, b, 1, 1, OC));
    }
    // SiLU: x * sigmoid(x)
    x = ggml_mul(g, x, ggml_sigmoid(g, x));
    return x;
}

static ggml_tensor * linear(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    if (!w) return x;
    // ggml_mul_mat(a, b): contracts over a->ne[0], result ne[0] = a->ne[1]
    // If w->ne[0] != x->ne[0], the weight needs transposing
    if (w->ne[0] != x->ne[0] && w->ne[1] == x->ne[0]) {
        w = ggml_cont(g, ggml_transpose(g, w));
    }
    x = ggml_mul_mat(g, w, x);
    if (b) {
        if (b->type != GGML_TYPE_F32) b = ggml_cast(g, b, GGML_TYPE_F32);
        x = ggml_add(g, x, b);
    }
    return x;
}

static ggml_tensor * layer_norm(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                                float eps = 1e-5f) {
    if (!w) return x;
    x = ggml_norm(g, x, eps);
    if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
    x = ggml_mul(g, x, w);
    if (b) {
        if (b->type != GGML_TYPE_F32) b = ggml_cast(g, b, GGML_TYPE_F32);
        x = ggml_add(g, x, b);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Shared RT-DETR / D-FINE building blocks
// ---------------------------------------------------------------------------

// AIFI transformer encoder layer (HybridEncoder.encoder; post-LN, GELU-erf,
// position added to Q/K only). x: [C, N] token-major; pos_embed: C*N values.
struct aifi_w {
    ggml_tensor *qkv_w, *qkv_b, *out_w, *out_b, *norm1_w, *norm1_b;
    ggml_tensor *ffn1_w, *ffn1_b, *ffn2_w, *ffn2_b, *norm2_w, *norm2_b;
};
static ggml_tensor * aifi_encoder_layer(ggml_context * g, ggml_tensor * p5_flat, ggml_tensor * pos_embed,
                                        const aifi_w & w, int heads) {
    const int s5_c = (int)p5_flat->ne[0];
    const int n_tokens = (int)p5_flat->ne[1];
    {
        auto * x_pos = p5_flat;
        if (pos_embed) {
            auto * pe = pos_embed;
            if (pe->type != GGML_TYPE_F32) pe = ggml_cast(g, pe, GGML_TYPE_F32);
            pe = ggml_reshape_2d(g, pe, s5_c, n_tokens);
            x_pos = ggml_add(g, p5_flat, pe); // x + pos (for Q/K projection)
        }

        // Self-attention: Q/K from (x+pos), V from x — matches HF DETR pattern
        int D_a = s5_c, N_tok = n_tokens, hd = D_a / heads;
        // Compute QKV from x+pos (Q and K correct)
        auto * qkv_pos = linear(g, x_pos, w.qkv_w, w.qkv_b);
        // Compute QKV from raw x (V correct)
        auto * qkv_raw = linear(g, p5_flat, w.qkv_w, w.qkv_b);

        // Q, K from pos-projected; V from raw-projected
        auto * Q = ggml_cont(g, ggml_view_2d(g, qkv_pos, D_a, N_tok, qkv_pos->nb[1], 0));
        auto * K = ggml_cont(g, ggml_view_2d(g, qkv_pos, D_a, N_tok, qkv_pos->nb[1], D_a * sizeof(float)));
        auto * V = ggml_cont(g, ggml_view_2d(g, qkv_raw, D_a, N_tok, qkv_raw->nb[1], 2 * D_a * sizeof(float)));

        // Reshape: [D, N] → [hd, heads, N] → permute → [hd, N, heads]
        Q = ggml_reshape_3d(g, Q, hd, heads, N_tok);
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_reshape_3d(g, K, hd, heads, N_tok);
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_reshape_3d(g, V, hd, heads, N_tok);
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        auto * attn = layout_attn(g, Q, K, V, 1.0f / sqrtf((float)hd));
        // flash_attn_ext already applies permute(0,2,1,3): output is [hd, heads, N].
        // Reshape straight to [D, N] — the old manual path's extra permute scrambled it.
        attn = ggml_reshape_2d(g, attn, D_a, N_tok);

        // Output projection
        attn = linear(g, attn, w.out_w, w.out_b);

        // Post-LN: residual (raw x, no pos) + attn → norm1
        p5_flat = ggml_add(g, p5_flat, attn);
        p5_flat = layer_norm(g, p5_flat, w.norm1_w, w.norm1_b);

        // Post-LN: FFN → residual + FFN → norm2
        auto * residual = p5_flat;
        auto * ffn = linear(g, p5_flat, w.ffn1_w, w.ffn1_b);
        ffn = ggml_gelu_erf(g, ffn);
        ffn = linear(g, ffn, w.ffn2_w, w.ffn2_b);
        p5_flat = ggml_add(g, residual, ffn);
        p5_flat = layer_norm(g, p5_flat, w.norm2_w, w.norm2_b);
    }
    return p5_flat;
}

// CCFM: top-down FPN then bottom-up PAN over (S3, S4, S5), as in RT-DETR's and
// D-FINE's HybridEncoder.forward. PAN concatenates the lateral-conv'd FPN
// features. The architecture-specific blocks are supplied as callbacks.
struct ccfm_ops {
    std::function<ggml_tensor *(int, ggml_tensor *)> lateral, fpn, down, pan;
};
static void ccfm_forward(ggml_context * g, ggml_tensor * p3, ggml_tensor * p4, ggml_tensor * p5, const ccfm_ops & ops,
                         ggml_tensor ** s3, ggml_tensor ** s4, ggml_tensor ** s5) {
    // FPN step 0: lateral(p5) → upsample → cat(ip4) → block → fpn_0
    auto * lat5 = ops.lateral(0, p5);
    auto * up5 =
        ggml_interpolate(g, lat5, (int)p4->ne[0], (int)p4->ne[1], (int)lat5->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto * cat4 = ggml_concat(g, up5, p4, 2); // [W, H, 512]
    auto * fpn_0 = ops.fpn(0, cat4);

    // FPN step 1: lateral(fpn_0) → upsample → cat(ip3) → block → fpn_1
    auto * lat4 = ops.lateral(1, fpn_0);
    auto * up4 =
        ggml_interpolate(g, lat4, (int)p3->ne[0], (int)p3->ne[1], (int)lat4->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto * cat3 = ggml_concat(g, up4, p3, 2); // [W, H, 512]
    auto * fpn_1 = ops.fpn(1, cat3);

    // PAN bottom-up: uses lateral-conv'd features from FPN, not raw features!
    auto * down3 = ops.down(0, fpn_1);
    auto * cat4p = ggml_concat(g, down3, lat4, 2); // cat with lat_fpn_0, not raw fpn_0
    auto * pan_0 = ops.pan(0, cat4p);

    auto * down4 = ops.down(1, pan_0);
    auto * cat5p = ggml_concat(g, down4, lat5, 2); // cat with lat_aifi_p5, not raw p5
    auto * pan_1 = ops.pan(1, cat5p);

    *s3 = fpn_1;
    *s4 = pan_0;
    *s5 = pan_1;
}

// Decoder self-attention block: Q/K from (inp + pe), V from inp, output
// projection, residual, post-LayerNorm. inp/pe/res: [D, N]; qkv_w: [D, 3D].
static ggml_tensor * decoder_self_attn(ggml_context * gc, ggml_tensor * inp, ggml_tensor * pe_in, ggml_tensor * res_in,
                                       ggml_tensor * qkv_w_in, ggml_tensor * qkv_b_in, ggml_tensor * out_w_in,
                                       ggml_tensor * out_b_in, ggml_tensor * norm_w, ggml_tensor * norm_b, int D,
                                       int N_queries, int N_heads) {
    const int hd = D / N_heads;
    // Q/K from (inp + pe), V from inp — using input weight tensors
    auto * inp_pos = ggml_add(gc, inp, pe_in);
    // ggml_mul_mat(W[in_d, out_d], x[in_d, N]) → result[out_d, N]
    auto * qkv_pos = ggml_add(gc, ggml_mul_mat(gc, qkv_w_in, inp_pos), qkv_b_in);
    auto * qkv_raw = ggml_add(gc, ggml_mul_mat(gc, qkv_w_in, inp), qkv_b_in);

    // Extract Q[D,N], K[D,N] from qkv_pos; V[D,N] from qkv_raw
    auto * Q = ggml_cont(gc, ggml_view_2d(gc, qkv_pos, D, N_queries, qkv_pos->nb[1], 0));
    auto * K = ggml_cont(gc, ggml_view_2d(gc, qkv_pos, D, N_queries, qkv_pos->nb[1], D * sizeof(float)));
    auto * V = ggml_cont(gc, ggml_view_2d(gc, qkv_raw, D, N_queries, qkv_raw->nb[1], 2 * D * sizeof(float)));

    // Multi-head: [D, N] → [hd, heads, N] → permute → [hd, N, heads]
    Q = ggml_reshape_3d(gc, Q, hd, N_heads, N_queries);
    Q = ggml_cont(gc, ggml_permute(gc, Q, 0, 2, 1, 3));
    K = ggml_reshape_3d(gc, K, hd, N_heads, N_queries);
    K = ggml_cont(gc, ggml_permute(gc, K, 0, 2, 1, 3));
    V = ggml_reshape_3d(gc, V, hd, N_heads, N_queries);
    V = ggml_cont(gc, ggml_permute(gc, V, 0, 2, 1, 3));

    auto * attn = layout_attn(gc, Q, K, V, 1.0f / sqrtf((float)hd));
    // flash_attn_ext already applies permute(0,2,1,3): output is [hd, heads, N].
    // Reshape straight to [D, N] — the old manual path's extra permute scrambled it.
    attn = ggml_reshape_2d(gc, attn, D, N_queries);

    // Output projection + residual + norm
    attn = ggml_add(gc, ggml_mul_mat(gc, out_w_in, attn), out_b_in);
    auto * out = ggml_add(gc, res_in, attn);
    return layer_norm(gc, out, norm_w, norm_b);
}

// Decoder FFN block: linear → ReLU → linear, residual, [clamp], post-LayerNorm.
static ggml_tensor * decoder_ffn(ggml_context * gc, ggml_tensor * inp, ggml_tensor * res, ggml_tensor * w1,
                                 ggml_tensor * b1, ggml_tensor * w2, ggml_tensor * b2, ggml_tensor * norm_w,
                                 ggml_tensor * norm_b, bool clamp_fp16_range) {
    auto * up = linear(gc, inp, w1, b1);
    up = ggml_relu(gc, up);
    auto * dn = linear(gc, up, w2, b2);
    auto * out = ggml_add(gc, res, dn);
    if (clamp_fp16_range) out = ggml_clamp(gc, out, -65504.0f, 65504.0f);
    return layer_norm(gc, out, norm_w, norm_b);
}

// Multi-scale deformable sampling core (MSDeformableAttention with 4-d
// reference boxes): location = ref_xy + offset * point_scale * ref_wh *
// offset_scale, bilinear grid_sample (zeros padding, align_corners=False),
// weighted by the per-head softmaxed attention weights.
//   values:  [D][total_tokens] (channel-major), channel = head * head_dim + d
//   offsets: [(off_idx * 2 + xy) * N_queries + q], off_idx = h * total_points + point
//   weights: [off_idx * N_queries + q];  ref_points: [q * 4 + (cx, cy, w, h)]
//   cross_out (accumulated): [channel * N_queries + q]
// torch_grid=false keeps RT-DETRv2's original sample-coordinate arithmetic
// (loc * size - 0.5, (1 - f) weights); true follows torch's grid_sampler
// arithmetic (unnormalize 2 * loc - 1, corner-difference weights).
struct deform_level {
    int W, H, start, points;
    float point_scale;
};
static void deformable_sample(const float * values, int total_tokens, const deform_level * levels, int N_levels,
                              int N_heads, int value_hd, int N_queries, const float * ref_points, const float * offsets,
                              const float * attn_weights, float offset_scale, bool torch_grid, float * cross_out) {
    int total_points = 0;
    for (int lv = 0; lv < N_levels; lv++) total_points += levels[lv].points;
    for (int q = 0; q < N_queries; q++) {
        float ref_cx = ref_points[q * 4 + 0]; // cx in [0, 1]
        float ref_cy = ref_points[q * 4 + 1]; // cy in [0, 1]
        float ref_w = ref_points[q * 4 + 2];  // w in [0, 1]
        float ref_h = ref_points[q * 4 + 3];  // h in [0, 1]

        for (int h = 0; h < N_heads; h++) {
            int point_base = 0;
            for (int lv = 0; lv < N_levels; lv++) {
                int fW = levels[lv].W, fH = levels[lv].H;
                const int N_points = levels[lv].points;
                const float point_scale = levels[lv].point_scale;
                const int level_start = levels[lv].start;
                for (int pt = 0; pt < N_points; pt++) {
                    int off_idx = (h * total_points + point_base + pt);
                    float dx = offsets[(off_idx * 2 + 0) * N_queries + q];
                    float dy = offsets[(off_idx * 2 + 1) * N_queries + q];

                    // offset_scaled = raw_offset * point_scale * ref_size * offset_scale
                    // position = ref_xy + offset_scaled  (in [0, 1])
                    float px = ref_cx + dx * point_scale * ref_w * offset_scale;
                    float py = ref_cy + dy * point_scale * ref_h * offset_scale;

                    float attn_w = attn_weights[off_idx * N_queries + q];
                    if (!torch_grid) {
                        // Convert from [0,1] normalized to feature map pixel coords
                        // grid_sample(align_corners=False): pixel = loc * N - 0.5
                        float sx = px * fW - 0.5f;
                        float sy = py * fH - 0.5f;

                        int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
                        int x1 = x0 + 1, y1 = y0 + 1;
                        float fx = sx - x0, fy = sy - y0;

                        float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy);
                        float w10 = (1 - fx) * fy, w11 = fx * fy;

                        for (int d = 0; d < value_hd; d++) {
                            int vd = h * value_hd + d;
                            int base = vd * total_tokens + level_start;
                            // Zero-pad: out-of-bounds samples contribute 0
                            float v = 0;
                            if (x0 >= 0 && x0 < fW && y0 >= 0 && y0 < fH) v += w00 * values[base + y0 * fW + x0];
                            if (x1 >= 0 && x1 < fW && y0 >= 0 && y0 < fH) v += w01 * values[base + y0 * fW + x1];
                            if (x0 >= 0 && x0 < fW && y1 >= 0 && y1 < fH) v += w10 * values[base + y1 * fW + x0];
                            if (x1 >= 0 && x1 < fW && y1 >= 0 && y1 < fH) v += w11 * values[base + y1 * fW + x1];
                            cross_out[vd * N_queries + q] += attn_w * v;
                        }
                    } else {
                        // torch grid_sampler_2d (bilinear, zeros, align_corners=False)
                        const float gx = 2 * px - 1, gy = 2 * py - 1;
                        const float ix = ((gx + 1) * fW - 1) / 2;
                        const float iy = ((gy + 1) * fH - 1) / 2;
                        const float ix_nw = floorf(ix), iy_nw = floorf(iy);
                        const float ix_ne = ix_nw + 1, iy_ne = iy_nw;
                        const float ix_sw = ix_nw, iy_sw = iy_nw + 1;
                        const float ix_se = ix_nw + 1, iy_se = iy_nw + 1;
                        const float nw = (ix_se - ix) * (iy_se - iy);
                        const float ne = (ix - ix_sw) * (iy_sw - iy);
                        const float sw = (ix_ne - ix) * (iy - iy_ne);
                        const float se = (ix - ix_nw) * (iy - iy_nw);
                        const int x0 = (int)ix_nw, y0 = (int)iy_nw, x1 = x0 + 1, y1 = y0 + 1;
                        const bool in00 = x0 >= 0 && x0 < fW && y0 >= 0 && y0 < fH;
                        const bool in01 = x1 >= 0 && x1 < fW && y0 >= 0 && y0 < fH;
                        const bool in10 = x0 >= 0 && x0 < fW && y1 >= 0 && y1 < fH;
                        const bool in11 = x1 >= 0 && x1 < fW && y1 >= 0 && y1 < fH;
                        for (int d = 0; d < value_hd; d++) {
                            int vd = h * value_hd + d;
                            const float * base = values + (size_t)vd * total_tokens + level_start;
                            float v = 0;
                            if (in00) v += base[y0 * fW + x0] * nw;
                            if (in01) v += base[y0 * fW + x1] * ne;
                            if (in10) v += base[y1 * fW + x0] * sw;
                            if (in11) v += base[y1 * fW + x1] * se;
                            cross_out[vd * N_queries + q] += v * attn_w;
                        }
                    }
                }
                point_base += N_points;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ResNet-50 backbone
// ---------------------------------------------------------------------------

// Returns 3 feature maps: C3 (stride 8), C4 (stride 16), C5 (stride 32)
static void backbone_forward(ggml_context * g, const resnet50_backbone & bb, ggml_tensor * input, ggml_tensor ** c3,
                             ggml_tensor ** c4, ggml_tensor ** c5) {
    // Stem: 3 × (3×3 conv, stride 2 for first, stride 1 for rest) + maxpool
    auto * x = conv_relu(g, input, bb.stem[0], 3, 3, 3, 2, 1);
    x = conv_relu(g, x, bb.stem[1], 32, 3, 3, 1, 1);
    x = conv_relu(g, x, bb.stem[2], 32, 3, 3, 1, 1);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    // x is now stride 4, 64 channels

    // Helper: Bottleneck block
    auto bottleneck_fwd = [&](ggml_tensor * inp, const bottleneck & blk, int in_ch, int mid_ch, int out_ch,
                              int stride) {
        auto * identity = inp;
        // branch2a: 1×1, reduce (NO stride — ResNet-D puts stride on branch2b)
        auto * out = conv_relu(g, inp, blk.branch2a, in_ch, 1, 1, 1, 0);
        // branch2b: 3×3 (stride applied HERE in ResNet-D)
        out = conv_relu(g, out, blk.branch2b, mid_ch, 3, 3, stride, 1);
        // branch2c: 1×1, expand, NO relu
        out = conv_relu(g, out, blk.branch2c, mid_ch, 1, 1, 1, 0, false);
        // Shortcut (ResNet-D: avgpool + 1×1 conv for downsampling)
        if (blk.shortcut.w) {
            if (stride > 1) {
                identity = ggml_pool_2d(g, inp, GGML_OP_POOL_AVG, stride, stride, stride, stride, 0, 0);
            }
            identity = conv_relu(g, identity, blk.shortcut, in_ch, 1, 1, 1, 0, false);
        }
        // Residual + ReLU
        return ggml_relu(g, ggml_add(g, out, identity));
    };

    // Stage 0: 3 blocks, 64→256, stride 1
    int channels[] = { 64, 256, 512, 1024, 2048 };
    int mid_channels[] = { 64, 128, 256, 512 };
    int strides[] = { 1, 2, 2, 2 };

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
            int blk_in = (b == 0) ? in_ch : channels[s + 1];
            int stride = (b == 0) ? strides[s] : 1;
            x = bottleneck_fwd(x, bb.stages[s][b], blk_in, mid_channels[s], channels[s + 1], stride);
            // Mark each block output for diff comparison
            char name[32];
            snprintf(name, sizeof(name), "bb_s%d_b%d", s, b);
            ggml_set_name(x, name);
            ggml_set_output(x);
        }
        // Save outputs for FPN
        if (s == 1) *c3 = x; // stride 8, 512 ch
        if (s == 2) *c4 = x; // stride 16, 1024 ch
        if (s == 3) *c5 = x; // stride 32, 2048 ch
    }
}

// ---------------------------------------------------------------------------
// Hybrid encoder forward
// ---------------------------------------------------------------------------

// Multi-head self-attention for AIFI encoder (single layer, 8 heads, 256d)
static ggml_tensor * aifi_self_attn(ggml_context * g, ggml_tensor * x, const hybrid_encoder & enc) {
    // x: [256, N] where N = spatial tokens
    auto * residual = x;

    // QKV projection: split in_proj into Q, K, V
    auto * qkv = linear(g, x, enc.aifi_qkv_w, enc.aifi_qkv_b); // [768, N]
    LDBG("  AIFI: qkv=[%lld,%lld]\n", (long long)qkv->ne[0], (long long)qkv->ne[1]);
    int D = 256, N_heads = 8, head_dim = D / N_heads; // 32

    // Split QKV → Q[256,N], K[256,N], V[256,N]
    auto * q = ggml_cont(g, ggml_view_2d(g, qkv, D, qkv->ne[1], qkv->nb[1], 0));
    auto * k = ggml_cont(g, ggml_view_2d(g, qkv, D, qkv->ne[1], qkv->nb[1], D * sizeof(float)));
    auto * v = ggml_cont(g, ggml_view_2d(g, qkv, D, qkv->ne[1], qkv->nb[1], 2 * D * sizeof(float)));

    // Reshape for multi-head: [head_dim, N, N_heads]
    int64_t N = qkv->ne[1];
    q = ggml_reshape_3d(g, q, head_dim, N, N_heads);
    k = ggml_reshape_3d(g, k, head_dim, N, N_heads);
    v = ggml_reshape_3d(g, v, head_dim, N, N_heads);

    // Permute to [head_dim, N, N_heads] → compatible with flash_attn
    // ggml flash_attn_ext expects q[D,N,H], k[D,N,H], v[D,N,H]
    auto * attn = layout_attn(g, q, k, v, 1.0f / sqrtf(head_dim));
    // flash_attn_ext already applies permute(0,2,1,3): output is [hd, nh, N].
    // Reshape straight to [D, N]; the old manual-path permute scrambled it.
    // (NB: this helper is currently unused — encoder_forward inlines its own AIFI block.)
    attn = ggml_reshape_2d(g, attn, D, N); // [D=hd*nh, N]

    // Output projection
    attn = linear(g, attn, enc.aifi_out_w, enc.aifi_out_b);

    // Residual + LayerNorm
    x = ggml_add(g, residual, attn);
    x = layer_norm(g, x, enc.aifi_norm1_w, enc.aifi_norm1_b);

    // FFN: linear1 → ReLU → linear2
    residual = x;
    auto * ffn = linear(g, x, enc.aifi_ffn1_w, enc.aifi_ffn1_b);
    ffn = ggml_relu(g, ffn);
    ffn = linear(g, ffn, enc.aifi_ffn2_w, enc.aifi_ffn2_b);

    x = ggml_add(g, residual, ffn);
    x = layer_norm(g, x, enc.aifi_norm2_w, enc.aifi_norm2_b);

    return x;
}

// Encoder: project backbone features → FPN → PAN → AIFI on top scale
// Returns 3 multi-scale features: [256, H*W] for each scale
static void encoder_forward(ggml_context * g, const hybrid_encoder & enc, ggml_tensor * c3, ggml_tensor * c4,
                            ggml_tensor * c5, ggml_tensor ** s3, ggml_tensor ** s4, ggml_tensor ** s5) {
    // Input projection: reduce channels to 256
    int c3_ch = (int)c3->ne[2], c4_ch = (int)c4->ne[2], c5_ch = (int)c5->ne[2];
    LDBG("  encoder input_proj: c3_ch=%d c4_ch=%d c5_ch=%d\n", c3_ch, c4_ch, c5_ch);
    if (enc.input_proj[0].w)
        LDBG("  input_proj[0].w: [%lld,%lld] ndim=%d\n", (long long)enc.input_proj[0].w->ne[0],
             (long long)enc.input_proj[0].w->ne[1], ggml_n_dims(enc.input_proj[0].w));
    auto * p3 = conv_relu(g, c3, enc.input_proj[0], c3_ch, 1, 1, 1, 0, false);
    ggml_set_name(p3, "ip3");
    ggml_set_output(p3);
    auto * p4 = conv_relu(g, c4, enc.input_proj[1], c4_ch, 1, 1, 1, 0, false);
    ggml_set_name(p4, "ip4");
    ggml_set_output(p4);
    auto * p5 = conv_relu(g, c5, enc.input_proj[2], c5_ch, 1, 1, 1, 0, false);
    ggml_set_name(p5, "ip5");
    ggml_set_output(p5);

    // AIFI self-attention on S5 (smallest scale, 20×20 = 400 tokens)
    // p5 is [W, H, C] in ggml. We need [D, N] = [C, W*H] for attention.
    // Permute [W,H,C] → [C,W,H] then reshape [C, W*H]
    int s5_w = (int)p5->ne[0], s5_h = (int)p5->ne[1], s5_c = (int)p5->ne[2];
    LDBG("  AIFI: p5=[%lld,%lld,%lld] s5_w=%d s5_h=%d s5_c=%d\n", (long long)p5->ne[0], (long long)p5->ne[1],
         (long long)p5->ne[2], s5_w, s5_h, s5_c);
    auto * p5_perm = ggml_cont(g, ggml_permute(g, p5, 1, 2, 0, 3)); // ne: [C, W, H]
    LDBG("  AIFI: p5_perm=[%lld,%lld,%lld,%lld] nelems=%lld\n", (long long)p5_perm->ne[0], (long long)p5_perm->ne[1],
         (long long)p5_perm->ne[2], (long long)p5_perm->ne[3], (long long)ggml_nelements(p5_perm));
    LDBG("  AIFI: reshape to [%d, %d] = %d\n", s5_c, s5_w * s5_h, s5_c * s5_w * s5_h);
    auto * p5_flat = ggml_reshape_2d(g, p5_perm, s5_c, s5_w * s5_h); // ne: [C, W*H]
    LDBG("  AIFI: p5_flat done\n");
    // Transpose to [256, N] — ggml reshape gives [C, H*W], need [C, N] which is same
    // AIFI encoder: pos_embed → self-attn → residual+norm1 → FFN(GELU) → residual+norm2
    // AIFI: self-attn(pos on Q/K only) → residual+norm1 → FFN(GELU_erf) → residual+norm2
    // HF: normalize_before=False (post-LN), pos added to Q/K only (not V)
    p5_flat = aifi_encoder_layer(g, p5_flat, enc.pos_embed,
                                 { enc.aifi_qkv_w, enc.aifi_qkv_b, enc.aifi_out_w, enc.aifi_out_b, enc.aifi_norm1_w,
                                   enc.aifi_norm1_b, enc.aifi_ffn1_w, enc.aifi_ffn1_b, enc.aifi_ffn2_w, enc.aifi_ffn2_b,
                                   enc.aifi_norm2_w, enc.aifi_norm2_b },
                                 8);
    ggml_set_name(p5_flat, "aifi_flat");
    ggml_set_output(p5_flat);
    LDBG("  AIFI done: p5_flat=[%lld,%lld]\n", (long long)p5_flat->ne[0], (long long)p5_flat->ne[1]);
    // Reshape back to spatial: [C, W*H] → [C, W, H] → permute to [W, H, C]
    auto * p5_spatial = ggml_reshape_3d(g, p5_flat, s5_c, s5_w, s5_h);
    p5 = ggml_cont(g, ggml_permute(g, p5_spatial, 2, 0, 1, 3)); // [W, H, C]
    LDBG("  AIFI reshape back: p5=[%lld,%lld,%lld]\n", (long long)p5->ne[0], (long long)p5->ne[1],
         (long long)p5->ne[2]);

    // FPN top-down pathway
    LDBG("  FPN: p3=[%lld,%lld,%lld] p4=[%lld,%lld,%lld] p5=[%lld,%lld,%lld]\n", (long long)p3->ne[0],
         (long long)p3->ne[1], (long long)p3->ne[2], (long long)p4->ne[0], (long long)p4->ne[1], (long long)p4->ne[2],
         (long long)p5->ne[0], (long long)p5->ne[1], (long long)p5->ne[2]);
    // FPN CSP block helper:
    // cat = concat(upsample/downsample, lateral)  [512ch]
    // branch_a = conv1(cat) → SiLU → bottleneck0 → SiLU → ... → SiLU  [256ch]
    // branch_b = conv2(cat) → SiLU  [256ch]
    // output = Add(branch_a, branch_b)  [256ch]
    static int csp_idx = 0;
    auto run_csp = [&](ggml_tensor * cat_input, const csp_block & blk) -> ggml_tensor * {
        int cat_ch = (int)cat_input->ne[2];
        // Branch A: conv1 → SiLU → bottlenecks → SiLU
        auto * a = conv_silu(g, cat_input, blk.conv1, cat_ch, 1, 1, 1, 0);
        {
            char n[32];
            snprintf(n, 32, "csp%d_conv1", csp_idx);
            ggml_set_name(a, n);
            ggml_set_output(a);
        }
        for (int j = 0; j < 3; j++) {
            if (blk.bottlenecks[j].w) {
                a = conv_silu(g, a, blk.bottlenecks[j], 256, 3, 3, 1, 1);
                char n[32];
                snprintf(n, 32, "csp%d_bn%d", csp_idx, j);
                ggml_set_name(a, n);
                ggml_set_output(a);
            }
        }
        // Branch B: conv2 → SiLU
        auto * b = conv_silu(g, cat_input, blk.conv2, cat_ch, 1, 1, 1, 0);
        {
            char n[32];
            snprintf(n, 32, "csp%d_conv2", csp_idx);
            ggml_set_name(b, n);
            ggml_set_output(b);
        }
        // Add branches
        auto * out = ggml_add(g, a, b);
        {
            char n[32];
            snprintf(n, 32, "csp%d_out", csp_idx);
            ggml_set_name(out, n);
            ggml_set_output(out);
        }
        csp_idx++;
        return out;
    };

    // Mark AIFI output for comparison
    ggml_set_name(p5, "aifi_out");
    ggml_set_output(p5);

    // FPN top-down + PAN bottom-up (matching HF RTDetrV2HybridEncoder):
    // HF stores lateral-conv'd features and reuses them in the PAN.
    ccfm_ops ops;
    ops.lateral = [&](int i, ggml_tensor * x) {
        auto * lat = conv_silu(g, x, enc.lateral_convs[i], 256, 1, 1, 1, 0);
        if (i == 0) {
            ggml_set_name(lat, "lat5_silu");
            ggml_set_output(lat);
        }
        return lat;
    };
    ops.fpn = [&](int i, ggml_tensor * x) { return run_csp(x, enc.fpn_blocks[i]); };
    ops.down = [&](int i, ggml_tensor * x) { return conv_silu(g, x, enc.downsample_convs[i], 256, 3, 3, 2, 1); };
    ops.pan = [&](int i, ggml_tensor * x) { return run_csp(x, enc.pan_blocks[i]); };
    ccfm_forward(g, p3, p4, p5, ops, s3, s4, s5);
}

// ---------------------------------------------------------------------------
// Bilinear grid sampling (CPU-side, for deformable attention)
// ---------------------------------------------------------------------------

// Sample a 2D feature map at fractional position (px, py) using bilinear interpolation.
// feat: [C, H, W] row-major. Returns C-dimensional vector.
static void bilinear_sample(const float * feat, int W, int H, int C, float px, float py, float * out) {
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
        out[c] = w00 * feat[c * H * W + y0 * W + x0] + w01 * feat[c * H * W + y0 * W + x1] +
                 w10 * feat[c * H * W + y1 * W + x0] + w11 * feat[c * H * W + y1 * W + x1];
    }
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

std::vector<region> detect(context * ctx, const float * pixels, int orig_h, int orig_w, float score_threshold) {
    if (!ctx || !pixels || ctx->arch != arch_id::rt_detr_v2) return {};

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    const int H = ctx->input_h, W = ctx->input_w; // 640
    const int D = 256, N_heads = 8, N_queries = 300;
    const int N_levels = 3, N_points = 4;

    // --- Phase 1: Backbone + Encoder (ggml graph) ---
    // This produces multi-scale features via ggml graph compute
    int max_nodes = 4096;
    size_t buf_size = ggml_tensor_overhead() * (max_nodes + 500) + ggml_graph_overhead_custom(max_nodes, false) +
                      256 * 1024 * 1024; // 256 MB extra for tensor descriptors
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params ip = { buf_size, buf.data(), true };
    ggml_context * g = ggml_init(ip);

    // Input: [W, H, 3] in ggml layout
    ggml_tensor * input = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, 3);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    // Backbone
    // Verify stem weight is the folded version
    if (std::getenv("CRISPEMBED_LAYOUT_DEBUG")) {
        auto * t = ctx->backbone.stem[0].w;
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<float> fv(3);
        if (t->type == GGML_TYPE_F32)
            memcpy(fv.data(), raw.data(), 3 * sizeof(float));
        else if (t->type == GGML_TYPE_F16) {
            auto * src = (const ggml_fp16_t *)raw.data();
            for (int i = 0; i < 3; i++) fv[i] = ggml_fp16_to_fp32(src[i]);
        }
        LDBG("  stem[0].w first3: %.6f %.6f %.6f (expected: 0.002018)\n", fv[0], fv[1], fv[2]);
    }
    LDBG("layout_detect: building backbone + encoder graph...\n");
    ggml_tensor *c3, *c4, *c5;
    backbone_forward(g, ctx->backbone, input, &c3, &c4, &c5);

    // Encoder
    ggml_tensor *s3, *s4, *s5;
    encoder_forward(g, ctx->encoder, c3, c4, c5, &s3, &s4, &s5);

    // Mark backbone + encoder outputs
    ggml_set_name(c3, "c3");
    ggml_set_output(c3);
    ggml_set_name(c4, "c4");
    ggml_set_output(c4);
    ggml_set_name(c5, "c5");
    ggml_set_output(c5);
    ggml_set_name(s3, "s3");
    ggml_set_output(s3);
    ggml_set_name(s4, "s4");
    ggml_set_output(s4);
    ggml_set_name(s5, "s5");
    ggml_set_output(s5);

    // Build + compute backbone+encoder graph
    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);
    // Expand all outputs (backbone debug + encoder outputs)
    ggml_build_forward_expand(gf, s3);
    ggml_build_forward_expand(gf, s4);
    ggml_build_forward_expand(gf, s5);

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "layout_detect: backbone graph alloc failed\n");
        ggml_free(g);
        return {};
    }

    // Set input pixels
    ggml_tensor * inp_t = ggml_graph_get_tensor(gf, "input");
    if (inp_t) ggml_backend_tensor_set(inp_t, pixels, 0, 3 * H * W * sizeof(float));

    // Compute
    LDBG("layout_detect: computing backbone + encoder...\n");
    auto t_phase1 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(ctx->backend, gf);
    const double _p1_compute_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_phase1).count();

    // Per-block backbone comparison (diagnostic only — the tensor read-backs are
    // pure debug work, so gate the whole block, not just the range prints).
    if (layout_debug()) {
        // Read stem
        auto read_range = [&](const char * name) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (!t) {
                fprintf(stderr, "  %s: NOT FOUND\n", name);
                return;
            }
            int n = ggml_nelements(t);
            std::vector<float> data(n);
            ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
            float fmin = 1e9, fmax = -1e9;
            for (auto v : data) {
                fmin = std::min(fmin, v);
                fmax = std::max(fmax, v);
            }
            LDBG("  %s: [%lld,%lld,%lld] range=[%.4f, %.4f]\n", name, (long long)t->ne[0], (long long)t->ne[1],
                 (long long)t->ne[2], fmin, fmax);
        };
        read_range("bb_stem");
        read_range("ip3");
        read_range("ip4");
        read_range("ip5");
        read_range("aifi_out");
        read_range("lat5_silu");
        // CSP block 0 internals
        read_range("csp0_conv1");
        read_range("csp0_bn0");
        read_range("csp0_bn1");
        read_range("csp0_bn2");
        read_range("csp0_conv2");
        read_range("csp0_out");
        int block_counts[] = { 3, 4, 6, 3 };
        for (int s = 0; s < 4; s++) {
            for (int b = 0; b < block_counts[s]; b++) {
                char name[32];
                snprintf(name, sizeof(name), "bb_s%d_b%d", s, b);
                read_range(name);
            }
        }
    }
    // Dump encoder intermediates for diff comparison (ggml raw byte order)
    if (getenv("LAYOUT_DEBUG")) {
        auto dump_ggml_tensor = [&](const char * name) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (!t) return;
            int n = ggml_nelements(t);
            std::vector<float> data(n);
            ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
            char fname[128];
            snprintf(fname, sizeof(fname), "/tmp/cpp_%s.bin", name);
            FILE * fp = fopen(fname, "wb");
            if (fp) {
                fwrite(data.data(), sizeof(float), n, fp);
                fclose(fp);
            }
        };
        dump_ggml_tensor("ip3");
        dump_ggml_tensor("ip4");
        dump_ggml_tensor("ip5");
        dump_ggml_tensor("aifi_flat");
    }

    // Read encoder outputs to CPU buffers
    // s3: [W3, H3, 256], s4: [W4, H4, 256], s5: [W5, H5, 256]
    struct scale_feat {
        std::vector<float> data; // CHW layout: [C, H, W]
        int W, H, C;
    };
    scale_feat feats[3];
    const char * feat_names[] = { "s3", "s4", "s5" };
    int total_tokens = 0;
    for (int i = 0; i < 3; i++) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, feat_names[i]);
        feats[i].W = (int)t->ne[0];
        feats[i].H = (int)t->ne[1];
        feats[i].C = (int)t->ne[2];
        int n = feats[i].W * feats[i].H * feats[i].C;
        feats[i].data.resize(n);
        ggml_backend_tensor_get(t, feats[i].data.data(), 0, n * sizeof(float));
        total_tokens += feats[i].W * feats[i].H;
        // Print stats for comparison with Python reference
        float fmin = 1e9, fmax = -1e9;
        for (auto v : feats[i].data) {
            fmin = std::min(fmin, v);
            fmax = std::max(fmax, v);
        }
        LDBG("  s%d: %dx%d x %d range=[%.4f, %.4f]\n", i + 3, feats[i].W, feats[i].H, feats[i].C, fmin, fmax);
    }

    // Dump s3/s4/s5 for diff comparison
    if (getenv("LAYOUT_DEBUG")) {
        for (int i = 0; i < 3; i++) {
            char fname[128];
            snprintf(fname, sizeof(fname), "/tmp/cpp_s%d.bin", i + 3);
            FILE * fp = fopen(fname, "wb");
            if (fp) {
                fwrite(feats[i].data.data(), sizeof(float), feats[i].data.size(), fp);
                fclose(fp);
            }
        }
    }

    ggml_free(g);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_phase1).count();
        fprintf(
            stderr,
            "[layout_detect-bench] Phase 1 backbone+encoder: %.1f ms (graph compute: %.1f, feat readback+misc: %.1f)\n",
            ms, _p1_compute_ms, ms - _p1_compute_ms);
    }

    // --- Phase 2: Decoder (CPU-side) ---
    auto t_phase2 = std::chrono::steady_clock::now();

    auto cpu_linear = [nthreads = ctx->n_threads](const float * x, float * y, int in_d, int out_d, int N,
                                                  ggml_tensor * w_t, ggml_tensor * b_t) {
        auto W = tensor_to_f32(w_t);
        auto b_v = tensor_to_f32(b_t);
        if (W.empty()) W.resize(out_d * in_d, 0.0f);
        if (b_v.empty()) b_v.resize(out_d, 0.0f);
        auto & b = b_v;
        // Auto-detect weight convention for non-square weights from ggml ne[0].
        // Square weights (in_d==out_d): always MatMul (in, out) convention.
        // Non-square: ne[0] reveals numpy col count. If ne[0]==out_d → (in,out).
        //   If ne[0]==in_d and ne[0]!=out_d → (out,in), use transposed access.
        // Gemm(transB=1) weights are pre-transposed by converter to (in,out).
        bool transposed = false;
        if (w_t && in_d != out_d) {
            int ne0 = (int)w_t->ne[0];
            transposed = (ne0 != out_d && ne0 == in_d);
        }
        // y[o,:] = b[o] + sum_i W[o,i] * x[i,:], computed as contiguous AXPYs over the
        // N axis. x[i,:] and y[o,:] are contiguous (col-major [dim, N]), so the inner
        // loop vectorizes — unlike the previous (o,n)-first form whose inner reduction
        // strode x by N. The per-output accumulation order (i ascending) is unchanged,
        // so the result is byte-identical to the old scalar loop.
        const size_t Nz = (size_t)N;
        // Each output row o writes a disjoint yo, so the o-loop parallelizes
        // byte-identically. This is the dominant Phase-2 cost (the deformable
        // sample loop is ~1.5%); honors ctx->n_threads (default 1 = old behavior).
        auto compute_rows = [&](int o0, int o1) {
            for (int o = o0; o < o1; o++) {
                float * yo = y + (size_t)o * Nz;
                const float bo = b[o];
                for (int n = 0; n < N; n++) yo[n] = bo;
                for (int i = 0; i < in_d; i++) {
                    const float w = transposed ? W[o * in_d + i] : W[o + i * out_d];
                    const float * xi = x + (size_t)i * Nz;
                    for (int n = 0; n < N; n++) yo[n] += w * xi[n];
                }
            }
        };
        int nt = std::min(nthreads, out_d);
        // Only thread when the matmul is big enough to amortize thread spawn.
        if (nt <= 1 || (size_t)out_d * in_d * Nz < (size_t)(1u << 20)) {
            compute_rows(0, out_d);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(nt);
            int chunk = (out_d + nt - 1) / nt;
            for (int t = 0; t < nt; t++) {
                int o0 = t * chunk, o1 = std::min(o0 + chunk, out_d);
                if (o0 >= o1) break;
                pool.emplace_back(compute_rows, o0, o1);
            }
            for (auto & th : pool) th.join();
        }
    };

    auto cpu_layernorm = [](float * x, int D, int N, ggml_tensor * w_t, ggml_tensor * b_t) {
        auto w = tensor_to_f32(w_t);
        auto b = tensor_to_f32(b_t);
        if (w.empty()) w.resize(D, 1.0f);
        if (b.empty()) b.resize(D, 0.0f);
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
    auto _t_lp = std::chrono::steady_clock::now();
    for (int lv = 0; lv < 3; lv++) {
        level_starts[lv] = offset;
        level_sizes[lv] = feats[lv].W * feats[lv].H;
        int N_lv = level_sizes[lv];

        // Convert ggml [W, H, C] to [C, N] column-major
        std::vector<float> feat_col(D * N_lv);
        for (int c = 0; c < D; c++)
            for (int s = 0; s < N_lv; s++) feat_col[c * N_lv + s] = feats[lv].data[s + c * N_lv];

        // Apply decoder input_proj (1×1 conv = linear projection)
        // Conv weight is in ONNX Conv (out, in) convention — NOT MatMul convention.
        // y[o] = sum_i W_conv[o, i] * x[i] + b[o]
        // In ggml flat: W_conv[o, i] = flat[i + o * in_d] (numpy row-major (out, in))
        if (ctx->decoder.input_proj[lv].w) {
            std::vector<float> proj(D * N_lv);
            // Use Conv convention: W[i + o * in_d]
            {
                auto * w_t = ctx->decoder.input_proj[lv].w;
                auto * b_t = ctx->decoder.input_proj[lv].b;
                auto W = tensor_to_f32(w_t);
                auto b = tensor_to_f32(b_t);
                if (W.empty()) W.resize(D * D, 0.0f);
                if (b.empty()) b.resize(D, 0.0f);
                // AXPY form of the same projection, threaded over output rows
                // (the 2a43e4f4 cpu_linear pattern; cpu_linear itself can't be
                // used here because its square-weight heuristic would pick the
                // MatMul convention and this weight is ONNX Conv (out, in)).
                // Per-element accumulation order over i is unchanged and each
                // output row is written by exactly one thread, so the result
                // is byte-identical to the old (n, o, i) scalar nest — which
                // strode feat_col by N_lv in its inner reduction and was
                // measured at 549 ms of the 856 ms Phase 2 (64%) on
                // scan_page_pd at -t 4, the actual dominant decoder cost.
                auto compute_rows = [&](int o0, int o1) {
                    for (int o = o0; o < o1; o++) {
                        float * po = proj.data() + (size_t)o * N_lv;
                        const float bo = b[o];
                        for (int n = 0; n < N_lv; n++) po[n] = bo;
                        const float * wrow = W.data() + (size_t)o * D;
                        for (int i = 0; i < D; i++) {
                            const float w = wrow[i];
                            const float * xi = feat_col.data() + (size_t)i * N_lv;
                            for (int n = 0; n < N_lv; n++) po[n] += w * xi[n];
                        }
                    }
                };
                const int nt = std::max(1, std::min(ctx->n_threads, D));
                if (nt <= 1) {
                    compute_rows(0, D);
                } else {
                    std::vector<std::thread> pool;
                    const int chunk = (D + nt - 1) / nt;
                    for (int t = 0; t < nt; t++) {
                        const int o0 = t * chunk, o1 = std::min(D, o0 + chunk);
                        if (o0 < o1) pool.emplace_back(compute_rows, o0, o1);
                    }
                    for (auto & th : pool) th.join();
                }
            }
            // Copy projected features to memory
            for (int c = 0; c < D; c++)
                for (int s = 0; s < N_lv; s++) memory[c * total_tokens + offset + s] = proj[c * N_lv + s];
        } else {
            // No projection — use raw features
            for (int c = 0; c < D; c++)
                for (int s = 0; s < N_lv; s++) memory[c * total_tokens + offset + s] = feat_col[c * N_lv + s];
        }
        offset += N_lv;
    }
    const double _levproj_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_lp).count();

    // Initialize queries from anchors
    // anchors: [300, 4] — reference points (cx, cy, w, h) in [0, 1]
    auto anchors = tensor_to_f32(ctx->decoder.anchors);
    if (anchors.empty()) anchors.resize(N_queries * 4, 0.0f);

    // Query embeddings: project encoder output to get initial queries
    // Helper: CPU-side matrix multiply y = W @ x + b
    // W: [out, in], x: [in, N], y: [out, N]

    // CPU LayerNorm

    // Apply valid_mask to memory (zeros out padding positions)
    if (ctx->decoder.valid_mask) {
        // valid_mask is [1, N_total, 1] — broadcast over channels
        auto mask = tensor_to_f32(ctx->decoder.valid_mask);
        for (int n = 0; n < total_tokens; n++)
            if (mask[n] == 0.0f)
                for (int d = 0; d < D; d++) memory[d * total_tokens + n] = 0.0f;
        if (getenv("LAYOUT_DEBUG")) {
            int n_valid = 0;
            for (auto v : mask)
                if (v > 0) n_valid++;
            LDBG("  valid_mask: %d/%d valid tokens\n", n_valid, total_tokens);
        }
    }

    // --- Query initialization: select top-K from encoder output ---
    // 1. Project memory through enc_output: linear + norm
    std::vector<float> enc_proj(D * total_tokens);
    cpu_linear(memory.data(), enc_proj.data(), D, D, total_tokens, ctx->decoder.enc_proj_w, nullptr);
    // Add enc_output.proj.bias if exists
    {
        ggml_tensor * proj_b = ctx->wl.tensors.count("model.decoder.enc_output.proj.bias")
                                   ? ctx->wl.tensors.at("model.decoder.enc_output.proj.bias")
                                   : nullptr;
        if (proj_b) {
            auto bias = tensor_to_f32(proj_b);
            for (int n = 0; n < total_tokens; n++)
                for (int d = 0; d < D; d++) enc_proj[d * total_tokens + n] += bias[d];
        }
    }
    cpu_layernorm(enc_proj.data(), D, total_tokens, ctx->decoder.enc_norm_w, ctx->decoder.enc_norm_b);

    // Debug: inject HF reference enc_output for decoder parity testing
    if (getenv("LAYOUT_INJECT_REF")) {
        FILE * rfp = fopen(getenv("LAYOUT_INJECT_REF"), "rb");
        if (rfp) {
            // Reference is [N_total, D] row-major → convert to [D, N_total] col-major
            std::vector<float> ref_row(D * total_tokens);
            fread(ref_row.data(), sizeof(float), D * total_tokens, rfp);
            fclose(rfp);
            for (int n = 0; n < total_tokens; n++)
                for (int d = 0; d < D; d++) enc_proj[d * total_tokens + n] = ref_row[n * D + d];
            LDBG("  INJECTED HF enc_output from %s\n", getenv("LAYOUT_INJECT_REF"));
        }
    }
    // Debug: also inject HF reference memory (raw encoder features for cross-attn)
    if (getenv("LAYOUT_INJECT_MEMORY")) {
        FILE * rfp = fopen(getenv("LAYOUT_INJECT_MEMORY"), "rb");
        if (rfp) {
            // Reference stored as [N_total, D] row-major → [D, N_total] col-major
            std::vector<float> ref_row(D * total_tokens);
            fread(ref_row.data(), sizeof(float), D * total_tokens, rfp);
            fclose(rfp);
            for (int n = 0; n < total_tokens; n++)
                for (int d = 0; d < D; d++) memory[d * total_tokens + n] = ref_row[n * D + d];
            LDBG("  INJECTED HF memory from %s\n", getenv("LAYOUT_INJECT_MEMORY"));
        }
    }

    // Debug: dump raw memory for parity comparison
    if (getenv("LAYOUT_DEBUG")) {
        // Dump memory as [N_total, D] row-major
        std::vector<float> mem_row(D * total_tokens);
        for (int n = 0; n < total_tokens; n++)
            for (int d = 0; d < D; d++) mem_row[n * D + d] = memory[d * total_tokens + n];
        FILE * mfp = fopen("/tmp/cpp_raw_memory.bin", "wb");
        if (mfp) {
            fwrite(mem_row.data(), sizeof(float), D * total_tokens, mfp);
            fclose(mfp);
        }
        // Print per-level ranges
        for (int lv = 0; lv < 3; lv++) {
            float lmin = 1e9, lmax = -1e9;
            for (int i = level_starts[lv]; i < level_starts[lv] + level_sizes[lv]; i++)
                for (int d = 0; d < D; d++) {
                    float v = memory[d * total_tokens + i];
                    lmin = std::min(lmin, v);
                    lmax = std::max(lmax, v);
                }
            LDBG("  memory s%d (%d tokens): range=[%.4f, %.4f]\n", lv + 3, level_sizes[lv], lmin, lmax);
        }
        LDBG("  memory[s3,0,:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", mem_row[0], mem_row[1], mem_row[2],
             mem_row[3], mem_row[4], mem_row[5], mem_row[6], mem_row[7]);
        LDBG("  memory[s4,0,:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", mem_row[6400 * D + 0],
             mem_row[6400 * D + 1], mem_row[6400 * D + 2], mem_row[6400 * D + 3], mem_row[6400 * D + 4],
             mem_row[6400 * D + 5], mem_row[6400 * D + 6], mem_row[6400 * D + 7]);
    }

    // Debug: dump enc_proj for parity comparison
    if (getenv("LAYOUT_DEBUG")) {
        // Convert from [D, N_total] col-major to [N_total, D] row-major for comparison
        std::vector<float> enc_proj_row(D * total_tokens);
        for (int n = 0; n < total_tokens; n++)
            for (int d = 0; d < D; d++) enc_proj_row[n * D + d] = enc_proj[d * total_tokens + n];
        FILE * fp = fopen("/tmp/cpp_enc_output.bin", "wb");
        if (fp) {
            fwrite(enc_proj_row.data(), sizeof(float), D * total_tokens, fp);
            fclose(fp);
        }
        LDBG("  enc_proj: range=[%.4f, %.4f] (dumped to /tmp/cpp_enc_output.bin)\n",
             *std::min_element(enc_proj.begin(), enc_proj.end()), *std::max_element(enc_proj.begin(), enc_proj.end()));
        LDBG("  enc_proj[0,:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", enc_proj_row[0], enc_proj_row[1],
             enc_proj_row[2], enc_proj_row[3], enc_proj_row[4], enc_proj_row[5], enc_proj_row[6], enc_proj_row[7]);
    }

    // 2. Score each token
    std::vector<float> enc_scores(ctx->num_classes * total_tokens);
    cpu_linear(enc_proj.data(), enc_scores.data(), D, ctx->num_classes, total_tokens, ctx->decoder.enc_score_w,
               ctx->decoder.enc_score_b);

    // 3. Find top-K tokens by max class score
    std::vector<std::pair<float, int>> token_scores(total_tokens);
    for (int n = 0; n < total_tokens; n++) {
        float max_s = -1e9f;
        for (int c = 0; c < ctx->num_classes; c++) max_s = std::max(max_s, enc_scores[c * total_tokens + n]);
        token_scores[n] = { max_s, n };
    }
    std::partial_sort(token_scores.begin(), token_scores.begin() + N_queries, token_scores.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    // 4. Initialize queries from selected tokens
    std::vector<float> queries(D * N_queries, 0.0f);
    for (int q = 0; q < N_queries; q++) {
        int token_idx = token_scores[q].second;
        for (int d = 0; d < D; d++) queries[d * N_queries + q] = enc_proj[d * total_tokens + token_idx];
    }

    // Debug: dump initial queries and top-K indices
    if (getenv("LAYOUT_DEBUG")) {
        LDBG("  top-K indices first 5: %d %d %d %d %d\n", token_scores[0].second, token_scores[1].second,
             token_scores[2].second, token_scores[3].second, token_scores[4].second);
        // Dump queries as [N, D] row-major
        std::vector<float> q_row(D * N_queries);
        for (int q = 0; q < N_queries; q++)
            for (int d = 0; d < D; d++) q_row[q * D + d] = queries[d * N_queries + q];
        FILE * fp = fopen("/tmp/cpp_queries_init.bin", "wb");
        if (fp) {
            fwrite(q_row.data(), sizeof(float), D * N_queries, fp);
            fclose(fp);
        }
        LDBG("  query 0 first 4: %.6f %.6f %.6f %.6f\n", q_row[0], q_row[1], q_row[2], q_row[3]);
    }

    // 5. Initialize reference points via enc_bbox_head MLP on ALL encoder tokens,
    //    then gather top-K, then add anchors, then sigmoid.
    //    HF: enc_outputs_coord_logits = enc_bbox_head(output_memory) + anchors
    //        reference_points = sigmoid(gather(enc_outputs_coord_logits, topk_ind))
    std::vector<float> ref_points(N_queries * 4);
    auto _t_eb = std::chrono::steady_clock::now();
    {
        // Run enc_bbox_head on ALL enc_output tokens (not just top-K queries)
        // enc_bbox_head is a 3-layer MLP: Linear(256,256) → ReLU → Linear(256,256) → ReLU → Linear(256,4)
        int N_total = total_tokens;
        std::vector<float> tmp(D * N_total);
        // enc_output is in [D, N_total] col-major layout = same as 'memory' before value_proj
        // but we need the raw enc_output, not the memory (which has been through layer norms)
        // enc_output is enc_proj[D, N_total] computed at step 1 (linear + layernorm)
        std::copy(enc_proj.begin(), enc_proj.end(), tmp.begin());

        for (int j = 0; j < 3; j++) {
            int out_d = (j < 2) ? D : 4;
            std::vector<float> out(out_d * N_total);
            cpu_linear(tmp.data(), out.data(), D, out_d, N_total, ctx->decoder.enc_bbox_w[j],
                       ctx->decoder.enc_bbox_b[j]);
            if (j < 2) {
                for (auto & v : out) v = std::max(0.0f, v); // ReLU
                tmp = out;
            } else {
                // Add anchors and gather top-K, then sigmoid
                // out is [4, N_total] col-major: out[d * N_total + token]
                for (int q = 0; q < N_queries; q++) {
                    int token_idx = token_scores[q].second;
                    for (int d = 0; d < 4; d++) {
                        float bbox_logit = out[d * N_total + token_idx];
                        float logit_anchor = anchors[token_idx * 4 + d];
                        ref_points[q * 4 + d] = 1.0f / (1.0f + expf(-(bbox_logit + logit_anchor)));
                    }
                }
            }
        }
    }
    const double _encbbox_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_eb).count();

    // Compute query position encoding from reference points via query_pos_head MLP
    // ref_points → linear0 → ReLU → linear1 → ReLU → linear2 → pos_enc  [D, N_queries]
    // Helper: compute query position encoding from reference points via MLP
    // HF: query_pos_head = 2-layer MLP: Linear(4, 2*D) → ReLU → Linear(2*D, D)
    int qpos_hidden = 2 * D; // 512
    auto compute_pos_enc = [&](const float * rp, float * pe) {
        std::vector<float> rp_col(4 * N_queries);
        for (int q = 0; q < N_queries; q++)
            for (int d = 0; d < 4; d++) rp_col[d * N_queries + q] = rp[q * 4 + d];
        std::vector<float> tmp(qpos_hidden * N_queries);
        cpu_linear(rp_col.data(), tmp.data(), 4, qpos_hidden, N_queries, ctx->decoder.qpos_w[0],
                   ctx->decoder.qpos_b[0]);
        for (auto & v : tmp) v = std::max(0.0f, v);
        cpu_linear(tmp.data(), pe, qpos_hidden, D, N_queries, ctx->decoder.qpos_w[1], ctx->decoder.qpos_b[1]);
    };

    std::vector<float> pos_enc(D * N_queries, 0.0f);

    // Debug: dump ref_points
    if (getenv("LAYOUT_DEBUG")) {
        FILE * fp = fopen("/tmp/cpp_ref_points.bin", "wb");
        if (fp) {
            fwrite(ref_points.data(), sizeof(float), N_queries * 4, fp);
            fclose(fp);
        }
        LDBG("  ref_points[0]: cx=%.6f cy=%.6f w=%.6f h=%.6f\n", ref_points[0], ref_points[1], ref_points[2],
             ref_points[3]);
    }

    // Compute initial pos_enc and dump for comparison
    compute_pos_enc(ref_points.data(), pos_enc.data());
    if (getenv("LAYOUT_DEBUG")) {
        std::vector<float> pe_row(D * N_queries);
        for (int q = 0; q < N_queries; q++)
            for (int d = 0; d < D; d++) pe_row[q * D + d] = pos_enc[d * N_queries + q];
        FILE * fp = fopen("/tmp/cpp_pos_enc_init.bin", "wb");
        if (fp) {
            fwrite(pe_row.data(), sizeof(float), D * N_queries, fp);
            fclose(fp);
        }
        LDBG("  pos_enc[0] first 4: %.6f %.6f %.6f %.6f\n", pe_row[0], pe_row[1], pe_row[2], pe_row[3]);
    }

    LDBG("layout_detect: query init done (top-K score: %.3f..%.3f)\n", token_scores[0].first,
         token_scores[N_queries - 1].first);
    LDBG("layout_detect: running decoder (6 layers, %d queries)...\n", N_queries);

    // Per-stage profiling accumulators for the bench line. The deform loop got
    // its timer first because it was once the dominant cost; after the
    // 2a43e4f4 cpu_linear threading it is ~1.5-2% of Phase 2, so the other
    // stages are timed too — do not optimize any of them without reading this
    // breakdown on the current build first.
    double _deform_ms = 0;
    double _sattn_ms = 0, _valproj_ms = 0, _ffn_ms = 0;
    // O2b (2026-08-06): all six value projections consume the SAME input —
    // `memory` is the encoder output and never changes across decoder layers —
    // so compute all six in ONE single-backend GPU graph (one upload of
    // memory, six mul_mats, one readback) instead of six threaded CPU GEMMs.
    // Measure-first: on current main value-proj was 178-211 ms of a
    // 545-588 ms Phase 2 at -t 4 (the #1 stage) and the AXPY+threaded
    // cpu_linear is already at its memory-bound ceiling, so the CPU lane has
    // no headroom left. The weights already live on ctx->backend (that is
    // where load_weights put them).
    //
    // OPT-IN (CRISPEMBED_LAYOUT_VALPROJ_GPU=1), NOT default: measured 3.5x on
    // the stage (178-211 -> 55 ms, Phase 2 -35%) but the GPU contraction
    // order is not byte-identical — on the 6-fixture sweep 3 pages were
    // byte-identical and the rest showed +-0.001 score / +-0.1 px jitter,
    // EXCEPT commons_test where a borderline region at score exactly 0.500
    // crossed the confidence threshold (13 -> 14 regions). Same drift class
    // that keeps LAYOUT_CONV_F16 gated. Candidate for CUDA boxes and for a
    // threshold-hysteresis follow-up; flip only with a region-level quality
    // gate across the full fixture set.
    std::vector<std::vector<float>> valproj_pre(6);
    bool valproj_gpu = false;
    {
        const char * vp_env = std::getenv("CRISPEMBED_LAYOUT_VALPROJ_GPU");
        const bool vp_enabled = vp_env && vp_env[0] && vp_env[0] != '0';
        if (vp_enabled && !ggml_backend_is_cpu(ctx->backend)) {
            auto _t_vp0 = std::chrono::steady_clock::now();
            const size_t gb = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false);
            std::vector<uint8_t> gbuf(gb);
            ggml_init_params gip = { gb, gbuf.data(), true };
            ggml_context * gc = ggml_init(gip);
            ggml_tensor * outs[6] = {};
            ggml_cgraph * gf = gc ? ggml_new_graph_custom(gc, 128, false) : nullptr;
            ggml_tensor * mem_in = nullptr;
            bool built = gf != nullptr;
            if (built) {
                // memory host layout is [D rows][N contiguous] = ggml [N, D].
                mem_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, total_tokens, D);
                ggml_set_input(mem_in);
                ggml_tensor * memT = ggml_cont(gc, ggml_transpose(gc, mem_in)); // [D, N]
                for (int li = 0; li < 6; li++) {
                    const auto & L = ctx->decoder.layers[li];
                    if (!L.cross_value_w || !L.cross_value_b) {
                        built = false;
                        break;
                    }
                    // cpu_linear's square-weight convention: flat[o + i*D] =
                    // W[o,i], i.e. the ggml fast axis indexes o. mul_mat
                    // contracts over ne0 of both operands, so transpose+cont
                    // the weight to put i on the fast axis.
                    ggml_tensor * wT = ggml_cont(gc, ggml_transpose(gc, L.cross_value_w));
                    ggml_tensor * y = ggml_mul_mat(gc, wT, memT); // [D_out, N]
                    y = ggml_add(gc, y, L.cross_value_b);         // bias broadcast over N
                    // Host wants values[o*N + n] (n fastest) = ggml [N, D_out].
                    y = ggml_cont(gc, ggml_transpose(gc, y));
                    ggml_set_output(y);
                    ggml_build_forward_expand(gf, y);
                    outs[li] = y;
                }
            }
            ggml_gallocr_t ga = nullptr;
            if (built) {
                ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
                built = ga && ggml_gallocr_alloc_graph(ga, gf);
            }
            if (built) {
                ggml_backend_tensor_set(mem_in, memory.data(), 0, (size_t)D * total_tokens * sizeof(float));
                built = ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS;
            }
            if (built) {
                for (int li = 0; li < 6; li++) {
                    valproj_pre[li].resize((size_t)D * total_tokens);
                    ggml_backend_tensor_get(outs[li], valproj_pre[li].data(), 0,
                                            valproj_pre[li].size() * sizeof(float));
                }
                valproj_gpu = true;
            }
            if (ga) ggml_gallocr_free(ga);
            if (gc) ggml_free(gc);
            _valproj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_vp0).count();
        }
    }
    for (int li = 0; li < 6; li++) {
        // Recompute pos_enc from current reference points each layer (matches HF)
        compute_pos_enc(ref_points.data(), pos_enc.data());
        const auto & layer = ctx->decoder.layers[li];

        // --- Self-attention via ggml graph (BLAS-accelerated) ---
        std::vector<float> residual = queries;
        auto _t_sa = std::chrono::steady_clock::now();
        {
            size_t sb = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false) + 16 * 1024 * 1024;
            std::vector<uint8_t> sbuf(sb);
            ggml_init_params sip = { sb, sbuf.data(), true };
            ggml_context * gc = ggml_init(sip);

            auto * inp = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, N_queries);
            ggml_set_name(inp, "sa_in");
            ggml_set_input(inp);
            auto * pe_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, N_queries);
            ggml_set_name(pe_in, "sa_pe");
            ggml_set_input(pe_in);
            auto * res_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, N_queries);
            ggml_set_name(res_in, "sa_res");
            ggml_set_input(res_in);

            // For decoder weights (pre-transposed by converter to Gemm convention):
            // cpu_linear uses W[i + o*in_d]. To match in ggml_mul_mat, we need a
            // weight tensor with ne[0]=in_d so it contracts over the input dimension.
            // The pre-transposed weight has ne=[768, 256] (ne[0]=768=out, ne[1]=256=in).
            // We need it as [256, 768] for ggml_mul_mat to contract over 256.
            // But the DATA must be reorganized, not just shape-swapped.
            // Solution: create a fresh [in_d, out_d] input tensor with transposed data.
            auto * qkv_w_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, 3 * D);
            ggml_set_name(qkv_w_in, "sa_qkv_w");
            ggml_set_input(qkv_w_in);
            auto * qkv_b_in = ggml_new_tensor_1d(gc, GGML_TYPE_F32, 3 * D);
            ggml_set_name(qkv_b_in, "sa_qkv_b");
            ggml_set_input(qkv_b_in);
            auto * out_w_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, D);
            ggml_set_name(out_w_in, "sa_out_w");
            ggml_set_input(out_w_in);
            auto * out_b_in = ggml_new_tensor_1d(gc, GGML_TYPE_F32, D);
            ggml_set_name(out_b_in, "sa_out_b");
            ggml_set_input(out_b_in);

            auto * out = decoder_self_attn(gc, inp, pe_in, res_in, qkv_w_in, qkv_b_in, out_w_in, out_b_in,
                                           layer.norm1_w, layer.norm1_b, D, N_queries, N_heads);

            ggml_set_name(out, "sa_out");
            ggml_set_output(out);
            ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_alloc_graph(ctx->galloc, gf);

            // Transpose CPU [D,N] (d*N+q) → ggml [D,N] (d+q*D) for activations
            std::vector<float> tq(D * N_queries), tp(D * N_queries), tr(D * N_queries);
            for (int d = 0; d < D; d++)
                for (int q = 0; q < N_queries; q++) {
                    tq[d + q * D] = queries[d * N_queries + q];
                    tp[d + q * D] = pos_enc[d * N_queries + q];
                    tr[d + q * D] = residual[d * N_queries + q];
                }
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_in"), tq.data(), 0, D * N_queries * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_pe"), tp.data(), 0, D * N_queries * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_res"), tr.data(), 0, D * N_queries * sizeof(float));

            // Transpose weight data from ggml [768, 256] to [256, 768] layout.
            // Original: flat_old[o + i*768] for o ∈ [0,768), i ∈ [0,256)
            // Target:   flat_new[i + o*256] = flat_old[o + i*768]
            // Cached in the layer after the first call — the weights are
            // immutable, so re-reading and re-transposing them per call was
            // pure repeated work (same bytes, byte-identical result).
            auto & lc = ctx->decoder.layers[li];
            if (lc.sa_qkv_w_t.empty()) {
                auto qkv_w_raw = tensor_to_f32(layer.self_qkv_w);
                lc.sa_qkv_w_t.resize((size_t)D * 3 * D);
                for (int i = 0; i < D; i++)
                    for (int o = 0; o < 3 * D; o++) lc.sa_qkv_w_t[i + o * D] = qkv_w_raw[o + i * 3 * D];
                lc.sa_qkv_b_v = tensor_to_f32(layer.self_qkv_b);
                // out_proj: ggml [256, 256] → same shape, but data needs transposing
                auto out_w_raw = tensor_to_f32(layer.self_out_w);
                lc.sa_out_w_t.resize((size_t)D * D);
                for (int i = 0; i < D; i++)
                    for (int o = 0; o < D; o++) lc.sa_out_w_t[i + o * D] = out_w_raw[o + i * D];
                lc.sa_out_b_v = tensor_to_f32(layer.self_out_b);
            }
            const auto & qkv_w_t = lc.sa_qkv_w_t;
            const auto & qkv_b_data = lc.sa_qkv_b_v;
            const auto & out_w_t = lc.sa_out_w_t;
            const auto & out_b_data = lc.sa_out_b_v;

            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_qkv_w"), qkv_w_t.data(), 0,
                                    D * 3 * D * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_qkv_b"), qkv_b_data.data(), 0, 3 * D * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_out_w"), out_w_t.data(), 0, D * D * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "sa_out_b"), out_b_data.data(), 0, D * sizeof(float));

            ggml_backend_graph_compute(ctx->backend, gf);

            // Transpose ggml → CPU layout
            std::vector<float> to(D * N_queries);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "sa_out"), to.data(), 0, D * N_queries * sizeof(float));
            for (int d = 0; d < D; d++)
                for (int q = 0; q < N_queries; q++) queries[d * N_queries + q] = to[d + q * D];

            ggml_free(gc);
        }
        _sattn_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_sa).count();
        if (layout_debug()) {
            float mn = 1e9, mx = -1e9;
            for (auto v : queries) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            fprintf(stderr, "  dec%d_after_sa: [%.4f,%.4f]\n", li, mn, mx);
        }

        // --- Deformable cross-attention ---
        residual = queries;

        // Value projection: [D, N_total] → [D, N_total] — precomputed for all
        // six layers in one GPU graph above when valproj_gpu is set.
        auto _t_vp = std::chrono::steady_clock::now();
        std::vector<float> values;
        if (valproj_gpu) {
            values = std::move(valproj_pre[li]);
        } else {
            values.resize((size_t)D * total_tokens);
            cpu_linear(memory.data(), values.data(), D, D, total_tokens, layer.cross_value_w, layer.cross_value_b);
        }
        _valproj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_vp).count();
        if (li == 0 && layout_debug()) {
            float vmin = 1e9, vmax = -1e9;
            for (auto v : values) {
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
            fprintf(stderr, "  dec0 values: [%.4f, %.4f] (Python: [-66.7, 76.0])\n", vmin, vmax);
        }

        // query + position encoding for cross-attention projections
        std::vector<float> qp(D * N_queries);
        for (int i = 0; i < D * N_queries; i++) qp[i] = queries[i] + pos_enc[i];

        // Sampling offsets: (query+pos) → linear → [192, N_queries]
        int n_offsets = N_heads * N_levels * N_points * 2; // 192
        std::vector<float> offsets(n_offsets * N_queries);
        cpu_linear(qp.data(), offsets.data(), D, n_offsets, N_queries, layer.cross_sampling_offsets_w,
                   layer.cross_sampling_offsets_b);

        // Attention weights: (query+pos) → linear → [96, N_queries]
        int n_weights = N_heads * N_levels * N_points; // 96
        std::vector<float> attn_weights(n_weights * N_queries);
        cpu_linear(qp.data(), attn_weights.data(), D, n_weights, N_queries, layer.cross_attn_weights_w,
                   layer.cross_attn_weights_b);

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

        auto _td0 = std::chrono::steady_clock::now();
        {
            deform_level lv_info[3];
            for (int lv = 0; lv < N_levels; lv++)
                lv_info[lv] = { feats[lv].W, feats[lv].H, level_starts[lv], N_points, 0.25f };
            deformable_sample(values.data(), total_tokens, lv_info, N_levels, N_heads, value_hd, N_queries,
                              ref_points.data(), offsets.data(), attn_weights.data(), 0.5f, false, cross_out.data());
        }
        _deform_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _td0).count();

        if (li == 0 && layout_debug()) {
            float mn = 1e9, mx = -1e9;
            for (auto v : cross_out) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            fprintf(stderr, "  dec0 cross_out (pre-proj): [%.4f, %.4f]\n", mn, mx);
            // query 0, first 8 dims: cross_out stored [D, N] col-major
            fprintf(stderr, "  dec0 cross_out[0,:8]:");
            for (int d = 0; d < 8; d++) fprintf(stderr, " %.6f", cross_out[d * N_queries + 0]);
            fprintf(stderr, "\n");
            float norm0 = 0;
            for (int d = 0; d < D; d++) {
                float v = cross_out[d * N_queries + 0];
                norm0 += v * v;
            }
            fprintf(stderr, "  dec0 cross_out[0] norm: %.4f\n", sqrtf(norm0));
            // Debug: print sampling stats for query 0
            float ref_cx0 = ref_points[0], ref_cy0 = ref_points[1];
            float ref_w0 = ref_points[2], ref_h0 = ref_points[3];
            fprintf(stderr, "  dec0 q0 ref: cx=%.4f cy=%.4f w=%.4f h=%.4f\n", ref_cx0, ref_cy0, ref_w0, ref_h0);
            float dx0 = offsets[0], dy0 = offsets[N_queries];
            fprintf(stderr, "  dec0 q0 offset[0]: dx=%.4f dy=%.4f → scaled dx=%.6f dy=%.6f\n", dx0, dy0,
                    dx0 * 0.25f * ref_w0 * 0.5f, dy0 * 0.25f * ref_h0 * 0.5f);
            float px0 = ref_cx0 + dx0 * 0.25f * ref_w0 * 0.5f;
            float py0 = ref_cy0 + dy0 * 0.25f * ref_h0 * 0.5f;
            fprintf(stderr, "  dec0 q0 sample pos: px=%.4f py=%.4f → pixel sx=%.1f sy=%.1f (80x80)\n", px0, py0,
                    px0 * 80 - 0.5f, py0 * 80 - 0.5f);
            // Print attention weights for q0
            float aw_sum = 0;
            for (int j = 0; j < N_heads * N_levels * N_points; j++) aw_sum += attn_weights[j * N_queries + 0];
            fprintf(stderr, "  dec0 q0 attn_weight sum: %.4f (expected ~%d from softmax)\n", aw_sum, N_heads);
        }

        // Debug: dump cross_out for parity comparison
        if (li == 0 && getenv("LAYOUT_DEBUG")) {
            std::vector<float> cross_row(D * N_queries);
            for (int d = 0; d < D; d++)
                for (int q = 0; q < N_queries; q++) cross_row[q * D + d] = cross_out[d * N_queries + q];
            FILE * fp = fopen("/tmp/cpp_cross_out.bin", "wb");
            if (fp) {
                fwrite(cross_row.data(), sizeof(float), D * N_queries, fp);
                fclose(fp);
            }

            // Compare with Python reference if available
            FILE * rfp = fopen("/tmp/py_cross_out.bin", "rb");
            if (rfp) {
                std::vector<float> py_ref(D * N_queries);
                fread(py_ref.data(), sizeof(float), D * N_queries, rfp);
                fclose(rfp);
                double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < D * N_queries; i++) {
                    dot += cross_row[i] * py_ref[i];
                    na += cross_row[i] * cross_row[i];
                    nb += py_ref[i] * py_ref[i];
                }
                double cos = dot / (sqrt(na) * sqrt(nb) + 1e-12);
                fprintf(stderr, "  dec0 cross_out vs Python: cos=%.6f\n", cos);
            }
        }

        // Cross-attention output projection
        std::vector<float> ca_out(D * N_queries);
        cpu_linear(cross_out.data(), ca_out.data(), D, D, N_queries, layer.cross_out_w, layer.cross_out_b);

        if (li == 0 && getenv("LAYOUT_DEBUG")) {
            float mn = 1e9, mx = -1e9;
            for (auto v : ca_out) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            fprintf(stderr, "  dec0 ca_out (post-proj): [%.4f, %.4f] (HF: [-2.97, 2.67])\n", mn, mx);
            // Dump post-projection cross_out for parity (matches Python dec_0_cross_out)
            std::vector<float> ca_row(D * N_queries);
            for (int d = 0; d < D; d++)
                for (int q = 0; q < N_queries; q++) ca_row[q * D + d] = ca_out[d * N_queries + q];
            FILE * fp2 = fopen("/tmp/cpp_cross_out.bin", "wb");
            if (fp2) {
                fwrite(ca_row.data(), sizeof(float), D * N_queries, fp2);
                fclose(fp2);
            }
        }

        // Residual + norm
        for (int i = 0; i < D * N_queries; i++) queries[i] = residual[i] + ca_out[i];
        cpu_layernorm(queries.data(), D, N_queries, layer.norm2_w, layer.norm2_b);

        if (layout_debug()) {
            float mn = 1e9, mx = -1e9;
            for (auto v : queries) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            fprintf(stderr, "  dec%d_norm2: [%.4f,%.4f]\n", li, mn, mx);
        }
        // --- FFN via ggml graph (BLAS-accelerated) ---
        residual = queries;
        auto _t_ffn = std::chrono::steady_clock::now();
        {
            size_t fb = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false) + 8 * 1024 * 1024;
            std::vector<uint8_t> fbuf(fb);
            ggml_init_params fip = { fb, fbuf.data(), true };
            ggml_context * gc = ggml_init(fip);

            // Input: [N, D] in ggml (ne[0]=D varies fastest = ggml col-major)
            auto * inp = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, N_queries);
            ggml_set_name(inp, "ffn_in");
            ggml_set_input(inp);
            auto * res = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, N_queries);
            ggml_set_name(res, "ffn_res");
            ggml_set_input(res);

            auto * out = decoder_ffn(gc, inp, res, layer.ffn1_w, layer.ffn1_b, layer.ffn2_w, layer.ffn2_b,
                                     layer.norm3_w, layer.norm3_b, false);
            ggml_set_name(out, "ffn_out");
            ggml_set_output(out);

            ggml_cgraph * gf = ggml_new_graph_custom(gc, 32, false);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_alloc_graph(ctx->galloc, gf);

            // Transpose CPU [D, N] (d*N+q) → ggml [D, N] (d+q*D)
            std::vector<float> tq(D * N_queries), tr(D * N_queries);
            for (int d = 0; d < D; d++)
                for (int q = 0; q < N_queries; q++) {
                    tq[d + q * D] = queries[d * N_queries + q];
                    tr[d + q * D] = residual[d * N_queries + q];
                }
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ffn_in"), tq.data(), 0, D * N_queries * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ffn_res"), tr.data(), 0, D * N_queries * sizeof(float));

            ggml_backend_graph_compute(ctx->backend, gf);

            // Transpose back
            std::vector<float> to(D * N_queries);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "ffn_out"), to.data(), 0, D * N_queries * sizeof(float));
            for (int d = 0; d < D; d++)
                for (int q = 0; q < N_queries; q++) queries[d * N_queries + q] = to[d + q * D];

            ggml_free(gc);
        }
        _ffn_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_ffn).count();

        if (layout_debug()) {
            float mn = 1e9, mx = -1e9;
            for (auto v : queries) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            fprintf(stderr, "  dec%d_norm3: [%.4f,%.4f]\n", li, mn, mx);
        }

        // Dump decoder layer output for diff comparison
        if (getenv("LAYOUT_DEBUG")) {
            char fname[128];
            snprintf(fname, sizeof(fname), "/tmp/cpp_dec_%d.bin", li);
            // Convert from [D, N] col-major to [N, D] row-major
            std::vector<float> q_row(D * N_queries);
            for (int q = 0; q < N_queries; q++)
                for (int d = 0; d < D; d++) q_row[q * D + d] = queries[d * N_queries + q];
            FILE * fp = fopen(fname, "wb");
            if (fp) {
                fwrite(q_row.data(), sizeof(float), D * N_queries, fp);
                fclose(fp);
            }
            // Also dump after sa_norm and ca_norm
            // (already printed ranges above — just dump the full data for the sub-steps)
        }

        // Update reference points via bbox head (iterative refinement)
        std::vector<float> bbox_delta(4 * N_queries);
        std::vector<float> tmp(D * N_queries);
        // MLP: 256 → 256 → 256 → 4
        std::copy(queries.begin(), queries.end(), tmp.begin());
        for (int j = 0; j < 3; j++) {
            std::vector<float> out((j < 2 ? D : 4) * N_queries);
            cpu_linear(tmp.data(), out.data(), (j == 0 ? D : D), (j < 2 ? D : 4), N_queries,
                       ctx->decoder.dec_bbox_w[li][j], ctx->decoder.dec_bbox_b[li][j]);
            if (j < 2) {
                for (auto & v : out) v = std::max(0.0f, v); // ReLU
                tmp = out;
            } else {
                bbox_delta = out;
            }
        }

        // Refine reference points: sigmoid(inverse_sigmoid(ref) + delta)
        for (int q = 0; q < N_queries; q++) {
            for (int d = 0; d < 4; d++) {
                float r = ref_points[q * 4 + d];
                // inverse sigmoid: log(r / (1 - r))
                r = std::max(1e-6f, std::min(r, 1.0f - 1e-6f));
                float inv_sig = logf(r / (1.0f - r));
                float refined = 1.0f / (1.0f + expf(-(inv_sig + bbox_delta[d * N_queries + q])));
                ref_points[q * 4 + d] = refined;
            }
        }

        LDBG("  decoder layer %d done\n", li);

        // Debug: dump decoder layer 0 output
        if (li == 0 && getenv("LAYOUT_DEBUG")) {
            float qmin = 1e9, qmax = -1e9;
            std::vector<float> q_row(D * N_queries);
            for (int q = 0; q < N_queries; q++)
                for (int d = 0; d < D; d++) {
                    float v = queries[d * N_queries + q];
                    q_row[q * D + d] = v;
                    qmin = std::min(qmin, v);
                    qmax = std::max(qmax, v);
                }
            LDBG("  dec0_output: range=[%.4f, %.4f]\n", qmin, qmax);
            fprintf(stderr, "  dec0_output[0,:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", q_row[0], q_row[1],
                    q_row[2], q_row[3], q_row[4], q_row[5], q_row[6], q_row[7]);
            FILE * fp = fopen("/tmp/cpp_dec0_out.bin", "wb");
            if (fp) {
                fwrite(q_row.data(), sizeof(float), D * N_queries, fp);
                fclose(fp);
            }
        }
    }

    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_phase2).count();
        fprintf(stderr,
                "[layout_detect-bench] Phase 2 decoder: %.1f ms (level-proj: %.1f, enc-bbox-head: %.1f, "
                "self-attn: %.1f, value-proj: %.1f, ffn: %.1f, deformable-sample loop: %.1f, other: %.1f)\n",
                ms, _levproj_ms, _encbbox_ms, _sattn_ms, _valproj_ms, _ffn_ms, _deform_ms,
                ms - _levproj_ms - _encbbox_ms - _sattn_ms - _valproj_ms - _ffn_ms - _deform_ms);
    }

    // --- Phase 3: Detection heads ---
    auto t_phase3 = std::chrono::steady_clock::now();
    // Class scores from final decoder output
    std::vector<float> class_scores(ctx->num_classes * N_queries);
    cpu_linear(queries.data(), class_scores.data(), D, ctx->num_classes, N_queries, ctx->decoder.dec_score_w,
               ctx->decoder.dec_score_b);
    // Sigmoid
    for (auto & v : class_scores) v = 1.0f / (1.0f + expf(-v));

    // Collect results
    std::vector<region> results;
    for (int q = 0; q < N_queries; q++) {
        // Find best class
        int best_cls = 0;
        float best_score = 0;
        for (int c = 0; c < ctx->num_classes; c++) {
            float s = class_scores[c * N_queries + q];
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
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
        r.raw_class = best_cls;
        r.raw_label = LABEL_NAMES[best_cls];

        results.push_back(r);
    }

    // Sort by score descending
    std::sort(results.begin(), results.end(), [](const region & a, const region & b) { return a.score > b.score; });

    // Debug: print top scores
    if (results.empty() && layout_debug()) {
        float max_score = 0;
        for (auto & v : class_scores) max_score = std::max(max_score, v);
        fprintf(stderr, "layout_detect: max class score after sigmoid: %.6f\n", max_score);
    }
    if (bench) {
        double ms3 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_phase3).count();
        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[layout_detect-bench] Phase 3 heads: %.1f ms\n", ms3);
        fprintf(stderr, "[layout_detect-bench] total: %.1f ms\n", total_ms);
    }
    if (layout_debug())
        fprintf(stderr, "layout_detect: %zu detections (score > %.2f)\n", results.size(), score_threshold);
    return results;
}

// RT-DETRv2 path of detect_file/detect_rgb on tightly packed RGB pixels.
static std::vector<region> heron_detect_rgb(context * ctx, const uint8_t * raw, int img_w, int img_h,
                                            float score_threshold) {
    // Resize to 640×640 with bilinear interpolation, normalize, CHW
    // Matches HF RTDetrImageProcessor (resample=PIL.BILINEAR)
    std::vector<float> pixels(3 * ctx->input_h * ctx->input_w, 0.0f);
    int H = ctx->input_h, W = ctx->input_w;

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < H; y++) {
            // PIL bilinear: src = (dst + 0.5) * src_size / dst_size - 0.5
            float src_y = ((float)y + 0.5f) * img_h / H - 0.5f;
            src_y = std::max(0.0f, std::min(src_y, (float)(img_h - 1)));
            int sy0 = (int)floorf(src_y);
            int sy1 = std::min(sy0 + 1, img_h - 1);
            float fy = src_y - sy0;
            for (int x = 0; x < W; x++) {
                float src_x = ((float)x + 0.5f) * img_w / W - 0.5f;
                src_x = std::max(0.0f, std::min(src_x, (float)(img_w - 1)));
                int sx0 = (int)floorf(src_x);
                int sx1 = std::min(sx0 + 1, img_w - 1);
                float fx = src_x - sx0;

                float v00 = raw[(sy0 * img_w + sx0) * 3 + c] / 255.0f;
                float v01 = raw[(sy0 * img_w + sx1) * 3 + c] / 255.0f;
                float v10 = raw[(sy1 * img_w + sx0) * 3 + c] / 255.0f;
                float v11 = raw[(sy1 * img_w + sx1) * 3 + c] / 255.0f;

                float val = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v01 + (1 - fx) * fy * v10 + fx * fy * v11;
                // RT-DETRv2 (docling-layout-heron) uses do_normalize=False:
                // pixels are just rescaled to [0,1] by rescale_factor=1/255,
                // NO ImageNet mean/std normalization.
                pixels[c * H * W + y * W + x] = val;
            }
        }
    }
    float scale_x = (float)W / img_w;
    float scale_y = (float)H / img_h;

    auto results = detect(ctx, pixels.data(), img_h, img_w, score_threshold);

    // Scale coordinates back to original image
    for (auto & r : results) {
        r.x1 /= scale_x;
        r.x2 /= scale_x;
        r.y1 /= scale_y;
        r.y2 /= scale_y;
    }

    return results;
}

std::vector<region> detect_file(context * ctx, const char * path, float score_threshold) {
    if (!ctx || !path) return {};

    int img_w, img_h, img_c;
    stbi_uc * raw = stbi_load(path, &img_w, &img_h, &img_c, 3);
    if (!raw) {
        fprintf(stderr, "layout_detect: cannot load image: %s\n", path);
        return {};
    }
    if (score_threshold < 0) score_threshold = default_score_threshold(ctx);
    auto results = ctx->arch == arch_id::dfine ? detect_rgb(ctx, raw, img_w, img_h, 3, score_threshold)
                                               : heron_detect_rgb(ctx, raw, img_w, img_h, score_threshold);
    stbi_image_free(raw);
    return results;
}

// ===========================================================================
// D-FINE (OXR-1.0 layout detector): HGNetv2-B2 + hybrid encoder + D-FINE decoder
// ===========================================================================
//
// Blueprint: SoMarkAI/OXR @ c1a1dcb3 services/layout_analyze/dfine_la (an
// inference-only adaptation of Peterande/D-FINE), weights SoMarkAI/OXR-1.0 @
// c708407f layout_analysis/last.pth, converted by
// models/convert-oxr-dfine-to-gguf.py. Every stage below names the module of
// that blueprint it reproduces; the parity tool tests/oxr_dfine_diff.cpp
// compares the stages captured here against tools/oxr_dfine_reference.py.

static const char * const OXR_LABELS[OXR_NUM_CLASSES] = {
    "Caption", "Footnote", "Formula", "Text",       "Page-footer", "Page-header",
    "Picture", "Title",    "Table",   "Code Block", "Stamp",       "Chemical Structure"
};
// OXR class -> canonical class. Stamp and Chemical Structure have no
// canonical equivalent; they map to picture with canonical_exact=false.
static const struct {
    label_id label;
    bool exact;
} OXR_TO_CANONICAL[OXR_NUM_CLASSES] = {
    { label_id::caption, true }, { label_id::footnote, true },    { label_id::formula, true },
    { label_id::text, true },    { label_id::page_footer, true }, { label_id::page_header, true },
    { label_id::picture, true }, { label_id::title, true },       { label_id::table, true },
    { label_id::code, true },    { label_id::picture, false },    { label_id::picture, false },
};

const char * oxr_label_name(int raw_class) {
    return raw_class >= 0 && raw_class < OXR_NUM_CLASSES ? OXR_LABELS[raw_class] : nullptr;
}

label_id oxr_canonical_label(int raw_class, bool * exact) {
    if (raw_class < 0 || raw_class >= OXR_NUM_CLASSES) {
        if (exact) *exact = false;
        return label_id::NUM_CLASSES;
    }
    if (exact) *exact = OXR_TO_CANONICAL[raw_class].exact;
    return OXR_TO_CANONICAL[raw_class].label;
}

namespace {

struct dfine_conv {
    ggml_tensor * w = nullptr;     // [KW, KH, IC(/groups), OC]
    ggml_tensor * scale = nullptr; // BN scale (unfused) or nullptr
    ggml_tensor * shift = nullptr; // BN shift (unfused) or fused conv bias
    bool lab = false;
    float lab_scale = 1.0f, lab_bias = 0.0f;
};
struct dfine_linear {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};
struct dfine_hg_layer {
    dfine_conv conv, conv1, conv2; // conv: ConvBNAct; conv1/conv2: LightConvBNAct
};
struct dfine_hg_block {
    std::vector<dfine_hg_layer> layers;
    dfine_conv agg0, agg1;
};
struct dfine_hg_stage {
    bool downsample = false, light = false;
    int kernel = 3;
    dfine_conv down;
    std::vector<dfine_hg_block> blocks;
};
struct dfine_csp {
    dfine_conv conv1, conv2, bottlenecks[2];
};
struct dfine_elan {
    dfine_conv cv1, cv4, cv2_conv, cv3_conv;
    dfine_csp cv2_csp, cv3_csp;
};
struct dfine_dec_layer {
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    dfine_linear out_proj, offsets, attn_w, gate, ffn1, ffn2, norm1, gate_norm, norm3, bbox[3];
};

} // namespace

struct dfine_model {
    dfine_conv stem[5];
    dfine_hg_stage stages[4];
    dfine_conv input_proj[3];
    aifi_w aifi{};
    ggml_tensor * pos_embed = nullptr;
    dfine_conv lateral[2], scd_cv1[2], scd_cv2[2];
    dfine_elan fpn[2], pan[2];
    dfine_linear enc_proj, enc_norm, enc_score, enc_bbox[3], pre_bbox[3], qpos[2], score_head;
    ggml_tensor * valid_mask = nullptr;
    dfine_dec_layer layers[4];
    // Host-side tables and the tiny LQE head (evaluated on the CPU).
    std::vector<float> anchors, project, nps, lqe_w0, lqe_b0, lqe_w1, lqe_b1;
    float reg_scale = 4.0f;
    int image = 960, pad_value = 114, num_queries = 300, num_classes = 12, num_layers = 4, reg_max = 32;
    int hidden = 256, heads = 8, total_tokens = 18900;
    int num_points[3] = { 3, 6, 3 };
    bool gpu = false;
};

namespace {

bool dfine_trace() {
    static int v = -1;
    if (v < 0) v = core_env::on("CRISPEMBED_OXR_LAYOUT_TRACE") ? 1 : 0;
    return v == 1;
}

void capture_tensor(context * ctx, const std::string & name, ggml_tensor * t) {
    if (!ctx->capture || !t) return;
    auto & st = (*ctx->capture)[name];
    st.shape.assign(t->ne, t->ne + ggml_n_dims(t));
    st.data.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, st.data.data(), 0, st.data.size() * sizeof(float));
}

void capture_host(context * ctx, const std::string & name, const float * data, std::vector<int64_t> shape) {
    if (!ctx->capture) return;
    auto & st = (*ctx->capture)[name];
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    st.shape = std::move(shape);
    st.data.assign(data, data + n);
}

ggml_tensor * f32(ggml_context * g, ggml_tensor * t) {
    return t && t->type != GGML_TYPE_F32 ? ggml_cast(g, t, GGML_TYPE_F32) : t;
}

enum class dact { none, relu, silu };

// ConvBNAct / ConvNormLayer_fuse / VGGBlock (deploy) / input_proj.
ggml_tensor * dfine_conv_fwd(ggml_context * g, ggml_tensor * x, const dfine_conv & c, int stride, int pad,
                             bool depthwise, dact act) {
    ggml_tensor * w = f32(g, c.w);
    // CRISPEMBED_OXR_LAYOUT_DW_DIRECT=1 (opt-in, A/B): direct depthwise kernel
    // instead of im2col + mul_mat. Default stays the im2col path.
    static const bool dw_direct = core_env::on("CRISPEMBED_OXR_LAYOUT_DW_DIRECT");
    ggml_tensor * y = !depthwise  ? conv2d_dispatch(g, w, x, stride, pad)
                      : dw_direct ? ggml_conv_2d_dw_direct(g, w, x, stride, stride, pad, pad, 1, 1)
                                  : ggml_conv_2d_dw(g, w, x, stride, stride, pad, pad, 1, 1);
    const int oc = (int)c.w->ne[3];
    if (c.scale) y = ggml_mul(g, y, ggml_reshape_3d(g, f32(g, c.scale), 1, 1, oc));
    if (c.shift) y = ggml_add(g, y, ggml_reshape_3d(g, f32(g, c.shift), 1, 1, oc));
    if (act == dact::relu) y = ggml_relu(g, y);
    if (act == dact::silu) y = ggml_silu(g, y);
    if (c.lab) y = ggml_scale_bias(g, y, c.lab_scale, c.lab_bias);
    return y;
}

ggml_tensor * mlp(ggml_context * g, ggml_tensor * x, const dfine_linear * layers, int n) {
    for (int i = 0; i < n; i++) {
        x = linear(g, x, layers[i].w, layers[i].b);
        if (i + 1 < n) x = ggml_relu(g, x);
    }
    return x;
}

// HGNetv2 backbone (backbone.py): returns stage outputs 1..3 (stride 8/16/32).
void hgnetv2_forward(context * ctx, ggml_context * g, const dfine_model & m, ggml_tensor * input, ggml_tensor ** f) {
    auto named = [&](ggml_tensor * t, const char * name) {
        ggml_set_name(t, name);
        ggml_set_output(t);
        return t;
    };
    // StemBlock
    auto * x = dfine_conv_fwd(g, input, m.stem[0], 2, 1, false, dact::relu);
    x = ggml_pad(g, x, 1, 1, 0, 0);
    auto * x2 = dfine_conv_fwd(g, x, m.stem[1], 1, 0, false, dact::relu);
    x2 = ggml_pad(g, x2, 1, 1, 0, 0);
    x2 = dfine_conv_fwd(g, x2, m.stem[2], 1, 0, false, dact::relu);
    auto * x1 = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0, 0);
    x = ggml_concat(g, x1, x2, 2);
    x = dfine_conv_fwd(g, x, m.stem[3], 2, 1, false, dact::relu);
    x = named(dfine_conv_fwd(g, x, m.stem[4], 1, 0, false, dact::relu), "bb.stem");
    for (int s = 0; s < 4; s++) {
        const auto & st = m.stages[s];
        if (st.downsample) x = dfine_conv_fwd(g, x, st.down, 2, 1, true, dact::none);
        for (size_t b = 0; b < st.blocks.size(); b++) {
            const auto & blk = st.blocks[b];
            auto * identity = x;
            auto * cat = x;
            auto * y = x;
            for (const auto & layer : blk.layers) {
                if (st.light) {
                    y = dfine_conv_fwd(g, y, layer.conv1, 1, 0, false, dact::none);
                    y = dfine_conv_fwd(g, y, layer.conv2, 1, (st.kernel - 1) / 2, true, dact::relu);
                } else {
                    y = dfine_conv_fwd(g, y, layer.conv, 1, (st.kernel - 1) / 2, false, dact::relu);
                }
                cat = ggml_concat(g, cat, y, 2);
            }
            y = dfine_conv_fwd(g, cat, blk.agg0, 1, 0, false, dact::relu);
            y = dfine_conv_fwd(g, y, blk.agg1, 1, 0, false, dact::relu);
            x = b > 0 ? ggml_add(g, y, identity) : y;
        }
        char name[32];
        snprintf(name, sizeof(name), "bb.stage%d", s);
        named(x, name);
        if (s >= 1) f[s - 1] = x;
    }
    (void)ctx;
}

// CSPLayer with VGGBlock bottlenecks (encoder.py), expansion 1, conv3 = Identity.
ggml_tensor * csp_forward(ggml_context * g, ggml_tensor * x, const dfine_csp & c) {
    auto * a = dfine_conv_fwd(g, x, c.conv1, 1, 0, false, dact::silu);
    for (const auto & b : c.bottlenecks) a = dfine_conv_fwd(g, a, b, 1, 1, false, dact::silu);
    auto * b = dfine_conv_fwd(g, x, c.conv2, 1, 0, false, dact::silu);
    return ggml_add(g, a, b);
}

// RepNCSPELAN4: split cv1 output; cv2/cv3 chain on the second half; concat all.
ggml_tensor * elan_forward(ggml_context * g, ggml_tensor * x, const dfine_elan & e) {
    auto * y = dfine_conv_fwd(g, x, e.cv1, 1, 0, false, dact::silu);
    const int64_t half = y->ne[2] / 2;
    auto * y1 = ggml_cont(g, ggml_view_3d(g, y, y->ne[0], y->ne[1], half, y->nb[1], y->nb[2], half * y->nb[2]));
    auto * a = csp_forward(g, y1, e.cv2_csp);
    a = dfine_conv_fwd(g, a, e.cv2_conv, 1, 1, false, dact::silu);
    auto * b = csp_forward(g, a, e.cv3_csp);
    b = dfine_conv_fwd(g, b, e.cv3_conv, 1, 1, false, dact::silu);
    auto * cat = ggml_concat(g, ggml_concat(g, y, a, 2), b, 2);
    return dfine_conv_fwd(g, cat, e.cv4, 1, 0, false, dact::silu);
}

// [W, H, C] -> [C, W*H] token-major (torch flatten(2).permute(0, 2, 1)).
ggml_tensor * tokens(ggml_context * g, ggml_tensor * x) {
    auto * p = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));
    return ggml_reshape_2d(g, p, x->ne[2], x->ne[0] * x->ne[1]);
}

struct graph_scope {
    std::vector<uint8_t> buf;
    ggml_context * g = nullptr;
    ggml_cgraph * gf = nullptr;
    explicit graph_scope(int nodes) {
        buf.resize(ggml_tensor_overhead() * (nodes + 64) + ggml_graph_overhead_custom(nodes, false));
        ggml_init_params ip = { buf.size(), buf.data(), true };
        g = ggml_init(ip);
        gf = ggml_new_graph_custom(g, nodes, false);
    }
    ~graph_scope() {
        if (g) ggml_free(g);
    }
    ggml_tensor * input(const char * name, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        auto * t = ggml_new_tensor_3d(g, GGML_TYPE_F32, n0, n1, n2);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    }
    void output(ggml_tensor * t, const char * name) {
        ggml_set_name(t, name);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
};

bool run_graph(context * ctx, graph_scope & gs, const std::vector<std::pair<const char *, const float *>> & inputs) {
    if (!ggml_gallocr_alloc_graph(ctx->galloc, gs.gf)) {
        ctx->error = "D-FINE graph allocation failed";
        return false;
    }
    for (const auto & in : inputs) {
        ggml_tensor * t = ggml_graph_get_tensor(gs.gf, in.first);
        if (!t) continue; // input not used by this graph variant (e.g. layer 0)
        ggml_backend_tensor_set(t, in.second, 0, ggml_nbytes(t));
    }
    if (ggml_backend_graph_compute(ctx->backend, gs.gf) != GGML_STATUS_SUCCESS) {
        ctx->error = "D-FINE graph compute failed";
        return false;
    }
    return true;
}

std::vector<float> read(ggml_cgraph * gf, const char * name) {
    ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    std::vector<float> v(t ? ggml_nelements(t) : 0);
    if (t) ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// utils.inverse_sigmoid(eps=1e-5)
float inverse_sigmoidf(float x) {
    x = std::min(std::max(x, 0.0f), 1.0f);
    return std::log(std::max(x, 1e-5f) / std::max(1.0f - x, 1e-5f));
}

// Transpose [rows][cols] -> [cols][rows].
std::vector<float> transpose(const std::vector<float> & a, int rows, int cols) {
    std::vector<float> t(a.size());
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) t[(size_t)c * rows + r] = a[(size_t)r * cols + c];
    return t;
}

} // namespace

static bool dfine_load(context * ctx, const char * path, gguf_context * meta) {
    auto * m = new dfine_model();
    ctx->dfine = m;
    ctx->arch = arch_id::dfine;
    if (core_gguf::kv_u32(meta, "layout.input_height", 0) != 960 ||
        core_gguf::kv_u32(meta, "layout.num_classes", 0) != 12 ||
        core_gguf::kv_u32(meta, "layout.num_queries", 0) != 300 ||
        core_gguf::kv_u32(meta, "layout.num_decoder_layers", 0) != 4 ||
        core_gguf::kv_u32(meta, "layout.reg_max", 0) != 32 || core_gguf::kv_u32(meta, "layout.hidden_dim", 0) != 256 ||
        core_gguf::kv_i32_array(meta, "layout.num_points") != std::vector<int>({ 3, 6, 3 })) {
        fprintf(stderr, "layout_detect: unsupported D-FINE hyperparameters in %s\n", path);
        return false;
    }
    const auto labels = core_gguf::kv_str_array(meta, "layout.labels");
    if ((int)labels.size() != OXR_NUM_CLASSES) return false;
    for (int i = 0; i < OXR_NUM_CLASSES; i++)
        if (labels[i] != OXR_LABELS[i]) {
            fprintf(stderr, "layout_detect: D-FINE label %d is '%s', expected '%s'\n", i, labels[i].c_str(),
                    OXR_LABELS[i]);
            return false;
        }
    ctx->num_queries = 300;
    ctx->num_classes = 12;
    ctx->input_h = ctx->input_w = 960;

    // Backend: CPU unless explicitly opted into the GPU path (unqualified).
    const bool force_cpu = (getenv("LAYOUT_DETECT_FORCE_CPU") && atoi(getenv("LAYOUT_DETECT_FORCE_CPU")));
    m->gpu = core_env::on("CRISPEMBED_OXR_LAYOUT_GPU") && !force_cpu;
    ctx->backend = m->gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
        m->gpu = false;
    }
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (!core_gguf::load_weights(path, ctx->backend, "layout", ctx->wl)) {
        fprintf(stderr, "layout_detect: failed to load D-FINE weights\n");
        return false;
    }
    std::vector<std::string> missing;
    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = ctx->wl.tensors.find(n);
        if (it == ctx->wl.tensors.end()) {
            missing.push_back(n);
            return nullptr;
        }
        return it->second;
    };
    auto host = [&](const std::string & n) { return tensor_to_f32(get(n)); };
    auto lab = [&](dfine_conv & c, const std::string & p) {
        auto s = host(p + ".lab.scale"), b = host(p + ".lab.bias");
        if (s.size() == 1 && b.size() == 1) {
            c.lab = true;
            c.lab_scale = s[0];
            c.lab_bias = b[0];
        }
    };
    auto cba = [&](dfine_conv & c, const std::string & p, bool with_lab) {
        c.w = get(p + ".conv.weight");
        c.scale = get(p + ".bn.scale");
        c.shift = get(p + ".bn.shift");
        if (with_lab) lab(c, p);
    };
    auto fused = [&](dfine_conv & c, const std::string & p) {
        c.w = get(p + ".fused.weight");
        c.shift = get(p + ".fused.bias");
    };
    auto lin = [&](dfine_linear & l, const std::string & p) {
        l.w = get(p + ".weight");
        l.b = get(p + ".bias");
    };
    const char * stems[5] = { "stem1", "stem2a", "stem2b", "stem3", "stem4" };
    for (int i = 0; i < 5; i++) cba(m->stem[i], std::string("backbone.stem.") + stems[i], true);
    // HGNetv2-B2 stage config: blocks, downsample, light, kernel, layers.
    const int cfg[4][5] = { { 1, 0, 0, 3, 4 }, { 1, 1, 0, 3, 4 }, { 3, 1, 1, 5, 4 }, { 1, 1, 1, 5, 4 } };
    for (int s = 0; s < 4; s++) {
        auto & st = m->stages[s];
        const std::string p = "backbone.stages." + std::to_string(s);
        st.downsample = cfg[s][1] != 0;
        st.light = cfg[s][2] != 0;
        st.kernel = cfg[s][3];
        if (st.downsample) cba(st.down, p + ".downsample", false);
        st.blocks.resize(cfg[s][0]);
        for (int b = 0; b < cfg[s][0]; b++) {
            auto & blk = st.blocks[b];
            const std::string q = p + ".blocks." + std::to_string(b);
            blk.layers.resize(cfg[s][4]);
            for (int i = 0; i < cfg[s][4]; i++) {
                const std::string l = q + ".layers." + std::to_string(i);
                if (st.light) {
                    cba(blk.layers[i].conv1, l + ".conv1", false);
                    cba(blk.layers[i].conv2, l + ".conv2", true);
                } else {
                    cba(blk.layers[i].conv, l, true);
                }
            }
            cba(blk.agg0, q + ".aggregation.0", true);
            cba(blk.agg1, q + ".aggregation.1", true);
        }
    }
    for (int i = 0; i < 3; i++) {
        const std::string p = "encoder.input_proj." + std::to_string(i);
        m->input_proj[i].w = get(p + ".conv.weight");
        m->input_proj[i].scale = get(p + ".norm.scale");
        m->input_proj[i].shift = get(p + ".norm.shift");
    }
    const std::string a = "encoder.encoder.0.layers.0";
    m->aifi = { get(a + ".self_attn.in_proj_weight"),
                get(a + ".self_attn.in_proj_bias"),
                get(a + ".self_attn.out_proj.weight"),
                get(a + ".self_attn.out_proj.bias"),
                get(a + ".norm1.weight"),
                get(a + ".norm1.bias"),
                get(a + ".linear1.weight"),
                get(a + ".linear1.bias"),
                get(a + ".linear2.weight"),
                get(a + ".linear2.bias"),
                get(a + ".norm2.weight"),
                get(a + ".norm2.bias") };
    m->pos_embed = get("encoder.pos_embed2");
    auto elan = [&](dfine_elan & e, const std::string & p) {
        fused(e.cv1, p + ".cv1");
        fused(e.cv4, p + ".cv4");
        for (int k = 0; k < 2; k++) {
            const std::string cv = p + (k == 0 ? ".cv2" : ".cv3");
            auto & csp = k == 0 ? e.cv2_csp : e.cv3_csp;
            fused(csp.conv1, cv + ".0.conv1");
            fused(csp.conv2, cv + ".0.conv2");
            for (int j = 0; j < 2; j++) fused(csp.bottlenecks[j], cv + ".0.bottlenecks." + std::to_string(j));
            fused(k == 0 ? e.cv2_conv : e.cv3_conv, cv + ".1");
        }
    };
    for (int i = 0; i < 2; i++) {
        const std::string n = std::to_string(i);
        fused(m->lateral[i], "encoder.lateral_convs." + n);
        fused(m->scd_cv1[i], "encoder.downsample_convs." + n + ".0.cv1");
        fused(m->scd_cv2[i], "encoder.downsample_convs." + n + ".0.cv2");
        elan(m->fpn[i], "encoder.fpn_blocks." + n);
        elan(m->pan[i], "encoder.pan_blocks." + n);
    }
    lin(m->enc_proj, "decoder.enc_output.proj");
    lin(m->enc_norm, "decoder.enc_output.norm");
    lin(m->enc_score, "decoder.enc_score_head");
    for (int i = 0; i < 3; i++) {
        lin(m->enc_bbox[i], "decoder.enc_bbox_head.layers." + std::to_string(i));
        lin(m->pre_bbox[i], "decoder.pre_bbox_head.layers." + std::to_string(i));
    }
    for (int i = 0; i < 2; i++) lin(m->qpos[i], "decoder.query_pos_head.layers." + std::to_string(i));
    lin(m->score_head, "decoder.dec_score_head.3");
    m->valid_mask = get("decoder.valid_mask");
    for (int li = 0; li < 4; li++) {
        auto & L = m->layers[li];
        const std::string p = "decoder.decoder.layers." + std::to_string(li);
        L.qkv_w = get(p + ".self_attn.in_proj_weight");
        L.qkv_b = get(p + ".self_attn.in_proj_bias");
        lin(L.out_proj, p + ".self_attn.out_proj");
        lin(L.offsets, p + ".cross_attn.sampling_offsets");
        lin(L.attn_w, p + ".cross_attn.attention_weights");
        lin(L.gate, p + ".gateway.gate");
        lin(L.gate_norm, p + ".gateway.norm");
        lin(L.ffn1, p + ".linear1");
        lin(L.ffn2, p + ".linear2");
        lin(L.norm1, p + ".norm1");
        lin(L.norm3, p + ".norm3");
        for (int j = 0; j < 3; j++)
            lin(L.bbox[j], "decoder.dec_bbox_head." + std::to_string(li) + ".layers." + std::to_string(j));
        auto nps = host(p + ".cross_attn.num_points_scale");
        if (li == 0)
            m->nps = nps;
        else if (nps != m->nps)
            missing.push_back(p + ".cross_attn.num_points_scale (differs)");
    }
    m->anchors = host("decoder.anchors");
    m->project = host("decoder.project");
    const auto reg = host("decoder.reg_scale");
    m->reg_scale = reg.size() == 1 ? std::fabs(reg[0]) : 0.0f;
    const std::string lqe = "decoder.decoder.lqe_layers.3.reg_conf.layers.";
    m->lqe_w0 = host(lqe + "0.weight");
    m->lqe_b0 = host(lqe + "0.bias");
    m->lqe_w1 = host(lqe + "1.weight");
    m->lqe_b1 = host(lqe + "1.bias");
    if (!missing.empty() || m->anchors.size() != 18900 * 4 || m->project.size() != 33 || m->nps.size() != 12 ||
        m->lqe_w0.size() != 64 * 20 || m->lqe_w1.size() != 64 || m->reg_scale <= 0) {
        fprintf(stderr, "layout_detect: D-FINE GGUF is incomplete (%zu missing, first: %s)\n", missing.size(),
                missing.empty() ? "-" : missing[0].c_str());
        return false;
    }
    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    LDBG("layout_detect: D-FINE loaded %zu tensors on %s\n", ctx->wl.tensors.size(), m->gpu ? "GPU" : "CPU");
    return ctx->galloc != nullptr;
}

// Service preprocessing (dfine_la/preprocess.py): RGB uint8 -> OpenCV
// Lanczos4 letterbox into a 960x960 canvas filled with 114 -> CHW / 255.
static std::vector<float> dfine_preprocess(const dfine_model & m, const uint8_t * pixels, int width, int height,
                                           int channels) {
    const int S = m.image;
    std::vector<uint8_t> rgb;
    const uint8_t * src = pixels;
    if (channels != 3) {
        rgb.resize((size_t)width * height * 3);
        for (size_t i = 0; i < (size_t)width * height; i++)
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = pixels[i * channels + (channels == 1 ? 0 : c)];
        src = rgb.data();
    }
    // Python: ratio = target / max(w, h) (double); int() truncation.
    const double ratio = (double)S / (double)std::max(width, height);
    const int new_w = (int)(width * ratio), new_h = (int)(height * ratio);
    const int pad_left = (S - new_w) / 2, pad_top = (S - new_h) / 2;
    std::vector<uint8_t> resized((size_t)new_w * new_h * 3);
    core_cv::opencv_resize_lanczos4_u8(src, width, height, 3, resized.data(), new_w, new_h);
    std::vector<float> chw((size_t)3 * S * S);
    const float fill = (float)m.pad_value / 255.0f;
    std::fill(chw.begin(), chw.end(), fill);
    for (int y = 0; y < new_h; y++)
        for (int x = 0; x < new_w; x++)
            for (int c = 0; c < 3; c++)
                chw[(size_t)c * S * S + (size_t)(y + pad_top) * S + (x + pad_left)] =
                    (float)resized[((size_t)y * new_w + x) * 3 + c] / 255.0f;
    return chw;
}

static bool dfine_forward(context * ctx, const float * chw, int orig_w, int orig_h, std::vector<candidate> & out) {
    const dfine_model & m = *ctx->dfine;
    const int S = m.image, D = m.hidden, NQ = m.num_queries, NC = m.num_classes, NT = m.total_tokens;
    const int R = m.reg_max + 1;
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    };
    const auto t_enc = clock::now();
    capture_host(ctx, "input", chw, { S, S, 3 });

    // ---- Phase 1: backbone + hybrid encoder + encoder head (one graph) ----
    std::vector<float> values, om, logits;
    int level_w[3], level_h[3];
    {
        graph_scope gs(16384);
        ggml_context * g = gs.g;
        auto * input = gs.input("input", S, S, 3);
        ggml_tensor * f[3];
        hgnetv2_forward(ctx, g, m, input, f);
        ggml_tensor * p[3];
        for (int i = 0; i < 3; i++) {
            p[i] = dfine_conv_fwd(g, f[i], m.input_proj[i], 1, 0, false, dact::none);
            char name[16];
            snprintf(name, sizeof(name), "enc.proj%d", i);
            ggml_set_name(p[i], name);
            ggml_set_output(p[i]);
        }
        // AIFI on the stride-32 level.
        const int w5 = (int)p[2]->ne[0], h5 = (int)p[2]->ne[1];
        auto * aifi = aifi_encoder_layer(g, tokens(g, p[2]), m.pos_embed, m.aifi, m.heads);
        ggml_set_name(aifi, "enc.aifi");
        ggml_set_output(aifi);
        auto * p5 = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, aifi, D, w5, h5), 2, 0, 1, 3));
        ccfm_ops ops;
        auto named = [&](ggml_tensor * t, const char * base, int i) {
            char name[16];
            snprintf(name, sizeof(name), "enc.%s%d", base, i);
            ggml_set_name(t, name);
            ggml_set_output(t);
            return t;
        };
        ops.lateral = [&](int i, ggml_tensor * x) {
            return named(dfine_conv_fwd(g, x, m.lateral[i], 1, 0, false, dact::none), "lat", i);
        };
        ops.fpn = [&](int i, ggml_tensor * x) { return named(elan_forward(g, x, m.fpn[i]), "fpn", i); };
        ops.down = [&](int i, ggml_tensor * x) {
            auto * y = dfine_conv_fwd(g, x, m.scd_cv1[i], 1, 0, false, dact::none);
            return named(dfine_conv_fwd(g, y, m.scd_cv2[i], 2, 1, true, dact::none), "down", i);
        };
        ops.pan = [&](int i, ggml_tensor * x) { return named(elan_forward(g, x, m.pan[i]), "pan", i); };
        ggml_tensor * s[3];
        ccfm_forward(g, p[0], p[1], p5, ops, &s[0], &s[1], &s[2]);
        for (int i = 0; i < 3; i++) {
            level_w[i] = (int)s[i]->ne[0];
            level_h[i] = (int)s[i]->ne[1];
        }
        // DFINETransformer._get_encoder_input / _get_decoder_input
        auto * memory = ggml_concat(g, ggml_concat(g, tokens(g, s[0]), tokens(g, s[1]), 1), tokens(g, s[2]), 1);
        gs.output(memory, "dec.memory");
        gs.output(ggml_cont(g, ggml_transpose(g, memory)), "values");
        auto * masked = ggml_mul(g, memory, ggml_reshape_2d(g, f32(g, m.valid_mask), 1, NT));
        auto * output_memory = layer_norm(g, linear(g, masked, m.enc_proj.w, m.enc_proj.b), m.enc_norm.w, m.enc_norm.b);
        gs.output(output_memory, "dec.output_memory");
        gs.output(linear(g, output_memory, m.enc_score.w, m.enc_score.b), "dec.enc_logits");
        if (!run_graph(ctx, gs, { { "input", chw } })) return false;
        values = read(gs.gf, "values");
        om = read(gs.gf, "dec.output_memory");
        logits = read(gs.gf, "dec.enc_logits");
        if (ctx->capture) {
            const char * names[] = { "bb.stem",   "bb.stage0", "bb.stage1",  "bb.stage2",         "bb.stage3",
                                     "enc.proj0", "enc.proj1", "enc.proj2",  "enc.aifi",          "enc.lat0",
                                     "enc.lat1",  "enc.fpn0",  "enc.fpn1",   "enc.down0",         "enc.down1",
                                     "enc.pan0",  "enc.pan1",  "dec.memory", "dec.output_memory", "dec.enc_logits" };
            for (const char * n : names) capture_tensor(ctx, n, ggml_graph_get_tensor(gs.gf, n));
        }
    }
    if (level_w[0] * level_h[0] + level_w[1] * level_h[1] + level_w[2] * level_h[2] != NT) {
        ctx->error = "unexpected D-FINE level sizes";
        return false;
    }
    ctx->last.encoder_ms = ms_since(t_enc);
    const auto t_dec = clock::now();

    // ---- Query selection (_select_topk, query_select_method="default") ----
    std::vector<int> order(NT);
    std::vector<float> best(NT);
    for (int t = 0; t < NT; t++) {
        float v = logits[(size_t)t * NC];
        for (int c = 1; c < NC; c++) v = std::max(v, logits[(size_t)t * NC + c]);
        best[t] = v;
        order[t] = t;
    }
    std::partial_sort(order.begin(), order.begin() + NQ, order.end(),
                      [&](int a, int b) { return best[a] > best[b] || (best[a] == best[b] && a < b); });
    order.resize(NQ);
    std::vector<float> content((size_t)D * NQ), topk_anchor((size_t)4 * NQ), topk_index(NQ);
    for (int q = 0; q < NQ; q++) {
        std::copy_n(om.data() + (size_t)order[q] * D, D, content.data() + (size_t)q * D);
        std::copy_n(m.anchors.data() + (size_t)order[q] * 4, 4, topk_anchor.data() + (size_t)q * 4);
        topk_index[q] = (float)order[q];
    }
    capture_host(ctx, "dec.topk_index", topk_index.data(), { NQ });
    capture_host(ctx, "dec.content", content.data(), { D, NQ });
    capture_host(ctx, "dec.topk_anchor", topk_anchor.data(), { 4, NQ });
    std::vector<float> ref_unact;
    {
        graph_scope gs(64);
        auto * c = gs.input("content", D, NQ);
        auto * delta = mlp(gs.g, c, m.enc_bbox, 3);
        gs.output(delta, "dec.enc_bbox_delta");
        if (!run_graph(ctx, gs, { { "content", content.data() } })) return false;
        auto d = read(gs.gf, "dec.enc_bbox_delta");
        capture_host(ctx, "dec.enc_bbox_delta", d.data(), { 4, NQ });
        ref_unact.resize(d.size());
        for (size_t i = 0; i < d.size(); i++) ref_unact[i] = d[i] + topk_anchor[i];
    }
    capture_host(ctx, "dec.ref_unact", ref_unact.data(), { 4, NQ });

    // ---- TransformerDecoder.forward (deploy: all 4 layers, eval_idx = 3) ----
    std::vector<float> target = content, ref(ref_unact.size()), ref_initial(ref_unact.size());
    for (size_t i = 0; i < ref.size(); i++) ref[i] = sigmoidf(ref_unact[i]);
    std::vector<float> prev_out((size_t)D * NQ, 0.0f), prev_corners((size_t)4 * R * NQ, 0.0f), final_logits;
    std::vector<float> boxes((size_t)4 * NQ);
    deform_level levels[3];
    {
        int start = 0, point = 0;
        for (int lv = 0; lv < 3; lv++) {
            levels[lv] = { level_w[lv], level_h[lv], start, m.num_points[lv], m.nps[point] };
            start += level_w[lv] * level_h[lv];
            point += m.num_points[lv];
        }
    }
    const int total_points = m.num_points[0] + m.num_points[1] + m.num_points[2];
    const int head_dim = D / m.heads;
    for (int li = 0; li < m.num_layers; li++) {
        const auto & L = m.layers[li];
        const std::string pre = "dec." + std::to_string(li) + ".";
        // Graph A: query position head, self-attention, sampling offsets and weights.
        std::vector<float> self_out, offsets, weights;
        {
            graph_scope gs(256);
            auto * tgt = gs.input("tgt", D, NQ);
            auto * rp = gs.input("ref", 4, NQ);
            // ggml_clamp is an in-place view; materialize it so the output buffer is not reused.
            auto * qpos = ggml_cont(gs.g, ggml_clamp(gs.g, mlp(gs.g, rp, m.qpos, 2), -10.0f, 10.0f));
            gs.output(qpos, "qpos");
            auto * self = decoder_self_attn(gs.g, tgt, qpos, tgt, f32(gs.g, L.qkv_w), L.qkv_b, f32(gs.g, L.out_proj.w),
                                            L.out_proj.b, L.norm1.w, L.norm1.b, D, NQ, m.heads);
            gs.output(self, "self");
            auto * qp = ggml_add(gs.g, self, qpos);
            gs.output(ggml_cont(gs.g, ggml_transpose(gs.g, linear(gs.g, qp, L.offsets.w, L.offsets.b))), "offsets");
            auto * aw = linear(gs.g, qp, L.attn_w.w, L.attn_w.b);
            aw = ggml_soft_max(gs.g, ggml_reshape_2d(gs.g, aw, total_points, m.heads * NQ));
            aw = ggml_reshape_2d(gs.g, aw, total_points * m.heads, NQ);
            gs.output(ggml_cont(gs.g, ggml_transpose(gs.g, aw)), "weights");
            if (!run_graph(ctx, gs, { { "tgt", target.data() }, { "ref", ref.data() } })) return false;
            self_out = read(gs.gf, "self");
            offsets = read(gs.gf, "offsets");
            weights = read(gs.gf, "weights");
            if (ctx->capture) {
                capture_tensor(ctx, pre + "qpos", ggml_graph_get_tensor(gs.gf, "qpos"));
                capture_host(ctx, pre + "self", self_out.data(), { D, NQ });
            }
        }
        // Deformable cross-attention (MSDeformableAttention, no value/output projection).
        std::vector<float> cross_cm((size_t)D * NQ, 0.0f);
        deformable_sample(values.data(), NT, levels, 3, m.heads, head_dim, NQ, ref.data(), offsets.data(),
                          weights.data(), 0.5f, true, cross_cm.data());
        const auto cross = transpose(cross_cm, D, NQ);
        capture_host(ctx, pre + "cross", cross.data(), { D, NQ });
        // Graph B: gate, FFN, box/score heads.
        std::vector<float> layer_out, corners, pre_delta;
        {
            graph_scope gs(256);
            auto * self = gs.input("self", D, NQ);
            auto * cr = gs.input("cross", D, NQ);
            auto * po = gs.input("prev_out", D, NQ);
            auto * pc = gs.input("prev_corners", 4 * R, NQ);
            auto * gates = ggml_sigmoid(gs.g, linear(gs.g, ggml_concat(gs.g, self, cr, 0), L.gate.w, L.gate.b));
            auto * g1 = ggml_view_2d(gs.g, gates, D, NQ, gates->nb[1], 0);
            auto * g2 = ggml_view_2d(gs.g, gates, D, NQ, gates->nb[1], (size_t)D * sizeof(float));
            auto * x = layer_norm(gs.g, ggml_add(gs.g, ggml_mul(gs.g, g1, self), ggml_mul(gs.g, g2, cr)), L.gate_norm.w,
                                  L.gate_norm.b);
            gs.output(x, "gate");
            auto * o = decoder_ffn(gs.g, x, x, L.ffn1.w, L.ffn1.b, L.ffn2.w, L.ffn2.b, L.norm3.w, L.norm3.b, true);
            gs.output(o, "out");
            auto * cin = li == 0 ? o : ggml_add(gs.g, o, po);
            auto * cor = mlp(gs.g, cin, L.bbox, 3);
            if (li > 0) cor = ggml_add(gs.g, cor, pc);
            gs.output(cor, "corners");
            if (li == 0) gs.output(mlp(gs.g, o, m.pre_bbox, 3), "pre");
            if (li == m.num_layers - 1) gs.output(linear(gs.g, o, m.score_head.w, m.score_head.b), "logits");
            if (!run_graph(ctx, gs,
                           { { "self", self_out.data() },
                             { "cross", cross.data() },
                             { "prev_out", prev_out.data() },
                             { "prev_corners", prev_corners.data() } }))
                return false;
            layer_out = read(gs.gf, "out");
            corners = read(gs.gf, "corners");
            if (li == 0) pre_delta = read(gs.gf, "pre");
            if (li == m.num_layers - 1) final_logits = read(gs.gf, "logits");
            if (ctx->capture) {
                capture_tensor(ctx, pre + "gate", ggml_graph_get_tensor(gs.gf, "gate"));
                capture_host(ctx, pre + "out", layer_out.data(), { D, NQ });
                capture_host(ctx, pre + "corners", corners.data(), { 4 * R, NQ });
            }
        }
        if (li == 0) {
            capture_host(ctx, "dec.pre_bbox_delta", pre_delta.data(), { 4, NQ });
            for (size_t i = 0; i < ref.size(); i++) ref_initial[i] = sigmoidf(pre_delta[i] + inverse_sigmoidf(ref[i]));
            capture_host(ctx, "dec.ref_initial", ref_initial.data(), { 4, NQ });
        }
        // Integral + distance2bbox (FDR box decoding).
        const float rs = m.reg_scale;
        for (int q = 0; q < NQ; q++) {
            float dist[4];
            for (int side = 0; side < 4; side++) {
                const float * c = corners.data() + (size_t)q * 4 * R + side * R;
                float mx = c[0];
                for (int j = 1; j < R; j++) mx = std::max(mx, c[j]);
                float e[64], sum = 0;
                for (int j = 0; j < R; j++) sum += (e[j] = std::exp(c[j] - mx));
                float acc = 0;
                for (int j = 0; j < R; j++) acc += (e[j] / sum) * m.project[j];
                dist[side] = acc;
            }
            const float * p = ref_initial.data() + (size_t)q * 4;
            const float half = 0.5f * rs;
            const float x1 = p[0] - (half + dist[0]) * (p[2] / rs);
            const float y1 = p[1] - (half + dist[1]) * (p[3] / rs);
            const float x2 = p[0] + (half + dist[2]) * (p[2] / rs);
            const float y2 = p[1] + (half + dist[3]) * (p[3] / rs);
            float * b = boxes.data() + (size_t)q * 4;
            b[0] = (x1 + x2) / 2;
            b[1] = (y1 + y2) / 2;
            b[2] = x2 - x1;
            b[3] = y2 - y1;
        }
        capture_host(ctx, pre + "bbox", boxes.data(), { 4, NQ });
        ref = boxes;
        prev_out = layer_out;
        prev_corners = corners;
        target = layer_out;
    }
    // LQE on the last layer: scores + MLP(top-4 bin probabilities and their mean).
    for (int q = 0; q < NQ; q++) {
        float stat[20];
        for (int side = 0; side < 4; side++) {
            const float * c = prev_corners.data() + (size_t)q * 4 * R + side * R;
            float mx = c[0];
            for (int j = 1; j < R; j++) mx = std::max(mx, c[j]);
            float prob[64], sum = 0;
            for (int j = 0; j < R; j++) sum += (prob[j] = std::exp(c[j] - mx));
            for (int j = 0; j < R; j++) prob[j] /= sum;
            std::partial_sort(prob, prob + 4, prob + R, std::greater<float>());
            float mean = (prob[0] + prob[1] + prob[2] + prob[3]) / 4;
            for (int k = 0; k < 4; k++) stat[side * 5 + k] = prob[k];
            stat[side * 5 + 4] = mean;
        }
        float quality = m.lqe_b1[0];
        for (int o = 0; o < 64; o++) {
            float h = m.lqe_b0[o];
            for (int i = 0; i < 20; i++) h += m.lqe_w0[(size_t)o * 20 + i] * stat[i];
            quality += m.lqe_w1[o] * std::max(h, 0.0f);
        }
        for (int c = 0; c < NC; c++) final_logits[(size_t)q * NC + c] += quality;
    }
    capture_host(ctx, "dec.logits", final_logits.data(), { NC, NQ });
    ctx->last.decoder_ms = ms_since(t_dec);
    const auto t_post = clock::now();

    // ---- DFINEPostProcessor (focal: top-k over sigmoid(query, class)) ----
    const int NF = NQ * NC;
    std::vector<float> score(NF);
    for (int i = 0; i < NF; i++) score[i] = sigmoidf(final_logits[i]);
    std::vector<int> idx(NF);
    for (int i = 0; i < NF; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + NQ, idx.end(),
                      [&](int a, int b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
    // postprocess_boxes(letterbox=True) in float32, torch op by op.
    const float ow = (float)orig_w, oh = (float)orig_h;
    const float max_dim = std::max(ow, oh);
    const float scale = (1.0f / max_dim) * (float)S; // int / tensor == reciprocal(tensor) * int
    const float new_w = std::floor(ow * scale), new_h = std::floor(oh * scale);
    const float pad_left = std::floor(((float)S - new_w) / 2), pad_top = std::floor(((float)S - new_h) / 2);
    out.clear();
    out.reserve(NQ);
    std::vector<float> post_labels(NQ), post_boxes((size_t)4 * NQ), post_scores(NQ);
    for (int k = 0; k < NQ; k++) {
        const int i = idx[k], q = i / NC;
        const float * b = boxes.data() + (size_t)q * 4;
        float xy[4] = { (b[0] - 0.5f * b[2]) * (float)S, (b[1] - 0.5f * b[3]) * (float)S,
                        (b[0] + 0.5f * b[2]) * (float)S, (b[1] + 0.5f * b[3]) * (float)S };
        candidate c;
        c.raw_class = i % NC;
        c.score = score[i];
        for (int j = 0; j < 4; j++) {
            float v = std::min(std::max(xy[j], 0.0f), (float)(S - 1));
            v = (v - ((j & 1) ? pad_top : pad_left)) / scale;
            c.box[j] = v;
            post_boxes[(size_t)k * 4 + j] = v;
        }
        post_labels[k] = (float)c.raw_class;
        post_scores[k] = c.score;
        out.push_back(c);
    }
    capture_host(ctx, "post.labels", post_labels.data(), { NQ });
    capture_host(ctx, "post.boxes", post_boxes.data(), { 4, NQ });
    capture_host(ctx, "post.scores", post_scores.data(), { NQ });
    ctx->last.postprocess_ms = ms_since(t_post);
    if (dfine_trace())
        fprintf(stderr, "[oxr-layout] encoder %.1f ms, decoder %.1f ms, post %.1f ms\n", ctx->last.encoder_ms,
                ctx->last.decoder_ms, ctx->last.postprocess_ms);
    return true;
}

static bool dfine_require(context * ctx) {
    if (!ctx || ctx->arch != arch_id::dfine || !ctx->dfine) {
        if (ctx) ctx->error = "not a D-FINE layout context";
        return false;
    }
    if (!core_env::on("CRISPEMBED_OXR") || !core_env::on("CRISPEMBED_ACCEPT_OXR_LICENSE")) {
        ctx->error = "OXR layout requires CRISPEMBED_OXR=1 and CRISPEMBED_ACCEPT_OXR_LICENSE=1";
        return false;
    }
    return true;
}

bool detect_candidates_preprocessed(context * ctx, const float * chw, int orig_width, int orig_height,
                                    std::vector<candidate> & out) {
    out.clear();
    if (!dfine_require(ctx)) return false;
    ctx->error.clear();
    if (!chw || orig_width <= 0 || orig_height <= 0) {
        ctx->error = "invalid preprocessed input";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ctx->last = timings{};
    const bool ok = dfine_forward(ctx, chw, orig_width, orig_height, out);
    ctx->last.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

bool detect_candidates(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                       std::vector<candidate> & out) {
    out.clear();
    if (!dfine_require(ctx)) return false;
    ctx->error.clear();
    if (!pixels || width <= 0 || height <= 0 || (uint64_t)width * height > 200000000ULL ||
        (channels != 1 && channels != 3 && channels != 4)) {
        ctx->error = "expected 8-bit gray/RGB/RGBA pixels with valid dimensions";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ctx->last = timings{};
    const auto chw = dfine_preprocess(*ctx->dfine, pixels, width, height, channels);
    const double pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = dfine_forward(ctx, chw.data(), width, height, out);
    ctx->last.preprocess_ms = pre_ms;
    ctx->last.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

std::vector<region> oxr_select(const std::vector<candidate> & candidates, int width, int height, float score_threshold,
                               float nms_iou_threshold) {
    // Python floats (doubles) throughout, as parse_layout_batch receives JSON.
    struct kept_t {
        int pos;
        int cls;
        double box[4];
        double score;
    };
    std::vector<kept_t> cand;
    const double thr = score_threshold;
    for (size_t i = 0; i < candidates.size(); i++) {
        const auto & c = candidates[i];
        if (c.raw_class < 0 || c.raw_class >= OXR_NUM_CLASSES) continue;
        const double score = c.score;
        if (!(score >= thr)) continue;
        double b[4] = { c.box[0], c.box[1], c.box[2], c.box[3] };
        if (b[0] >= b[2] || b[1] >= b[3]) continue; // upstream raises; never kept
        b[0] = std::min(std::max(b[0], 0.0), (double)width);
        b[1] = std::min(std::max(b[1], 0.0), (double)height);
        b[2] = std::min(std::max(b[2], 0.0), (double)width);
        b[3] = std::min(std::max(b[3], 0.0), (double)height);
        if (b[0] >= b[2] || b[1] >= b[3]) continue;
        cand.push_back({ (int)i, c.raw_class, { b[0], b[1], b[2], b[3] }, score });
    }
    std::vector<uint8_t> keep(cand.size(), 1);
    if (nms_iou_threshold >= 0) {
        const double nms = nms_iou_threshold;
        std::vector<int> order(cand.size());
        for (size_t i = 0; i < cand.size(); i++) order[i] = (int)i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return cand[a].score > cand[b].score || (cand[a].score == cand[b].score && cand[a].pos < cand[b].pos);
        });
        auto iou = [](const double * l, const double * r) {
            const double iw = std::max(0.0, std::min(l[2], r[2]) - std::max(l[0], r[0]));
            const double ih = std::max(0.0, std::min(l[3], r[3]) - std::max(l[1], r[1]));
            const double inter = iw * ih;
            if (inter == 0.0) return 0.0;
            const double la = (l[2] - l[0]) * (l[3] - l[1]), ra = (r[2] - r[0]) * (r[3] - r[1]);
            return inter / (la + ra - inter);
        };
        std::fill(keep.begin(), keep.end(), 0);
        std::vector<int> kept;
        for (int i : order) {
            bool drop = false;
            for (int k : kept)
                if (cand[k].cls == cand[i].cls && iou(cand[i].box, cand[k].box) > nms) {
                    drop = true;
                    break;
                }
            if (drop) continue;
            kept.push_back(i);
            keep[i] = 1;
        }
    }
    std::vector<region> out;
    for (size_t i = 0; i < cand.size(); i++) {
        if (!keep[i]) continue;
        region r;
        r.x1 = (float)cand[i].box[0];
        r.y1 = (float)cand[i].box[1];
        r.x2 = (float)cand[i].box[2];
        r.y2 = (float)cand[i].box[3];
        r.score = (float)cand[i].score;
        r.raw_class = cand[i].cls;
        r.raw_label = OXR_LABELS[cand[i].cls];
        r.label = oxr_canonical_label(cand[i].cls, &r.canonical_exact);
        r.label_name = LABEL_NAMES[(int)r.label];
        out.push_back(r);
    }
    return out;
}

const char * last_error(const context * ctx) {
    return ctx ? ctx->error.c_str() : "null layout context";
}

timings last_timings(const context * ctx) {
    return ctx ? ctx->last : timings{};
}

void set_stage_capture(context * ctx, stage_map * capture) {
    if (ctx) ctx->capture = capture;
}

arch_id arch(const context * ctx) {
    return ctx ? ctx->arch : arch_id::rt_detr_v2;
}

int num_raw_classes(const context * ctx) {
    return ctx && ctx->arch == arch_id::dfine ? OXR_NUM_CLASSES : (int)label_id::NUM_CLASSES;
}

const char * raw_label_name(const context * ctx, int raw_class) {
    if (ctx && ctx->arch == arch_id::dfine) return oxr_label_name(raw_class);
    return raw_class >= 0 && raw_class < (int)label_id::NUM_CLASSES ? LABEL_NAMES[raw_class] : nullptr;
}

float default_score_threshold(const context * ctx) {
    return ctx && ctx->arch == arch_id::dfine ? 0.5f : 0.3f;
}

std::vector<region> detect_rgb(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                               float score_threshold) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return {};
    if (score_threshold < 0) score_threshold = default_score_threshold(ctx);
    if (ctx->arch == arch_id::dfine) {
        std::vector<candidate> cands;
        if (!detect_candidates(ctx, pixels, width, height, channels, cands)) {
            fprintf(stderr, "layout_detect: %s\n", ctx->error.c_str());
            return {};
        }
        return oxr_select(cands, width, height, score_threshold, 0.6f);
    }
    if (channels != 3) return {};
    return heron_detect_rgb(ctx, pixels, width, height, score_threshold);
}

void free(context * ctx) {
    if (!ctx) return;
    delete ctx->dfine;
    if (ctx->galloc) ggml_gallocr_free(ctx->galloc);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

} // namespace layout_detect
