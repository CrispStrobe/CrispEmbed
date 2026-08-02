// tests/test_tps_locnet.cpp — test TPS localization network loading & inference
//
// Creates a tiny synthetic GGUF model (random weights) and verifies:
// 1. GGUF load succeeds
// 2. Forward pass produces valid control points
// 3. Full auto-dewarp pipeline runs without crash
// 4. Graceful failure on missing/bad model files

#include "tps_warp.h"
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) {
        printf("  %s[PASS]%s %s\n", GREEN, RESET, name);
        n_pass++;
    } else {
        printf("  %s[FAIL]%s %s\n", RED, RESET, name);
        n_fail++;
    }
}

// ---------------------------------------------------------------------------
// Create a tiny synthetic TPS localization GGUF model
// ---------------------------------------------------------------------------
// Architecture: 4 conv layers + 2 FC layers, "small" variant
// Conv channels: 3→16→32→64→128, FC: 128→64→N*2
// All weights random (model won't produce meaningful dewarping, but
// exercises the full code path).

// `flatten_convs` reproduces what tools/quantize.cpp emits: it collapses every
// 4-D F32 conv weight to 2-D [IC*KH*KW, OC] in the output header (a pure
// reinterpret — same bytes, same order, only ne changes). Any engine that reads
// an output-channel count off ne[3] silently sees 1 on such an artifact. The TPS
// converter defaults to F32 (--fp16 is opt-in), so a quantized tps-loc GGUF is
// exactly this shape.
static std::string create_test_gguf(int num_fiducial = 10, bool flatten_convs = false) {
    std::string path = flatten_convs ? "/tmp/test_tps_locnet_flat.gguf" : "/tmp/test_tps_locnet.gguf";

    // Fixed seed: the 4-D and flattened builds must contain byte-identical
    // weights, otherwise the layout comparison below is meaningless.
    srand(1234);

    gguf_context * gctx = gguf_init_empty();

    // Set metadata
    gguf_set_val_u32(gctx, "tps.num_fiducial", num_fiducial);
    gguf_set_val_u32(gctx, "tps.fc_dim", 64);
    gguf_set_val_u32(gctx, "tps.num_conv", 4);

    // Create a ggml context for tensor descriptors
    // We need space for 12 tensors (4 conv * {w,b} + 2 fc * {w,b})
    struct ggml_init_params params = {
        /* .mem_size   = */ 512 * 1024, // 512 KB for tensor metadata + data
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    int ic_list[] = { 3, 16, 32, 64 };
    int oc_list[] = { 16, 32, 64, 128 };

    // Conv layers: weights are [oc, ic, 3, 3] stored as ggml [kw=3, kh=3, ic, oc]
    for (int i = 0; i < 4; i++) {
        int ic = ic_list[i], oc = oc_list[i];

        std::string wn = "loc.conv" + std::to_string(i) + ".weight";
        ggml_tensor * w = flatten_convs ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)ic * 3 * 3, oc)
                                        : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, ic, oc);
        ggml_set_name(w, wn.c_str());
        // Fill with small random values
        float * wd = (float *)w->data;
        int n = ggml_nelements(w);
        float scale = 1.0f / sqrtf((float)(ic * 9));
        for (int j = 0; j < n; j++) wd[j] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 2.0f * scale;
        gguf_add_tensor(gctx, w);

        std::string bn = "loc.conv" + std::to_string(i) + ".bias";
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, oc);
        ggml_set_name(b, bn.c_str());
        memset(b->data, 0, oc * sizeof(float));
        gguf_add_tensor(gctx, b);
    }

    // FC1: [ic=128, oc=64] stored as ggml [ic=128, oc=64]
    {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
        ggml_set_name(w, "loc.fc1.weight");
        float * wd = (float *)w->data;
        float scale = 1.0f / sqrtf(128.0f);
        for (int j = 0; j < 128 * 64; j++) wd[j] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 2.0f * scale;
        gguf_add_tensor(gctx, w);

        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_set_name(b, "loc.fc1.bias");
        memset(b->data, 0, 64 * sizeof(float));
        gguf_add_tensor(gctx, b);
    }

    // FC2: [ic=64, oc=num_fiducial*2]
    {
        int oc = num_fiducial * 2;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, oc);
        ggml_set_name(w, "loc.fc2.weight");
        // Small non-zero weights so the predicted points actually depend on the
        // conv stack. With fc2 zeroed the output is fc2.bias alone, which makes
        // every downstream assertion blind to anything the convolutions do —
        // including a wrong channel count. Kept small so the points stay near
        // the initial grid and the existing finiteness checks still hold.
        {
            float * wd = (float *)w->data;
            float scale = 0.1f / sqrtf(64.0f);
            for (int j = 0; j < 64 * oc; j++) wd[j] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 2.0f * scale;
        }
        gguf_add_tensor(gctx, w);

        // Bias = initial fiducial grid (PaddleOCR convention)
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, oc);
        ggml_set_name(b, "loc.fc2.bias");
        float * bd = (float *)b->data;
        int half = num_fiducial / 2;
        for (int i = 0; i < half; i++) {
            float t = -1.0f + 2.0f * i / (half - 1);
            // Top row point i
            bd[i * 2 + 0] = t;     // x in [-1, 1]
            bd[i * 2 + 1] = -1.0f; // y = -1 (top)
            // Bottom row point i
            bd[(half + i) * 2 + 0] = t;
            bd[(half + i) * 2 + 1] = 1.0f; // y = 1 (bottom)
        }
        gguf_add_tensor(gctx, b);
    }

    gguf_write_to_file(gctx, path.c_str(), false);
    gguf_free(gctx);
    ggml_free(ctx);

    return path;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_load_and_predict() {
    printf("\n=== Load GGUF and predict control points ===\n");

    std::string path = create_test_gguf(10);
    printf("  Created test GGUF: %s\n", path.c_str());

    tps_locnet * net = tps_locnet_load(path.c_str());
    check("tps_locnet_load succeeds", net != nullptr);

    if (net) {
        check("num_fiducial = 10", tps_locnet_num_fiducial(net) == 10);

        // Create a simple test image
        const int W = 64, H = 32;
        std::vector<uint8_t> gray(W * H, 128);
        // Add some structure
        for (int y = 10; y < 22; y++)
            for (int x = 10; x < 54; x++) gray[y * W + x] = 40;

        std::vector<float> px(10), py(10);
        int n = tps_locnet_predict(net, gray.data(), W, H, px.data(), py.data());
        check("predict returns 10 points", n == 10);

        if (n > 0) {
            // Points should be in image pixel space
            bool all_valid = true;
            for (int i = 0; i < n; i++) {
                if (std::isnan(px[i]) || std::isnan(py[i]) || std::isinf(px[i]) || std::isinf(py[i])) {
                    all_valid = false;
                    break;
                }
            }
            check("all predicted points are finite", all_valid);

            printf("  Predicted points:\n");
            for (int i = 0; i < n && i < 4; i++) printf("    [%d] (%.1f, %.1f)\n", i, px[i], py[i]);
            if (n > 4) printf("    ... (%d more)\n", n - 4);
        }

        tps_locnet_free(net);
    }

    remove(path.c_str());
}

// A quantized tps-loc GGUF carries its conv weights flattened to 2-D (see
// create_test_gguf). The same bytes in a different ne must load to the same
// network and predict the same points — the layout is a header detail, not a
// model change. Comparing against the 4-D build rather than against a golden
// constant means this stays valid if the synthetic weights ever change.
static void test_flattened_conv_layout() {
    printf("\n=== Quantizer-flattened conv layout (2-D [IC*KH*KW, OC]) ===\n");

    std::string p4 = create_test_gguf(10, false);
    std::string p2 = create_test_gguf(10, true);

    tps_locnet * n4 = tps_locnet_load(p4.c_str());
    tps_locnet * n2 = tps_locnet_load(p2.c_str());
    check("4-D model loads", n4 != nullptr);
    check("flattened model loads", n2 != nullptr);

    if (n4 && n2) {
        check("flattened model reports the same num_fiducial",
              tps_locnet_num_fiducial(n2) == tps_locnet_num_fiducial(n4));

        const int W = 64, H = 32;
        std::vector<uint8_t> gray(W * H, 128);
        for (int y = 10; y < 22; y++)
            for (int x = 10; x < 54; x++) gray[y * W + x] = 40;

        std::vector<float> x4(10), y4(10), x2(10), y2(10);
        int r4 = tps_locnet_predict(n4, gray.data(), W, H, x4.data(), y4.data());
        int r2 = tps_locnet_predict(n2, gray.data(), W, H, x2.data(), y2.data());
        check("flattened model predicts the same point count", r2 == r4 && r2 > 0);

        if (r4 > 0 && r2 == r4) {
            float worst = 0.0f;
            for (int i = 0; i < r4; i++) {
                worst = std::max(worst, std::fabs(x2[i] - x4[i]));
                worst = std::max(worst, std::fabs(y2[i] - y4[i]));
            }
            printf("  worst |flattened - 4D| over %d control points: %.6f px\n", r4, worst);
            check("flattened layout predicts identical control points", worst == 0.0f);
        }
    }

    if (n4) tps_locnet_free(n4);
    if (n2) tps_locnet_free(n2);
    remove(p4.c_str());
    remove(p2.c_str());
}

static void test_auto_dewarp() {
    printf("\n=== Auto dewarp pipeline ===\n");

    std::string path = create_test_gguf(10);

    const int W = 64, H = 32;
    std::vector<uint8_t> gray(W * H, 180);
    std::vector<uint8_t> out(W * H, 0);

    int ret = tps_auto_dewarp(gray.data(), W, H, path.c_str(), out.data());
    check("tps_auto_dewarp returns 0", ret == 0);

    // Output should not be all zeros (warp wrote something)
    int n_zero = 0;
    for (int i = 0; i < W * H; i++)
        if (out[i] == 0) n_zero++;
    float zero_frac = (float)n_zero / (W * H);
    check("output not all zeros", zero_frac < 0.9f);

    remove(path.c_str());
}

static void test_bad_model() {
    printf("\n=== Error handling ===\n");

    tps_locnet * net = tps_locnet_load("/nonexistent/path.gguf");
    check("load returns NULL for missing file", net == nullptr);

    net = tps_locnet_load(nullptr);
    check("load returns NULL for null path", net == nullptr);

    int ret = tps_auto_dewarp(nullptr, 0, 0, nullptr, nullptr);
    check("auto_dewarp returns 1 for null inputs", ret == 1);

    tps_locnet_free(nullptr);
    check("tps_locnet_free(NULL) does not crash", true);
}

static int crispembed_test_main() {
    printf("TPS Localization Network — tests\n");

    srand(42); // deterministic random weights

    test_load_and_predict();
    test_flattened_conv_layout();
    test_auto_dewarp();
    test_bad_model();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
