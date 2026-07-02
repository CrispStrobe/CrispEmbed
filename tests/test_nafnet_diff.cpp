// test_nafnet_diff.cpp — parity test for NAFNet denoising.
//
// Usage: test-nafnet-diff <model.gguf> <ref.gguf>
//
// Mirrors test_restormer_diff.cpp: loads a PyTorch reference GGUF (input/output
// tensors, CHW float, 64x64 from tools/dump_nafnet_reference.py), runs the C++
// nafnet forward, and compares the final output. NAFNet is a same-size denoiser
// (no upscale), and nafnet_process writes into a caller-allocated buffer.

#include "nafnet_denoise.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-nafnet-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        printf("Failed to load ref\n");
        return 1;
    }

    nafnet_context * ctx = nafnet_init(argv[1], 2);
    if (!ctx) {
        printf("Failed to load model\n");
        return 1;
    }

    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input) {
        printf("No 'input' in ref\n");
        nafnet_free(ctx);
        return 1;
    }

    int W = 64, H = 64; // matches --size 64 in the reference dumper

    // Reference input is CHW float in [0,1]; nafnet_process wants HWC uint8.
    std::vector<uint8_t> input_u8(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                input_u8[(y * W + x) * 3 + c] =
                    (uint8_t)(std::max(0.0f, std::min(1.0f, ref_input[c * H * W + y * W + x])) * 255.0f + 0.5f);

    std::vector<uint8_t> output_u8(W * H * 3);
    int rc = nafnet_process(ctx, input_u8.data(), W, H, output_u8.data());
    if (rc != 0) {
        printf("nafnet_process failed\n");
        nafnet_free(ctx);
        return 1;
    }

    printf("Output: %dx%d\n", W, H);

    int n_pass = 0, n_fail = 0;
    if (ref.has("output")) {
        // Convert C++ HWC uint8 output back to CHW float [0,1].
        std::vector<float> cpp_out(3 * H * W);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int c = 0; c < 3; c++) cpp_out[c * H * W + y * W + x] = output_u8[(y * W + x) * 3 + c] / 255.0f;

        auto r = ref.compare("output", cpp_out.data(), cpp_out.size());
        // 0.99 floor: uint8 quantization + clamp loses precision vs the float ref,
        // but a real conv→ggml scramble craters cos far below this.
        bool pass = r.cos_min >= 0.99f;
        printf("output: cos_min=%.6f max_abs=%.2e %s\n", r.cos_min, r.max_abs, pass ? "PASS" : "FAIL");
        if (pass)
            n_pass++;
        else
            n_fail++;
    } else {
        printf("No 'output' in ref\n");
        n_fail++;
    }

    nafnet_free(ctx);
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
