#include "capture.h"
#include "core/clean_exit.h"
#include <limits>
#include <stdexcept>
#include <vector>
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
static ggml_tensor weight(const char * name, int cols, int rows) {
    ggml_tensor t{};
    t.type = GGML_TYPE_F32;
    t.ne[0] = cols;
    t.ne[1] = rows;
    t.ne[2] = t.ne[3] = 1;
    ggml_set_name(&t, name);
    return t;
}
static int body(int argc, char ** argv) {
    try {
        require(argc == 2, "Task-owned output directory required");
        const auto directory = std::filesystem::path(argv[1]);
        std::filesystem::create_directories(directory);
        auto gate = weight("l.layers.23.mlp.gate_proj.weight", 2048, 6144);
        auto up = weight("l.layers.23.mlp.up_proj.weight", 2048, 6144);
        auto norm = weight("l.layers.23.post_attention_layernorm.weight", 2048, 1);
        std::vector<float> input(2048, 0.5f);
        for (int test = 0; test < 6; ++test) {
            const auto folder = directory / std::to_string(test);
            std::filesystem::create_directory(folder);
            require(josev_ffn_imatrix_begin(folder.string().c_str()) == 1, "begin");
            auto bad = gate;
            if (test == 0) bad.type = GGML_TYPE_Q4_K;
            if (test == 1) bad.ne[0] = 1024;
            if (test == 2) ggml_set_name(&bad, "wrong.weight");
            if (test == 3) input[0] = std::numeric_limits<float>::quiet_NaN();
            require(!josev_ffn_imatrix_test::capture(input.data(), 2048, test == 4 ? 0 : 1, &bad, &up, &norm, 1e-6f,
                                                     test == 5 ? 2 : 1, 0),
                    "invalid capture admitted");
            require(josev_ffn_imatrix_finish() == 0, "invalid finish admitted");
            input[0] = 0.5f;
        }
        auto folder = directory / "positive";
        std::filesystem::create_directory(folder);
        require(josev_ffn_imatrix_begin(folder.string().c_str()) == 1, "positive begin");
        require(josev_ffn_imatrix_test::capture(input.data(), 2048, 1, &gate, &up, &norm, 1e-6f, 1, 0), "live input");
        require(josev_ffn_imatrix_finish() == 1, "positive finish");
        require(josev_ffn_imatrix_finish() == 0, "stale finish accepted");
        folder = directory / "duplicate";
        std::filesystem::create_directory(folder);
        require(josev_ffn_imatrix_begin(folder.string().c_str()) == 1, "duplicate begin");
        require(josev_ffn_imatrix_test::capture(input.data(), 2048, 1, &gate, &up, &norm, 1e-6f, 1, 0),
                "first capture");
        require(!josev_ffn_imatrix_test::capture(input.data(), 2048, 1, &gate, &up, &norm, 1e-6f, 1, 0) &&
                    josev_ffn_imatrix_finish() == 0,
                "duplicate capture accepted");
        std::puts("PASS dtype/shape/name/nonfinite/geometry/duplicate/recovery capture controls");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
