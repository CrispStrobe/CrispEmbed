// Model-free fixture/readback used with the ACTUAL C quantizer executable.
#include "core/clean_exit.h"
#include "core/gguf_file_type.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
int run(int argc, char ** argv) {
    require(argc >= 3, "fixture write PATH | check PATH TYPE FILETYPE");
    if (std::strcmp(argv[1], "write") == 0) {
        auto * ctx = ggml_init({ 1024 * 1024, nullptr, false });
        require(ctx, "tensor context");
        auto * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
        ggml_set_name(matrix, "fixture.weight");
        auto * values = static_cast<float *>(matrix->data);
        for (int i = 0; i < 512; ++i) values[i] = std::sin(float(i) * .13f) * .25f;
        auto * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_set_name(norm, "fixture_norm.weight");
        for (int i = 0; i < 256; ++i) static_cast<float *>(norm->data)[i] = 1.f;
        auto * meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "authored_fixture");
        gguf_set_val_u32(meta, "general.file_type", 0);
        gguf_add_tensor(meta, matrix);
        gguf_add_tensor(meta, norm);
        require(gguf_write_to_file(meta, argv[2], false), "write fixture");
        gguf_free(meta);
        ggml_free(ctx);
    } else {
        require(argc == 5 && std::strcmp(argv[1], "check") == 0, "check arguments");
        ggml_context * ctx = nullptr;
        gguf_init_params params{ true, &ctx };
        auto * meta = gguf_init_from_file(argv[2], params);
        require(meta && ctx, "read fixture");
        const auto key = gguf_find_key(meta, "general.file_type");
        require(key >= 0 && gguf_get_kv_type(meta, key) == GGUF_TYPE_UINT32 &&
                    gguf_get_val_u32(meta, key) == uint32_t(std::stoul(argv[4])),
                "actual standard header");
        require(gguf_get_n_tensors(meta) == 2 && gguf_get_tensor_type(meta, 0) == ggml_type(std::stoi(argv[3])) &&
                    gguf_get_tensor_type(meta, 1) == GGML_TYPE_F32,
                "actual quantized/critical tensor types");
        const auto * tensor = ggml_get_tensor(ctx, "fixture.weight");
        require(tensor && tensor->ne[0] == 256 && tensor->ne[1] == 2, "semantic fixture axes");
        gguf_free(meta);
        ggml_free(ctx);
    }
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        core_util::clean_exit(1);
    }
}
