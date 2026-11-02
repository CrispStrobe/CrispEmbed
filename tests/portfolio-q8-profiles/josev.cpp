// Metadata and sparse-file bounds only; no weights or inference are allocated.
#include "josev2b_packed_profile.h"
#include "core/clean_exit.h"
#include "core/temp_file.h"
#include <filesystem>
#include <memory>
#include <cstdio>
namespace {
struct shape_record {
    const char * name;
    int64_t ne[4];
};
#include "../josev-quant/backbone-shapes.h"
struct spec {
    const char * name;
    int64_t ne[4];
};
#include "../josev-packed/head-shapes.h"
void check(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
struct temporary {
    std::string path = core_tmp::make_private(".gguf");
    ~temporary() {
        if (!path.empty()) std::remove(path.c_str());
    }
};
bool bounds(const char * format, uint32_t header, ggml_type projection_type, bool allowed, bool bad_embedding = false) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 2 * 1024 * 1024, nullptr, true }), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
    check(bool(ctx) && bool(meta), "metadata allocation");
    gguf_set_val_u32(meta.get(), "general.alignment", 32);
    gguf_set_val_u32(meta.get(), "general.file_type", header);
    gguf_set_val_str(meta.get(), "josev2b.packed.format", format);
    ggml_tensor * last = nullptr;
    auto add = [&](const char * name, const int64_t * ne) {
        auto type = josev2b_packed::projection(name) ? projection_type : GGML_TYPE_F32;
        if (bad_embedding && std::strcmp(name, "l.embed_tokens.weight") == 0) type = GGML_TYPE_Q8_0;
        last = ggml_new_tensor_4d(ctx.get(), type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(last, name);
        gguf_add_tensor(meta.get(), last);
    };
    for (const auto & row : SHAPES) add(row.name, row.ne);
    for (const auto & row : head_rows) add(row.name, row.ne);
    temporary file;
    check(!file.path.empty() && gguf_write_to_file(meta.get(), file.path.c_str(), true), "sparse metadata fixture");
    const auto data = (gguf_get_meta_size(meta.get()) + 31) / 32 * 32;
    const auto count = gguf_get_n_tensors(meta.get());
    std::filesystem::resize_file(file.path, data + gguf_get_tensor_offset(meta.get(), count - 1) + ggml_nbytes(last));
    std::string error;
    return josev2b_packed::file_bounds(file.path.c_str(), error, allowed);
}
int run() {
    const char * old = "crispembed-josev-packed-q4k-v1";
    const char * mixed = "crispembed-josev-packed-mixed-v2";
    check(bounds("crispembed-josev-packed-f32-v1", 0, GGML_TYPE_F32, false), "F32 bounds changed");
    for (auto label : { 12u, 14u }) check(bounds(old, label, GGML_TYPE_Q4_K, true), "legacy Q4 bounds changed");
    check(bounds(mixed, 7, GGML_TYPE_Q8_0, true), "actual Q8 bounds rejected");
    check(bounds(mixed, 14, GGML_TYPE_Q4_K, true), "named mixed Q4 bounds rejected");
    check(!bounds(mixed, 7, GGML_TYPE_Q8_0, false), "Q8 bypassed quant gate");
    check(!bounds(old, 7, GGML_TYPE_Q8_0, true), "Q8 mislabeled as legacy Q4");
    check(!bounds(mixed, 7, GGML_TYPE_Q4_K, true), "Q8 header accepted Q4 payload");
    check(!bounds(mixed, 14, GGML_TYPE_Q8_0, true), "Q4 header accepted Q8-only payload");
    check(!bounds(mixed, 7, GGML_TYPE_F16, true), "floating-only payload mislabeled Q8");
    check(!bounds(mixed, 7, GGML_TYPE_Q8_0, true, true), "critical embedding quantized");
    std::puts("PASS: 11 Josev raw-bounds/profile controls; no model payload allocation or inference");
    return 0;
}
} // namespace
int main() {
    try {
        core_util::clean_exit(run());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
