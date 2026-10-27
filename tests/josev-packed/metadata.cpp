#include "core/clean_exit.h"
#include "core/temp_file.h"
#include "josev2b_composed.h"
#include "josev2b_head.h"
#include "josev2b_packed_profile.h"
#include "josev2b_text_frontend.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
struct spec {
    const char * name;
    int64_t ne[4];
};
#include "head-shapes.h"
void check(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
struct temporary {
    std::string path = core_tmp::make_private(".gguf");
    temporary() { check(!path.empty(), "private fixture creation failed"); }
    ~temporary() { std::remove(path.c_str()); }
};
void flag(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
void u64(std::ostream & out, uint64_t n, int width) {
    for (int i = 0; i < width; ++i) out.put(char((n >> (8 * i)) & 255));
}
int run() {
    std::string error;
    for (auto * key :
         { "CRISPEMBED_JOSEV_COMPOSED", "CRISPEMBED_JOSEV_BACKBONE", "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_JOSEV_HEAD",
           "CRISPEMBED_JOSEV_TEXT_FRONTEND", "CRISPEMBED_JOSEV_ANSWER_SERIALIZER" })
        flag(key, "1");
    for (auto * mode : { static_cast<const char *>(nullptr), "", "0" }) {
        flag("CRISPEMBED_JOSEV_PACKED", mode);
        check(!josev2b::composed_context::load_packed("not-a-model", 2, error) &&
                  error == "Josev packed/component gates are off",
              "packed off gate touched missing model");
    }
    flag("CRISPEMBED_JOSEV_PACKED", "false");
    check(!josev2b::composed_context::load_packed("not-a-model", 2, error) && error == "packed file unavailable",
          "literal false must retain shared value-parsed ON semantics");
    flag("CRISPEMBED_JOSEV_PACKED", "1");
    for (int control = 0; control < 4; ++control) {
        temporary f;
        std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
        u64(out, control == 0 ? 0 : 0x46554747, 4);
        u64(out, control == 1 ? 2 : 3, 4);
        u64(out, control == 2 ? 441 : 442, 8);
        u64(out, control == 3 ? 4097 : 0, 8);
        out.close();
        check(!josev2b_packed::file_bounds(f.path.c_str(), error) && !error.empty(), "bad packed header accepted");
    }
    auto * ctx = ggml_init({ 140 * ggml_tensor_overhead(), nullptr, true });
    check(ctx, "metadata-only tensor allocation failed");
    core_gguf::tensor_map tensors;
    for (const auto & row : head_rows) {
        auto * t = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, row.ne);
        ggml_set_name(t, row.name);
        tensors.emplace(row.name, t);
    }
    check(josev2b_head::validate_packed_shapes(tensors, error), "independent122 head shape fixture rejected");
    auto missing = tensors;
    missing.erase(head_rows[0].name);
    check(!josev2b_head::validate_packed_shapes(missing, error), "missing head tensor accepted");
    auto * t = tensors.at(head_rows[0].name);
    const auto old = t->ne[0];
    t->ne[0] = old + 1;
    check(!josev2b_head::validate_packed_shapes(tensors, error), "wrong head shape accepted");
    t->ne[0] = old;
    check(!josev2b_head::init_borrowed_cpu(tensors, 2, error) && error == "Borrowed head requires host storage",
          "metadata-only pointers must never become borrowed live weights");
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "josev2b_systemone");
    for (const auto & item : tensors) gguf_add_tensor(meta, item.second);
    temporary file;
    gguf_write_to_file(meta, file.path.c_str(), true);
    gguf_free(meta);
    josev2b::text_frontend legacy;
    check(!legacy.load(file.path.c_str(), error) && error == "expected metadata-only tokenizer GGUF",
          "old tokenizer loader accepted nonzero tensor profile");
    check(!josev2b_head::init(file.path.c_str(), josev2b_head::original_receipt_sha256,
                              josev2b_head::original_worker_source, "cpu", 2, error),
          "old head loader accepted packed architecture");
    ggml_free(ctx);
    std::cout << "PASS packed gates/header bounds/independent122shapes/borrowed storage/legacy rejection\n";
    return 0;
}
} // namespace
int main() {
    try {
        core_util::clean_exit(run());
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        core_util::clean_exit(1);
    }
}
