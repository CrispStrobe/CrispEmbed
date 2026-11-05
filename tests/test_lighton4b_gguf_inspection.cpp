#include "lighton4b_gguf_inspection.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
struct field {
    char type;
    std::string key, value;
};
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static int run(int argc, char ** argv) {
    if (argc != 4) return 2;
    ggml_init_params params{ 723 * ggml_tensor_overhead() + 4096, nullptr, true };
    auto * headers = ggml_init(params);
    auto * fixture = gguf_init_empty();
    if (!headers || !fixture) return 2;
    std::ifstream schema(argv[1]);
    std::string name;
    int64_t ne[4];
    int matrix;
    while (schema >> name >> ne[0] >> ne[1] >> ne[2] >> ne[3] >> matrix) {
        auto * t = ggml_new_tensor(headers, matrix ? GGML_TYPE_F16 : GGML_TYPE_F32, 4, ne);
        if (!t) return 2;
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(fixture, t);
    }
    if (!schema.eof() || gguf_get_n_tensors(fixture) != 723) return 2;
    std::ifstream metadata(argv[2]);
    std::vector<field> fields;
    field item;
    while (metadata >> item.type >> item.key >> item.value) fields.push_back(item);
    if (!metadata.eof() || fields.empty()) return 2;
    auto set = [&](const field & f) {
        if (f.type == 's')
            gguf_set_val_str(fixture, f.key.c_str(), f.value.c_str());
        else if (f.type == 'u')
            gguf_set_val_u32(fixture, f.key.c_str(), uint32_t(std::stoul(f.value)));
        else if (f.type == 'b')
            gguf_set_val_bool(fixture, f.key.c_str(), f.value == "1");
    };
    for (const auto & f : fields) set(f);
    const char * gates[]{ "CRISPEMBED_LIGHTON4B_GGUF_INSPECTION", "CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER",
                          "CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER", "CRISPEMBED_LIGHTON4B_VISION_OWNER",
                          "CRISPEMBED_QWEN35_OCR" };
    for (auto * g : gates) gate(g, "1");
    auto write = [&] { return gguf_write_to_file(fixture, argv[3], true); };
    auto open = [&] { return qwen35_ocr::lighton4b_gguf_inspection::open(argv[3]); };
    size_t controls = 0;
    bool ok = true;
    auto check = [&](bool value) {
        ++controls;
        if (!value) std::printf("FAIL inspection control=%zu\n", controls);
        ok &= value;
    };
    check(write());
    auto context = open();
    check(context && context->model().text.decoder.layers.size() == 32 && context->model().vision.layers.size() == 24 &&
          !context->model().text.decoder.embedding->data && !context->model().vision.patch->data);
    // Simultaneous contexts must independently own distinct tensor headers.
    auto second = open();
    check(context && second && context->model().vision.patch != second->model().vision.patch);
    second.reset();
    check(context && !context->model().vision.patch->data);
    context.reset();
    for (const auto & f : fields) {
        if (f.type == 's')
            gguf_set_val_str(fixture, f.key.c_str(), "wrong");
        else if (f.type == 'u')
            gguf_set_val_u32(fixture, f.key.c_str(), uint32_t(std::stoul(f.value) + 1));
        else
            gguf_set_val_bool(fixture, f.key.c_str(), false);
        check(write() && !open());
        set(f);
        // Type confusion must fail instead of applying a fallback geometry.
        if (f.type == 's')
            gguf_set_val_u32(fixture, f.key.c_str(), 1);
        else
            gguf_set_val_str(fixture, f.key.c_str(), "1");
        check(write() && !open());
        set(f);
    }
    check(write());
    for (auto * g : gates) {
        gate(g, "0");
        check(!open());
        gate(g, "1");
    }
    for (int i = 0; i < 8; ++i) check(bool(open()));
    std::remove(argv[3]);
    gguf_free(fixture);
    ggml_free(headers);
    std::printf("GGUF_INSPECTION controls=%zu passed=%d payload_bytes=0 model_inference=0\n", controls, ok);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
