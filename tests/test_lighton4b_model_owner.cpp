#include "lighton4b_model_owner.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_init_params params{ 723 * ggml_tensor_overhead() + 4096, nullptr, true };
    auto * context = ggml_init(params);
    if (!context) return 2;
    std::ifstream input(argv[1]);
    core_gguf::tensor_map tensors;
    std::string name;
    int64_t ne[4];
    int matrix;
    while (input >> name >> ne[0] >> ne[1] >> ne[2] >> ne[3] >> matrix) {
        auto * t = ggml_new_tensor(context, matrix ? GGML_TYPE_F16 : GGML_TYPE_F32, 4, ne);
        if (!t || t->data || !tensors.emplace(name, t).second) return 2;
    }
    if (!input.eof() || tensors.size() != 723) return 2;
    const char * gates[]{ "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER",
                          "CRISPEMBED_LIGHTON4B_VISION_OWNER", "CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER" };
    for (auto * g : gates) gate(g, "1");
    qwen35_ocr::lighton4b_model_owner owner;
    size_t checks = 0;
    bool ok = true;
    auto check = [&](bool value) {
        ++checks;
        if (!value) std::printf("FAIL control=%zu\n", checks);
        ok &= value;
    };
    auto map = [&] { return qwen35_ocr::map_lighton4b_model(tensors, owner); };
    auto reject = [&] {
        return !map() && owner.text.decoder.layers.empty() && owner.vision.layers.empty() &&
               !owner.text.decoder.embedding && !owner.vision.patch;
    };
    check(map());
    check(owner.text.decoder.layers.size() == 32 && owner.vision.layers.size() == 24 &&
          owner.text.decoder.head == tensors.at("l.embed_tokens.weight") && owner.vision_parameters.output == 2560 &&
          owner.vision_parameters.width == 1024 && owner.text.linear_key_heads == 16 &&
          owner.text.linear_value_heads == 32);
    for (auto * g : gates) {
        for (auto * value : { static_cast<const char *>(nullptr), "", "0" }) {
            gate(g, value);
            check(reject());
        }
        gate(g, "1");
        check(map());
    }
    // Copy keys so individual erasure never invalidates the traversal.
    const auto original = tensors;
    for (const auto & item : original) {
        auto * t = item.second;
        tensors.erase(item.first);
        check(reject());
        tensors.emplace(item);
        for (int axis = 0; axis < 4; ++axis) {
            ++t->ne[axis];
            check(reject());
            --t->ne[axis];
        }
        auto type = t->type;
        t->type = GGML_TYPE_I32;
        check(reject());
        t->type = type;
        if (item.first.rfind("v.", 0) == 0 && t->ne[1] == 1) {
            t->type = GGML_TYPE_F16;
            check(reject());
            t->type = type;
        }
        auto stride = t->nb[0];
        ++t->nb[0];
        check(reject());
        t->nb[0] = stride;
    }
    tensors["v.merger.norm.bias"] = tensors.at("v.merger.norm.weight");
    check(reject());
    tensors = original;
    tensors["v.merger.norm.bias"] = tensors.at("l.layers.3.self_attn.k_proj.weight");
    check(reject());
    tensors = original;
    tensors.erase("v.merger.norm.bias");
    tensors["unknown"] = original.at("v.merger.norm.bias");
    check(reject());
    tensors = original;
    tensors["lm_head.weight"] = original.at("l.embed_tokens.weight");
    check(reject());
    tensors = original;
    check(map());
    ggml_free(context);
    std::printf("MODEL_OWNER controls=%zu passed=%d payload_bytes=0 model_inference=0\n", checks, ok);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
