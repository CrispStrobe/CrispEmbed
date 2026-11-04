#include "core/clean_exit.h"
#include "qwen35_vision.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static int run() {
    // Header-only tensors: no checkpoint, backend allocation or inference.
    std::map<std::string, ggml_tensor> storage;
    core_gguf::tensor_map tensors;
    auto add = [&](const std::string & name, int columns, int rows) {
        auto & t = storage[name];
        t.type = GGML_TYPE_F32;
        t.ne[0] = columns;
        t.ne[1] = rows;
        t.ne[2] = t.ne[3] = 1;
        tensors[name] = &t;
    };
    add("v.patch_embed.proj.weight", 1536, 1024);
    add("v.patch_embed.proj.bias", 1024, 1);
    add("v.pos_embed.weight", 1024, 2304);
    add("v.merger.norm.weight", 1024, 1);
    add("v.merger.norm.bias", 1024, 1);
    add("v.merger.linear_fc1.weight", 4096, 4096);
    add("v.merger.linear_fc1.bias", 4096, 1);
    add("v.merger.linear_fc2.weight", 4096, 2560);
    add("v.merger.linear_fc2.bias", 2560, 1);
    for (int i = 0; i < 24; ++i) {
        auto p = "v.blocks." + std::to_string(i) + ".";
        for (auto name :
             { "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.proj.bias", "mlp.linear_fc2.bias" })
            add(p + name, 1024, 1);
        add(p + "attn.qkv.weight", 1024, 3072);
        add(p + "attn.qkv.bias", 3072, 1);
        add(p + "attn.proj.weight", 1024, 1024);
        add(p + "mlp.linear_fc1.weight", 1024, 4096);
        add(p + "mlp.linear_fc1.bias", 4096, 1);
        add(p + "mlp.linear_fc2.weight", 4096, 1024);
    }
    qwen35_ocr::vision_weights w;
    qwen35_ocr::vision_config c;
    auto map = [&] { return qwen35_ocr::map_lighton4b_vision(tensors, w, c); };
    auto rejected = [&] { return !map() && w.layers.empty() && !w.patch && c.width == 768; };
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_LIGHTON4B_VISION_OWNER", "0");
    bool ok = rejected();
    gate("CRISPEMBED_LIGHTON4B_VISION_OWNER", "1");
    ok &= map() && w.layers.size() == 24 && c.width == 1024 && c.heads == 16 && c.output == 2560;
    for (auto & item : storage) {
        auto & t = item.second;
        const auto saved = t;
        t.ne[0]++;
        ok &= rejected();
        t = saved;
        t.ne[2] = 2;
        ok &= rejected();
        t = saved;
        t.type = GGML_TYPE_I32;
        ok &= rejected();
        t = saved;
        tensors.erase(item.first);
        ok &= rejected();
        tensors[item.first] = &t;
        t.type = GGML_TYPE_F16;
        ok &= map();
        t = saved;
    }
    add("v.deepstack_merger_list.0.weight", 1, 1);
    ok &= rejected();
    tensors.erase("v.deepstack_merger_list.0.weight");
    // Shared 723-entry maps contain text names; this component owns v.* only.
    add("t.embed_tokens.weight", 2560, 248320);
    ok &= map();
    gate("CRISPEMBED_QWEN35_OCR", "0");
    ok &= rejected();
    gate("CRISPEMBED_LIGHTON4B_VISION_OWNER", "0");
    std::printf("%s LightOn4B vision ownership, 297 tensors and per-tensor malformed controls\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
int main() {
    core_util::clean_exit(run());
}
