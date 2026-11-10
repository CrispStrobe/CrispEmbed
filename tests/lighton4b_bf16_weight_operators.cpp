#include "lighton4b_native_context.h"
#include "lighton4b_bf16_dequant_weights.h"
#include "core/sha256_accelerated.h"
#include "core/clean_exit.h"
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstdlib>
#include <cstring>
using namespace qwen35_ocr;
static void gate(const char * v) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", v);
#else
    setenv("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", v, 1);
#endif
}
static std::vector<float> read(const std::string & p, size_t count) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> x(count);
    f.read(reinterpret_cast<char *>(x.data()), count * 4);
    if (!f || f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("source vector shape differs");
    return x;
}
static void emit(const std::string & p, const std::vector<float> & x) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
    if (!f) throw std::runtime_error("operator write failed");
}
static int run(int argc, char ** argv) {
    if (argc != 5) return 2;
    gate("0");
    auto context = lighton4b_native_context::open(argv[1]);
    if (!context) return 3;
    const auto & w = context->model().text.decoder;
    const std::string input = argv[2], out = argv[4];
    std::map<std::string, ggml_tensor *> tensors;
    auto add = [&](ggml_tensor * t) {
        if (!t || !tensors.emplace(ggml_get_name(t), t).second)
            throw std::runtime_error("quantized tensor identity differs");
    };
    add(w.embedding);
    for (const auto & layer : w.layers) {
        for (auto * t : { layer.gate, layer.up, layer.down }) add(t);
        if (layer.full_attention) {
            for (auto * t : { layer.attention.q, layer.attention.k, layer.attention.v, layer.attention.out }) add(t);
        } else {
            for (auto * t : { layer.linear.qkv, layer.linear.z, layer.linear.a, layer.linear.b, layer.linear.out })
                add(t);
        }
    }
    if (tensors.size() != 249 || w.head != w.embedding) return 4;
    std::ifstream expected(argv[3]);
    std::map<std::string, std::pair<std::string, std::string>> hashes;
    std::string name, raw_expected, bf16_expected;
    while (expected >> name >> raw_expected >> bf16_expected)
        if (!hashes.emplace(name, std::make_pair(raw_expected, bf16_expected)).second) return 5;
    if (hashes.size() != 249) return 5;
    std::ofstream proof(out + "/weight-hashes.tsv");
    gate("1");
    size_t maximum = 0;
    for (const auto & item : tensors) {
        auto * t = item.second;
        const auto found = hashes.find(item.first);
        if (found == hashes.end() || !lighton4b_bf16_dequant_weights_enabled(t)) return 6;
        const size_t width = size_t(t->ne[0]), rows = size_t(t->ne[1]), row_bytes = ggml_row_size(t->type, width),
                     tile_rows = std::max(size_t(1), size_t(65536) / width);
        std::vector<uint8_t> raw, bf16;
        std::vector<float> values;
        core_sha256_accel::Hasher raw_hash(true), bf16_hash(true);
        for (size_t begin = 0; begin < rows; begin += tile_rows) {
            const size_t count = std::min(tile_rows, rows - begin), elements = count * width;
            maximum = std::max(maximum, elements);
            raw.resize(count * row_bytes);
            values.resize(elements);
            bf16.resize(2 * elements);
            ggml_backend_tensor_get(t, raw.data(), begin * row_bytes, raw.size());
            if (!raw_hash.update(raw.data(), raw.size())) return 7;
            ggml_get_type_traits(t->type)->to_float(raw.data(), values.data(), int64_t(elements));
            lighton4b_bf16_dequant_weights(t, values.data(), values.size());
            for (size_t i = 0; i < elements; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &values[i], 4);
                if (bits & 0xffff) return 8;
                bf16[2 * i] = uint8_t(bits >> 16);
                bf16[2 * i + 1] = uint8_t(bits >> 24);
            }
            if (!bf16_hash.update(bf16.data(), bf16.size())) return 7;
        }
        const auto a = raw_hash.finish(), b = bf16_hash.finish();
        proof << item.first << '\t' << a << '\t' << b << '\n';
        if (a != found->second.first || b != found->second.second) return 9;
    }
    if (maximum > 65536) return 10;
    const auto embed = read(input + "/greedy.33.input.f32", 2560),
               norm = read(input + "/greedy.33.head.input.f32", 2560),
               post = read(input + "/greedy.33.layer0.post_norm.f32", 2560),
               activation = read(input + "/greedy.33.layer0.activation.f32", 9216);
    int pass = 0;
    std::map<std::string, std::vector<float>> baseline;
    for (const char * mode : { "0", "1", "0" }) {
        gate(mode);
        const std::string label = pass == 0 ? "off" : pass == 1 ? "on" : "off-repeat";
        std::vector<float> embedded, head;
        if (!embed_cpu(w.embedding, { 68100 }, embedded) ||
            !project_rows_cpu(w.head, norm.data(), 2560, 1, 4, 256, head))
            return 11;
        emit(out + "/" + label + ".embedding.f32", embedded);
        if (pass == 0) baseline["embedding"] = embedded;
        if (pass == 2 && embedded != baseline["embedding"]) return 14;
        if (label == "on" && embedded != embed) return 12;
        std::vector<std::vector<float>> mlp, down;
        if (!project_cpu({ w.layers[0].gate, w.layers[0].up }, post.data(), 2560, 1, 4, mlp) ||
            !project_cpu({ w.layers[0].down }, activation.data(), 9216, 1, 4, down))
            return 13;
        for (auto pair : { std::make_pair("head", &head), std::make_pair("gate", &mlp[0]),
                           std::make_pair("up", &mlp[1]), std::make_pair("down", &down[0]) }) {
            emit(out + "/" + label + "." + pair.first + ".raw.f32", *pair.second);
            if (pass == 0) baseline[pair.first] = *pair.second;
            if (pass == 2 && *pair.second != baseline[pair.first]) return 15;
            for (auto & x : *pair.second) x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
            emit(out + "/" + label + "." + pair.first + ".bf16.f32", *pair.second);
        }
        ++pass;
    }
    gate("0");
    std::cout << "BF16_WEIGHT_OPERATORS passed=1 tensors=249 max_chunk_elements=" << maximum
              << " tied_embedding_head=1 storage_sha_verified=1 source_embedding_exact=1 gate_off_repeat_exact=1 "
                 "full_model_f32=0\n";
    return 0;
}
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        core_util::clean_exit(1);
    }
}
