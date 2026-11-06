// Actual frozen RGB -> native frontend -> native vision; source arrays compare only.
#include "lighton4b_vision_bf16.h"
#define lighton4b_vision_bf16_cpu lighton4b_vision_observed_cpu
#include "lighton4b_vision_observed.inc"
#undef lighton4b_vision_bf16_cpu
#include "lighton4b_image_compose.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
#include <iterator>
using namespace qwen35_ocr;
static std::vector<float> read_floats(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("float source missing");
    auto size = f.tellg();
    if (size < 0 || size % 4) throw std::runtime_error("float source size");
    f.seekg(0);
    std::vector<float> x(size_t(size) / 4);
    f.read(reinterpret_cast<char *>(x.data()), size);
    if (!f) throw std::runtime_error("float source read");
    return x;
}
static void emit(const std::string & path, const std::vector<float> & x) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
    if (!f) throw std::runtime_error("output failed");
}
static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
static std::vector<std::pair<std::string, const std::vector<float> *>> stages(const image_inputs & i,
                                                                              const vision_outputs & o) {
    std::vector<std::pair<std::string, const std::vector<float> *>> v = {
        { "processor.patches", &i.patches },    { "vision.patch", &o.patch },
        { "vision.position", &o.position },     { "vision.embedded", &o.embedded },
        { "vision.rotary_cos", &o.rotary_cos }, { "vision.rotary_sin", &o.rotary_sin }
    };
    for (size_t k = 0; k < o.hidden.size(); ++k) v.emplace_back("vision.layer." + std::to_string(k), &o.hidden[k]);
    for (auto x : { std::pair<const char *, const std::vector<float> *>{ "vision.merger.norm", &o.merger_norm },
                    { "vision.merger.up", &o.merger_up },
                    { "vision.merger.activation", &o.merger_activation },
                    { "vision.merged", &o.merged } })
        v.emplace_back(x.first, x.second);
    return v;
}
int main(int argc, char ** argv) {
    if (argc != 9) return 2; // weights root, case root, height,width,reference H,W,threads,mode
    const std::string weights_root = argv[1], root = argv[2], mode = argv[8];
    int h = std::stoi(argv[3]), w = std::stoi(argv[4]), gh = std::stoi(argv[5]), gw = std::stoi(argv[6]),
        threads = std::stoi(argv[7]);
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    if (!ctx) return 3;
    std::ifstream meta(weights_root + "/weights.txt");
    core_gguf::tensor_map tensors;
    struct item {
        ggml_tensor * t;
        size_t offset, bytes;
    };
    std::vector<item> entries;
    std::string name;
    int columns, rows;
    size_t offset, bytes;
    while (meta >> name >> columns >> rows >> offset >> bytes) {
        if (columns < 1 || rows < 1 || bytes != size_t(columns) * rows * 4 || tensors.count(name)) return 4;
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        tensors[name] = t;
        entries.push_back({ t, offset, bytes });
    }
    if (!meta.eof() || entries.size() != 297) return 5;
    auto * backend = ggml_backend_cpu_init();
    if (!backend) return 6;
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 6;
    std::ifstream data(weights_root + "/weights.f32", std::ios::binary);
    for (auto e : entries) {
        std::vector<float> x(e.bytes / 4);
        data.seekg(e.offset);
        data.read(reinterpret_cast<char *>(x.data()), e.bytes);
        if (!data) return 7;
        ggml_backend_tensor_set(e.t, x.data(), 0, e.bytes);
    }
    vision_weights weights;
    vision_config config;
    if (!map_lighton4b_vision(tensors, weights, config)) return 8;
    std::ifstream input(root + "/input.rgb", std::ios::binary);
    std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(input)), {});
    if (rgb.size() != size_t(h) * w * 3) return 9;
    image_inputs image;
    if (mode == "native") {
        if (!lighton4b_preprocess_rgb(rgb.data(), h, w, image)) return 10;
    } else if (mode == "source-input") {
        image.grid = { 1, gh, gw };
        image.height = gh * 16;
        image.width = gw * 16;
        image.patches = read_floats(root + "/source/processor.patches.f32");
    } else
        return 2;
    vision_outputs output;
    if (!lighton4b_image_vision_cpu(weights, config, image, threads, output)) return 11;
    const std::string out = root + "/" + mode;
    auto values = stages(image, output);
    std::string first;
    int layer = -1;
    for (auto e : values) {
        emit(out + "/" + e.first + ".f32", *e.second);
        auto expected = read_floats(root + "/source/" + e.first + ".f32");
        if (first.empty() && !same(*e.second, expected)) {
            first = e.first;
            if (first.rfind("vision.layer.", 0) == 0) layer = std::stoi(first.substr(13));
        }
    }
    std::ofstream grid(out + "/grid.ids");
    for (int x : image.grid) grid << x << '\n';
    if (!image.resized_rgb.empty()) {
        std::ofstream resized(out + "/resized.rgb", std::ios::binary);
        resized.write(reinterpret_cast<const char *>(image.resized_rgb.data()), image.resized_rgb.size());
    }
    bool observed = false;
    if (layer >= 0) {
        vision_outputs trace;
        if (!lighton4b_vision_observed_cpu(weights, config, image.patches.data(), int(image.patches.size() / 1536),
                                           image.grid, threads, trace))
            return 12;
        auto traced = stages(image, trace);
        if (values.size() != traced.size()) return 13;
        for (size_t k = 0; k < values.size(); ++k)
            if (values[k].first != traced[k].first || !same(*values[k].second, *traced[k].second)) return 14;
        auto & p = trace.probes[layer];
        for (auto e : { std::pair<const char *, const std::vector<float> *>{ "norm1", &p.norm1 },
                        { "qkv", &p.qkv },
                        { "pooled", &p.pooled },
                        { "attention", &p.attention },
                        { "norm2", &p.norm2 },
                        { "up", &p.up },
                        { "activation", &p.activation },
                        { "down", &p.down } })
            emit(out + "/probe." + std::to_string(layer) + "." + e.first + ".f32", *e.second);
        observed = true;
    }
    std::ofstream report(out + "/native-summary.json");
    report << "{\"mapped_tensors\":297,\"first_differing_stage\":\"" << first
           << "\",\"first_differing_layer\":" << layer << ",\"observer_executed\":" << (observed ? "true" : "false")
           << ",\"observer_matches_actual\":" << (observed ? "true" : "null")
           << ",\"source_input_diagnostic\":" << (mode == "source-input" ? "true" : "false") << "}\n";
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
