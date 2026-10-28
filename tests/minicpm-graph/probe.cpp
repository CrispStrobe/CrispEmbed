// Hosted-only projection attribution. The model/input fixtures never enter
// the ordinary library; only this standalone build enables the observer.
#include "qwen35_graph_decoder.h"
#include "qwen35_math.h"
#include "core/clean_exit.h"
#include "core/json.h"
#include "crispembed_diff.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>

namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
void save(const std::string & path, const float * data, size_t count) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(data), std::streamsize(count * sizeof(float)));
    require(bool(out), "raw stage write");
}
bool include(const char * name, void *) {
    return std::string(name).rfind("l.", 0) == 0;
}
struct audit {
    std::string dir;
    std::ofstream rows;
    size_t calls = 0, written = 0;
    bool first = true;
    explicit audit(const std::string & directory) : dir(directory), rows(directory + "/projections.json") {
        require(bool(rows), "projection journal");
        rows << "{\"complete\":false,\"sampling\":\"first16-plus-last16-token-rows\",\"stages\":[";
    }
};
bool observed(ggml_tensor * weight, const float * input, const float * output, int tokens, int threads, void * user) {
    auto & a = *static_cast<audit *>(user);
    const size_t width = size_t(weight->ne[0]), rows = size_t(weight->ne[1]);
    std::vector<int> selected;
    for (int t = 0; t < tokens; ++t)
        if (t < 16 || t >= tokens - 16) selected.push_back(t);
    std::vector<float> subset(selected.size() * width), actual(selected.size() * rows);
    for (size_t i = 0; i < selected.size(); ++i) {
        std::copy_n(input + size_t(selected[i]) * width, width, subset.data() + i * width);
        std::copy_n(output + size_t(selected[i]) * rows, rows, actual.data() + i * rows);
    }
    std::vector<std::vector<float>> expected;
    require(qwen35_ocr::project_cpu({ weight }, subset.data(), int(width), int(selected.size()), threads, expected),
            "same-input retained dequantized projection control");
    const std::string key = "projection_" + std::to_string(a.calls++);
    a.written += (actual.size() + expected.at(0).size()) * sizeof(float);
    require(a.written <= 512ull * 1024 * 1024, "bounded projected journal");
    save(a.dir + "/" + key + ".native.f32", actual.data(), actual.size());
    save(a.dir + "/" + key + ".control.f32", expected[0].data(), expected[0].size());
    if (!a.first) a.rows << ',';
    a.first = false;
    a.rows << "{\"key\":\"" << key << "\",\"weight\":\"" << core_json::json_escape_strict(ggml_get_name(weight))
           << "\",\"stored_type\":\"" << ggml_type_name(weight->type) << "\",\"input_width\":" << width
           << ",\"output_width\":" << rows << ",\"tokens\":" << tokens << ",\"selected_rows\":[";
    for (size_t i = 0; i < selected.size(); ++i) {
        if (i) a.rows << ',';
        a.rows << selected[i];
    }
    a.rows << "],\"elements\":" << actual.size() << '}';
    a.rows.flush();
    require(bool(a.rows), "projection journal write");
    return true;
}
int body(int argc, char ** argv) {
    if (argc != 5) return 2;
    try {
        const std::string model = argv[1], fixture = argv[2], dir = argv[3];
        const int threads = std::stoi(argv[4]);
        require(threads >= 1 && threads <= 16, "thread bound");
        using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        backend_owner backend(ggml_backend_cpu_init(), ggml_backend_free);
        core_gguf::WeightLoad loaded;
        struct cleanup {
            core_gguf::WeightLoad & w;
            ~cleanup() { core_gguf::free_weights(w); }
        } owner{ loaded };
        require(backend && core_gguf::load_weights_filtered(model.c_str(), backend.get(), include, nullptr,
                                                            "minicpm-graph-probe", loaded),
                "model weights");
        qwen35_ocr::decoder_weights weights;
        require(qwen35_ocr::map_small_decoder(loaded.tensors, weights, 248100), "original strict model mapper");
        crispembed_diff::Ref input;
        require(input.load(fixture), "input adapter");
        auto ids = input.get_f32("input_ids"), merged = input.get_f32("merged");
        require(ids.first && ids.second > 0 && ids.second <= 4096 && merged.first && merged.second % 1024 == 0,
                "input coverage");
        std::vector<int> tokens;
        for (size_t i = 0; i < ids.second; ++i) {
            require(std::isfinite(ids.first[i]) && ids.first[i] >= 0 && ids.first[i] < 248100 &&
                        std::floor(ids.first[i]) == ids.first[i],
                    "exact token IDs");
            tokens.push_back(int(ids.first[i]));
        }
        std::vector<float> hidden;
        require(qwen35_ocr::embed_cpu(weights.embedding, tokens, hidden), "original embedding reader");
        size_t index = 0;
        for (size_t i = 0; i < tokens.size(); ++i)
            if (tokens[i] == 248056) {
                require(index + 1024 <= merged.second, "image scatter extent");
                std::copy_n(merged.first + index, 1024, hidden.data() + i * 1024);
                index += 1024;
            }
        require(index == merged.second, "all merged rows consumed");
        std::vector<float> positions(3 * tokens.size());
        for (int axis = 0; axis < 3; ++axis)
            for (size_t t = 0; t < tokens.size(); ++t) positions[size_t(axis) * tokens.size() + t] = float(t);
        qwen35_ocr::graph_decoder decoder;
        qwen35_ocr::decoder_outputs ordinary, output, repeat;
        require(
            decoder.run(weights, {}, hidden.data(), positions.data(), int(tokens.size()), threads, nullptr, ordinary),
            "ordinary prefill");
        audit journal(dir);
        decoder.observe(observed, &journal);
        require(decoder.run(weights, {}, hidden.data(), positions.data(), int(tokens.size()), threads, nullptr, output),
                "observed prefill");
        decoder.observe(nullptr, nullptr);
        require(decoder.run(weights, {}, hidden.data(), positions.data(), int(tokens.size()), threads, nullptr, repeat),
                "repeat prefill");
        require(ordinary.hidden == output.hidden && output.hidden == repeat.hidden && ordinary.norm == output.norm &&
                    output.norm == repeat.norm && ordinary.logits == output.logits && output.logits == repeat.logits,
                "observer/repeat output neutrality");
        require(ordinary.cache.tokens == output.cache.tokens && output.cache.tokens == repeat.cache.tokens &&
                    output.cache.layers.size() == 24,
                "cache structure neutrality");
        for (size_t i = 0; i < 24; ++i) {
            const auto & a = ordinary.cache.layers[i];
            const auto & b = output.cache.layers[i];
            const auto & c = repeat.cache.layers[i];
            require(a.conv == b.conv && b.conv == c.conv && a.recurrent == b.recurrent && b.recurrent == c.recurrent &&
                        a.attention.keys == b.attention.keys && b.attention.keys == c.attention.keys &&
                        a.attention.values == b.attention.values && b.attention.values == c.attention.values,
                    "cache value neutrality");
            save(dir + "/layer_" + std::to_string(i) + ".f32", output.hidden[i].data(), output.hidden[i].size());
        }
        save(dir + "/norm.f32", output.norm.data(), output.norm.size());
        save(dir + "/logits.f32", output.logits.data(), output.logits.size());
        // 18 linear blocks x5,6 full-attention blocks x4,24 FFNs x3,tied head.
        require(journal.calls == 187, "exact projection-stage coverage");
        journal.rows << "],\"completed\":true,\"ordinary_observer_repeat_exact\":true,\"cache_exact\":true,\"calls\":"
                     << journal.calls << "}\n";
        require(bool(journal.rows), "completed journal write");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
} // namespace
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
