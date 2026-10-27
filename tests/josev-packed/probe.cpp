#include "core/clean_exit.h"
#include "core/decision_schema.h"
#include "josev2b_composed.h"
#include "json.h"
#ifdef CRISPEMBED_JOSEV_COMPOSED_TESTING
#include "../josev2b_composed_observer.h"
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using composed_json::q;
namespace {
std::filesystem::path directory;
size_t stage_index = 0;
std::ofstream stages;
void check(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
std::string read(const char * path) {
    std::ifstream stream(path, std::ios::binary);
    check(bool(stream), "request packet unavailable");
    const auto size = std::filesystem::file_size(path);
    check(size <= 1024 * 1024, "request packet too large");
    std::string out(size, '\0');
    stream.read(out.data(), size);
    check(bool(stream), "request packet truncated");
    return out;
}
void raw(const std::string & label, size_t occurrence, const std::vector<int64_t> & shape,
         const std::vector<float> & values) {
    size_t n = 1;
    for (auto d : shape) {
        check(d > 0 && d <= 1048576 && n <= 16777216 / size_t(d), "bad observed shape");
        n *= size_t(d);
    }
    check(n == values.size(), "observed shape/data count differs");
    const std::string name = std::to_string(stage_index++) + ".f32";
    std::ofstream file(directory / name, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    check(bool(file), "raw write failed");
    stages << "{\"stage\":" << q(label) << ",\"occurrence\":" << occurrence << ",\"shape\":";
    composed_json::numbers(stages, shape);
    stages << ",\"file\":" << q(name) << "}\n";
    check(bool(stages), "stage manifest failed");
}
} // namespace
#ifdef CRISPEMBED_JOSEV_COMPOSED_TESTING
namespace josev2b {
namespace composed_test {
void observe_head(const josev2b_head::test::stage & stage) {
    raw(stage.label, stage.occurrence, stage.shape, stage.values);
}
void observe(const encoded_text & encoded, const std::vector<float> & embedding,
             const qwen35_ocr::decoder_outputs & backbone, const josev2b_head::input & input,
             const josev2b_head::output & output, const std::string &) {
    std::ofstream packet(directory / "encoded.json");
    composed_json::dump(packet, encoded);
    packet << '\n';
    check(bool(packet), "encoded write failed");
    const std::vector<int64_t> shape{ 1, input.tokens, 2048 };
    raw("backbone/embedding", 0, shape, embedding);
    check(backbone.hidden.size() == 24 && backbone.logits.empty(), "hidden-only stage count differs");
    for (size_t i = 0; i < backbone.hidden.size(); ++i)
        raw("backbone/layer/" + std::to_string(i), 0, shape, backbone.hidden[i]);
    raw("backbone/final_norm", 0, shape, backbone.norm);
    for (size_t i = 0; i < input.fields.size(); ++i) {
        const auto & f = input.fields[i];
        raw("composed/lexical/" + std::to_string(i), 0, { int64_t(f.options.size()), 2048 }, f.lexical_means);
        raw("composed/logits/" + std::to_string(i), 0, { int64_t(f.options.size()) }, output.fields[i].logits);
        raw("composed/probabilities/" + std::to_string(i), 0, { int64_t(f.options.size()) },
            output.fields[i].probabilities);
    }
}
} // namespace composed_test
} // namespace josev2b
#endif
static int run(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--gates") {
            std::string error;
            auto context = josev2b::composed_context::load_packed("missing-packed", 2, error);
            check(!context && error == "Josev packed/component gates are off", "off gate accessed model files");
            std::cout << "PASS packed gate rejects before model access\n";
            return 0;
        }
        check(argc == 4, "packed-model requests output-directory required");
        const std::filesystem::path root(argv[3]);
        check(!std::filesystem::exists(root), "fresh output directory required");
        std::filesystem::create_directories(root);
        const auto requests = core_decision::Parser(read(argv[2])).parse();
        check(requests.kind == core_decision::Value::array && requests.list.size() == 4,
              "exact four-case request packet required");
        std::string error;
        auto context = josev2b::composed_context::load_packed(argv[1], 2, error);
        if (!context) throw std::runtime_error(error);
        bool failed = false;
        std::ofstream report(root / "native.jsonl");
        for (size_t i = 0; i < requests.list.size(); ++i) {
            const auto & row = requests.list[i];
            check(row.get("id").kind == core_decision::Value::string &&
                      row.get("request_json").kind == core_decision::Value::string,
                  "request row invalid");
            directory = root / std::to_string(i);
            std::filesystem::create_directory(directory);
            stage_index = 0;
            stages.open(directory / "stages.jsonl");
            std::string output = "stale";
            check(!context->decide("{", output, error) && output.empty() && !error.empty(),
                  "malformed request did not clear stale response");
            josev2b::composed_result detailed;
            const bool ok = context->decide(row.get("request_json").text, detailed, error);
            output = detailed.response_json;
            if (ok) {
                for (size_t f = 0; f < detailed.native_fields.fields.size(); ++f) {
                    const auto & field = detailed.native_fields.fields[f];
                    raw("result/logits/" + std::to_string(f), 0, { int64_t(field.logits.size()) }, field.logits);
                    raw("result/probabilities/" + std::to_string(f), 0, { int64_t(field.probabilities.size()) },
                        field.probabilities);
                }
            }
            stages.close();
            report << "{\"id\":" << q(row.get("id").text) << ",\"completed\":" << (ok ? "true" : "false")
                   << ",\"error\":" << q(error) << ",\"response\":" << q(output) << "}\n";
            if (!ok) failed = true;
            check(bool(report), "native report failed");
        }
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
