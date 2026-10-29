// Engine-only test executable. Original fixtures are never injected into encoder forwards.
#include "gemma_decision.h"
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include "core/decision_schema.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static std::string read(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "Input open failed");
    std::string data((std::istreambuf_iterator<char>(f)), {});
    require(data.size() <= 16 * 1024 * 1024, "Bounded text manifest required");
    return data;
}
static void bytes(const std::string & path, const void * data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    require(bool(f), "Output open failed");
    f.write(static_cast<const char *>(data), n);
    f.close();
    require(bool(f), "Output write failed");
}
static std::vector<float> f32(const std::string & path, size_t n) {
    std::vector<float> data(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(data.data()), n * 4);
    require(bool(f) && f.peek() == std::char_traits<char>::eof(), "Exact F32 extent required");
    for (float v : data) require(std::isfinite(v), "Nonfinite F32 input");
    return data;
}
static std::vector<std::string> stages() {
    std::vector<std::string> result{ "post_embed", "ple" };
    for (int i = 0; i < 24; ++i) result.push_back("layer_" + std::to_string(i));
    result.insert(result.end(), { "final_norm", "tokens", "embedding" });
    return result;
}
static bool compare(crispembed_diff::Ref & reference, const std::string & name, const float * x, size_t n,
                    const std::vector<int64_t> & shape, std::ostream & out) {
    const auto target = reference.get_f32(name);
    require(x && target.first && n > 0 && target.second == n && reference.shape(name) == shape,
            "Strict reference/native shape required");
    const auto r = reference.compare(name, x, n, 0);
    require(r.found && r.n_elem == n, "Actual Ref must compare every element");
    double square = 0, signed_error = 0;
    float original_min = target.first[0], original_max = original_min, native_min = x[0], native_max = x[0];
    for (size_t i = 0; i < n; ++i) {
        require(std::isfinite(x[i]) && std::isfinite(target.first[i]), "Nonfinite stage value");
        const double d = double(x[i]) - target.first[i];
        square += d * d;
        signed_error += d;
        original_min = std::min(original_min, target.first[i]);
        original_max = std::max(original_max, target.first[i]);
        native_min = std::min(native_min, x[i]);
        native_max = std::max(native_max, x[i]);
    }
    const double relative = std::sqrt(square) / std::max(double(r.ref_norm), 1e-12);
    const bool passed = r.cos_min >= 0.99999f && relative <= 0.003;
    out << "{\"name\":" << core_decision::quote(name) << ",\"ggml_shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) out << (i ? "," : "") << shape[i];
    out << "],\"elements\":" << n << ",\"row_axis\":0,\"cosine_min\":" << r.cos_min << ",\"relative_l2\":" << relative
        << ",\"max_absolute\":" << r.max_abs << ",\"mean_absolute\":" << r.mean_abs << ",\"rms\":" << r.rms
        << ",\"mean_error\":" << signed_error / n << ",\"original_l2\":" << r.ref_norm
        << ",\"native_l2\":" << r.mine_norm << ",\"original_min\":" << original_min
        << ",\"original_max\":" << original_max << ",\"native_min\":" << native_min << ",\"native_max\":" << native_max
        << ",\"passed\":" << passed << '}';
    return passed;
}
static int stage(const char * model, const char * ref_path, const std::string & out) {
    crispembed_diff::Ref original;
    require(original.load(ref_path), "Original reference load");
    auto expected = stages();
    auto names = original.tensor_names();
    names.erase(std::remove(names.begin(), names.end(), "input_ids"), names.end());
    auto sorted = expected;
    std::sort(names.begin(), names.end());
    std::sort(sorted.begin(), sorted.end());
    require(names == sorted && original.has("input_ids"), "Exact five-probe stage inventory required");
    auto ids_ref = original.get_f32("input_ids");
    require(original.shape("input_ids") == std::vector<int64_t>({ int64_t(ids_ref.second), 1 }), "Batch1 IDs required");
    std::unique_ptr<gemma_decision_context, decltype(&gemma_decision_free)> ctx(gemma_decision_init(model, 2),
                                                                                gemma_decision_free);
    require(bool(ctx), "Actual Gemma engine init");
    std::vector<float> embedding, ordinary;
    std::vector<int32_t> ids;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const bool observed = repeat == 1;
        if (observed) {
            setenv("CRISPEMBED_GEMMA_DECISION_DUMP", "1", 1);
            setenv("CRISPEMBED_GEMMA_DECISION_DUMP_PATH", (out + "/stages.gguf").c_str(), 1);
        } else {
            unsetenv("CRISPEMBED_GEMMA_DECISION_DUMP");
            unsetenv("CRISPEMBED_GEMMA_DECISION_DUMP_PATH");
        }
        require(gemma_decision_encode(ctx.get(), original.meta("ref.text"), embedding, &ids), "Actual encode failed");
        require(embedding.size() == 768 && ids.size() == ids_ref.second, "Exact output/ID axes");
        for (size_t i = 0; i < ids.size(); ++i) require(ids[i] == ids_ref.first[i], "Actual token ID differs");
        for (float v : embedding) require(std::isfinite(v), "Nonfinite pooled embedding");
        bytes(out + "/embedding-" + std::to_string(repeat) + ".f32", embedding.data(), embedding.size() * 4);
        bytes(out + "/ids-" + std::to_string(repeat) + ".i32", ids.data(), ids.size() * 4);
        if (repeat == 0) ordinary = embedding;
        require(std::memcmp(ordinary.data(), embedding.data(), embedding.size() * 4) == 0,
                "Observer/repeat byte neutrality");
    }
    crispembed_diff::Ref native;
    require(native.load(out + "/stages.gguf"), "Actual native dump load");
    auto native_names = native.tensor_names();
    auto captured_names = stages();
    captured_names.pop_back();
    std::sort(native_names.begin(), native_names.end());
    std::sort(captured_names.begin(), captured_names.end());
    require(native_names == captured_names, "All28 native stage captures required");
    std::ofstream report(out + "/stage.json");
    report << std::setprecision(17) << std::boolalpha
           << "{\"completed\":true,\"encoder_forwards\":3,\"ids\":" << ids.size()
           << ",\"observer_repeat_bit_exact\":true,\"stages\":[";
    bool passed = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto & name = expected[i];
        if (i) report << ',';
        const auto values =
            name == "embedding" ? std::make_pair(embedding.data(), embedding.size()) : native.get_f32(name);
        const auto shape = name == "embedding" ? std::vector<int64_t>({ 768, 1 }) : native.shape(name);
        passed &= compare(original, name, values.first, values.second, shape, report);
    }
    report << "],\"strict_f32_passed\":" << passed << "}\n";
    report.close();
    require(bool(report), "Stage report write");
    return passed ? 0 : 1;
}
static int compare_only(const char * original_path, const char * native_path, const std::string & out) {
    crispembed_diff::Ref original, native;
    require(original.load(original_path) && native.load(native_path), "Lossless comparison adapters");
    auto names = original.tensor_names(), other = native.tensor_names();
    std::sort(names.begin(), names.end());
    std::sort(other.begin(), other.end());
    require(!names.empty() && names.size() <= 64 && names == other, "Exact adapter name coverage");
    std::ofstream report(out + "/comparison.json");
    report << std::setprecision(17) << std::boolalpha << "{\"completed\":true,\"stages\":[";
    bool passed = true;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) report << ',';
        const auto data = native.get_f32(names[i]);
        passed &= compare(original, names[i], data.first, data.second, native.shape(names[i]), report);
    }
    report << "],\"strict_f32_passed\":" << passed << "}\n";
    report.close();
    require(bool(report), "Comparison write");
    return passed ? 0 : 1;
}
static int typed(const char * model, const char * manifest, const std::string & out) {
    using namespace core_decision;
    const auto plan = Parser(read(manifest)).parse();
    require(plan.kind == Value::array && plan.list.size() == 3, "Three frozen typed scenarios");
    std::unique_ptr<gemma_decision_context, decltype(&gemma_decision_free)> ctx(gemma_decision_init(model, 2),
                                                                                gemma_decision_free);
    require(bool(ctx), "Actual typed engine init");
    for (const char * request :
         { "{\"state\":\"state\",\"questions\":{}}", "{\"state\":\"state\",\"questions\":{\"bad\":{\"criteria\":[]}}}",
           "{\"state\":\"state\",\"questions\":{\"bad\":\"not an object\"}}" }) {
        require(gemma_decision_run(ctx.get(), request).empty() && *gemma_decision_error(ctx.get()),
                "Invalid schema accepted");
    }
    std::ofstream report(out + "/typed.json");
    report << std::setprecision(17) << std::boolalpha << "{\"completed\":true,\"scenarios\":[";
    int bank_calls = 0, head_calls = 0, decision_calls = 0;
    for (size_t i = 0; i < plan.list.size(); ++i) {
        const auto & row = plan.list[i];
        const auto & bank = row.get("bank").list;
        require(bank.size() == (i == 0 ? 13 : i == 1 ? 11 : 6), "Frozen bank count");
        std::vector<float> pooled;
        for (size_t b = 0; b < bank.size(); ++b) {
            std::vector<float> embedding;
            std::vector<int32_t> ids;
            ++bank_calls;
            require(gemma_decision_encode(ctx.get(), read(bank[b].text), embedding, &ids) && embedding.size() == 768,
                    "Bank encode");
            pooled.insert(pooled.end(), embedding.begin(), embedding.end());
            bytes(out + "/bank-" + std::to_string(i) + "-" + std::to_string(b) + ".ids.i32", ids.data(),
                  ids.size() * 4);
        }
        bytes(out + "/bank-" + std::to_string(i) + ".f32", pooled.data(), pooled.size() * 4);
        for (size_t q = 0; q < row.get("heads").list.size(); ++q) {
            const auto & head = row.get("heads").list[q];
            const int n = std::stoi(head.get("options").text), kind = std::stoi(head.get("kind").text);
            require(n >= 1 && n <= 128 && kind >= 0 && kind <= 2, "Head control axes");
            auto state = f32(head.get("state").text, 768), question = f32(head.get("question").text, 768);
            auto options = f32(head.get("option_embeddings").text, size_t(n) * 768);
            std::vector<float> previous;
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::vector<float> values(size_t(n) * 2);
                ++head_calls;
                require(gemma_decision_scores(ctx.get(), state.data(), question.data(), options.data(), n, kind,
                                              values.data(), values.data() + n),
                        "Head-on-original control");
                for (float value : values) require(std::isfinite(value), "Nonfinite head control");
                if (repeat == 0) previous = values;
                require(std::memcmp(previous.data(), values.data(), values.size() * 4) == 0, "Head repeat neutrality");
                bytes(out + "/head-" + std::to_string(i) + "-" + std::to_string(q) + "-" + std::to_string(repeat) +
                          ".f32",
                      values.data(), values.size() * 4);
            }
        }
        const std::string request = read(row.get("request").text);
        const auto answer = gemma_decision_run(ctx.get(), request);
        ++decision_calls;
        bytes(out + "/answer-" + std::to_string(i) + ".json", answer.data(), answer.size());
        if (i == 0) {
            const auto repeat = gemma_decision_run(ctx.get(), request);
            ++decision_calls;
            bytes(out + "/answer-repeat.json", repeat.data(), repeat.size());
            require(answer == repeat, "Typed ordinary repeat bytes differ");
        }
        report << (i ? "," : "") << "{\"scenario\":" << i << ",\"bank\":" << bank.size()
               << ",\"answer_nonempty\":" << !answer.empty() << ",\"error\":" << quote(gemma_decision_error(ctx.get()))
               << '}';
    }
    report << "],\"explicit_bank_forwards\":" << bank_calls << ",\"head_on_original_calls\":" << head_calls
           << ",\"decision_requests\":" << decision_calls
           << ",\"invalid_requests\":3,\"decision_internal_forwards_source_derived\":43}\n";
    report.close();
    require(bool(report) && bank_calls == 30 && head_calls == 16 && decision_calls == 4,
            "Complete typed control calls");
    return 0;
}
static int body(int argc, char ** argv) {
    try {
        require(argc == 5, "Expected stages|typed|compare MODEL_OR_REFERENCE INPUT OUTPUT_DIRECTORY");
        if (std::string(argv[1]) == "compare") return compare_only(argv[2], argv[3], argv[4]);
        setenv("CRISPEMBED_GEMMA_DECISION", "1", 1);
        setenv("CRISPEMBED_GEMMA_DECISION_F32_DOT", "1", 1);
        require(!std::getenv("CRISPEMBED_GEMMA_DECISION_LEGACY_GELU"), "Legacy GELU not in frozen inference blueprint");
        if (std::string(argv[1]) == "stages") return stage(argv[2], argv[3], argv[4]);
        require(std::string(argv[1]) == "typed", "Declared engine-only probe mode");
        return typed(argv[2], argv[3], argv[4]);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Gemma Q4 engine probe: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
