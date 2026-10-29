// TEST EXECUTABLE ONLY: actual production helper definitions, not a replacement engine.
#include "../../src/moedern_text_engine.cpp"
#include "core/clean_exit.h"
#include "core/json.h"
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;
namespace native = moedern_text_detail;
namespace {
void ensure(bool okay, const char * why) {
    if (!okay) throw std::runtime_error(why);
}
struct counts {
    int linear = 0, softmax_rows = 0, selection_rows_started = 0, selection_rows_completed = 0;
};
void write_counts(const fs::path & out, const counts & c, const std::string & outcome, const std::string & error = {}) {
    std::ofstream f(out / "result.json");
    f << "{\"outcome\":\"" << outcome << "\",\"error\":\"" << core_json::json_escape_strict(error)
      << "\",\"linear_batches\":" << c.linear << ",\"softmax_rows\":" << c.softmax_rows
      << ",\"selection_rows_started\":" << c.selection_rows_started
      << ",\"selection_rows_completed\":" << c.selection_rows_completed
      << ",\"scalar_divisions_completed\":" << c.selection_rows_completed * 2 << ",\"full_model_forwards\":0}\n";
    ensure(bool(f), "Result receipt write failed");
}
template <class T> void save(const fs::path & out, const std::vector<T> & x) {
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char *>(x.data()), std::streamsize(x.size() * sizeof(T)));
    ensure(bool(f), "Array write failed");
}
void validate(const native::floats & x, const std::string & mode) {
    ensure(mode == "input" || mode == "logits" || mode == "probs", "Component mode rejected");
    ensure(x.size() == size_t(307 * (mode == "input" ? 128 : 8)) && native::finite(x),
           "Exact finite full307-row component input required");
    if (mode == "probs")
        for (float v : x) ensure(v >= 0 && v <= 1, "Captured probability outside unit interval");
}
native::floats weight(native::context * ctx, int layer) {
    auto * w = ctx->get("blk." + std::to_string(layer) + ".ffn_gate_inp.weight");
    ensure(w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == 128 && w->ne[1] == 8 && w->ne[2] == 1 &&
               w->ne[3] == 1,
           "Actual loaded router weight layout differs");
    native::floats values(128 * 8);
    ggml_backend_tensor_get(w, values.data(), 0, values.size() * sizeof(float));
    ensure(native::finite(values), "Nonfinite loaded router");
    return values;
}
void select(const native::floats & source, const fs::path & out, counts & c) {
    using namespace moedern_text_detail;
    const int S = 307;
    auto probs = source;
    floats top_weights(size_t(S) * TOP), margins(S);
    std::vector<int64_t> chosen(size_t(S) * TOP);
    for (int t = 0; t < S; ++t) {
        auto * p = probs.data() + size_t(t) * EXPERTS;
        ++c.selection_rows_started;
// This exact source fragment is extracted and hash-checked before the hosted build.
#include "native_router.inc"
        ++c.selection_rows_completed;
    }
    save(out / "indices.i64", chosen);
    save(out / "weights.f32", top_weights);
    // Diagnostic gather only; native raw selected values were not retained original gold.
    floats gathered(size_t(S) * TOP);
    for (int t = 0; t < S; ++t)
        for (int k = 0; k < TOP; ++k)
            gathered[size_t(t) * TOP + k] = probs[size_t(t) * EXPERTS + chosen[size_t(t) * TOP + k]];
    save(out / "values.f32", gathered);
}
int run(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--contracts") {
        int rejected = 0;
        for (const auto & x : { native::floats{}, native::floats(307 * 128, std::numeric_limits<float>::quiet_NaN()) })
            try {
                validate(x, "input");
            } catch (const std::exception &) {
                ++rejected;
            }
        ensure(rejected == 2, "Invalid inputs admitted before component math");
        std::puts("PASS two invalid inputs; zero model loads/linear/softmax/selection calls");
        return 0;
    }
    const bool weights_only = argc == 4 && std::string(argv[1]) == "--weights";
    ensure(weights_only || argc == 7, "Expected MODEL LAYER MODE INPUT OUTPUT THREADS or --weights MODEL OUTPUT");
    const fs::path out(weights_only ? argv[3] : argv[5]);
    ensure(!fs::exists(out), "Fresh component output required");
    fs::create_directories(out);
    counts counter;
    try {
        const int threads = weights_only ? 2 : std::stoi(argv[6]);
        ensure(threads == 1 || threads == 2, "One/two CPU threads required");
        native::floats x;
        std::string mode;
        int layer = 0;
        if (!weights_only) {
            layer = std::stoi(argv[2]);
            ensure(layer == 0 || layer == 1, "Two captured layers only");
            mode = argv[3];
            std::ifstream f(argv[4], std::ios::binary | std::ios::ate);
            const auto extent = static_cast<std::streamoff>(f.tellg());
            ensure(bool(f) && extent >= 0 && extent <= 307 * 128 * 4 && extent % 4 == 0, "Input extent rejected");
            x.resize(size_t(extent) / 4);
            f.seekg(0);
            f.read(reinterpret_cast<char *>(x.data()), std::streamsize(x.size() * 4));
            ensure(bool(f), "Input read failed");
            validate(x, mode);
        }
        std::string error;
        std::unique_ptr<native::context, decltype(&native::free)> ctx(
            native::init(weights_only ? argv[2] : argv[1], threads, error), native::free);
        ensure(bool(ctx), error.c_str());
        if (weights_only) {
            for (int i = 0; i < 2; ++i) save(out / ("weight" + std::to_string(i) + ".f32"), weight(ctx.get(), i));
        } else {
            save(out / "weight.f32", weight(ctx.get(), layer));
            auto probs = x;
            if (mode == "input") {
                ++counter.linear;
                probs = native::linear(ctx->get("blk." + std::to_string(layer) + ".ffn_gate_inp.weight"), x, threads);
                save(out / "logits.f32", probs);
            }
            if (mode != "probs") {
                for (int row = 0; row < 307; ++row) {
                    ++counter.softmax_rows;
                    native::softmax(probs.data() + row * 8, 8);
                }
                save(out / "probs.f32", probs);
            }
            select(probs, out, counter);
        }
        write_counts(out, counter, "completed");
        return 0;
    } catch (const std::exception & error) {
        const bool tie = std::string(error.what()) == "Exact router tie is unqualified";
        write_counts(out, counter, tie ? "native_exact_tie_rejection" : "structural_rejection", error.what());
        std::cerr << error.what() << '\n';
        return 2;
    }
}
} // namespace
int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
    }
    core_util::clean_exit(code);
}
