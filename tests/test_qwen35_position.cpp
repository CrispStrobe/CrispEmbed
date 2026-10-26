// Test-only replay of the actual position implementation and patch projection.
#include "core/clean_exit.h"
#include "qwen35_vision.h"
#include "qwen35_math.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include <cstring>
#define vision_cpu position_audit_vision_cpu
#define map_small_vision position_audit_map_small_vision
#include "../src/qwen35_vision.cpp"
#undef map_small_vision
#undef vision_cpu

static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION", value);
#else
    setenv("CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION", value, 1);
#endif
}
static bool exact(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
static bool include(const char * name, void *) {
    return std::string(name).find("v.") == 0;
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    crispembed_diff::Ref ref;
    if (!ref.load(argv[1])) return 2;
    auto array = [&](const std::string & name) {
        auto v = ref.get_f32(name);
        return v.first ? std::vector<float>(v.first, v.first + v.second) : std::vector<float>{};
    };
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad weights;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "position-probe", weights))
        return 2;
    auto table = array("v.pos_embed.weight");
    qwen35_ocr::vision_config c;
    if (table.size() != size_t(48 * 48 * 768)) return 2;
    bool ok = true;
    int legacy_differences = 0;
    for (int case_id = 0; case_id < 5; ++case_id) {
        const std::string prefix = "case." + std::to_string(case_id);
        auto grid_values = array(prefix + ".grid");
        std::vector<int> grid;
        for (float x : grid_values) {
            if (!std::isfinite(x) || x < 1 || x > 1024 || std::floor(x) != x) return 2;
            grid.push_back(int(x));
        }
        if (grid.size() != 3) return 2;
        const int tokens = grid[0] * grid[1] * grid[2];
        std::vector<int> rows, columns, offsets, heights, widths;
        if (!qwen35_ocr::geometry(c, tokens, grid, rows, columns, offsets, heights, widths)) return 2;
        std::vector<float> baseline;
        for (const char * mode : { "unset", "0", "1" }) {
            if (std::string(mode) == "unset") {
#ifdef _WIN32
                _putenv_s("CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION", "");
#else
                unsetenv("CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION");
#endif
            } else
                gate(mode);
            const bool source = core_env::on("CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION");
            std::vector<float> position, indices, coefficients, rvalues, cvalues;
            if (!qwen35_ocr::interpolate_position(table, c, rows, columns, heights, widths, position)) return 2;
            for (int t = 0; t < tokens; ++t) {
                auto taps = qwen35_ocr::interpolation_taps(rows[t], columns[t], heights[t], widths[t], 48, source);
                for (int i = 0; i < 4; ++i) {
                    indices.push_back(float(taps.index[i]));
                    coefficients.push_back(taps.weight[i]);
                }
                rvalues.push_back(qwen35_ocr::position_coordinate(rows[t], heights[t], 48, source));
                cvalues.push_back(qwen35_ocr::position_coordinate(columns[t], widths[t], 48, source));
            }
            std::printf("MODE %s %s\n", prefix.c_str(), mode);
            if (!source) {
                if (std::string(mode) == "unset")
                    baseline = position;
                else {
                    const bool same = exact(position, baseline);
                    ok &= same;
                    std::printf("%s literal_zero_bit_exact\n", same ? "PASS" : "FAIL");
                }
                if (std::string(mode) == "unset")
                    legacy_differences += !exact(coefficients, array(prefix + ".weights"));
                if (case_id == 0) {
                    const bool same = exact(position, array("legacy.position"));
                    ok &= same;
                    std::printf("%s retained_legacy_position_bit_exact\n", same ? "PASS" : "FAIL");
                }
                qwen35_compare(ref, prefix + ".position", position);
            } else {
                for (auto row : { std::make_pair("indices", &indices), std::make_pair("weights", &coefficients),
                                  std::make_pair("row_src", &rvalues), std::make_pair("col_src", &cvalues) }) {
                    const bool same = exact(*row.second, array(prefix + "." + row.first));
                    ok &= same;
                    std::printf("%s source_%s_bit_exact\n", same ? "PASS" : "FAIL", row.first);
                    ok &= qwen35_compare(ref, prefix + "." + row.first, *row.second);
                }
                ok &= qwen35_compare(ref, prefix + ".position", position);
            }
        }
    }
    auto pixels = array("input.pixel_values");
    std::vector<float> patch;
    auto get = [&](const char * name) { return core_gguf::try_get(weights.tensors, name); };
    if (pixels.size() != size_t(2048 * 1536) ||
        !qwen35_ocr::linear(get("v.patch_embed.proj.weight"), get("v.patch_embed.proj.bias"), pixels, 1536, 2048, 2,
                            patch))
        return 2;
    const bool patch_same = exact(patch, array("legacy.patch"));
    ok &= patch_same;
    std::printf("%s retained_legacy_patch_bit_exact\n", patch_same ? "PASS" : "FAIL");
    ok &= qwen35_compare(ref, "frozen.patch", patch);
    ok &= qwen35_compare(ref, "local.patch", patch);
    std::printf("LEGACY differing_coefficient_cases=%d\n", legacy_differences);
    ok &= legacy_differences > 0;
    core_gguf::free_weights(weights);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
