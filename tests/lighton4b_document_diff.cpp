#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <set>
#include <string>

// Diagnostic archives only. This binary never loads weights or runs inference.
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 4) return 2;
    for (int i = 1; i <= 2; ++i) {
        std::error_code error;
        const auto bytes = std::filesystem::file_size(argv[i], error);
        if (error || bytes > 2ULL * 1024 * 1024 * 1024) return 2;
    }
    crispembed_diff::Ref original, candidate;
    if (!original.load(argv[1]) || !candidate.load(argv[2])) return 2;
    std::ifstream order(argv[3]);
    std::set<std::string> names;
    std::string name, kind, first;
    int64_t axes[4];
    size_t count = 0;
    bool all = true;
    while (order >> name >> kind >> axes[0] >> axes[1] >> axes[2] >> axes[3]) {
        if (++count > 4096 || !names.insert(name).second || (kind != "exact" && kind != "float")) return 2;
        size_t expected = 1;
        for (int i = 0; i < 4; ++i) {
            if (axes[i] < 1 || axes[i] > 64 * 1024 * 1024 || expected > (64 * 1024 * 1024) / size_t(axes[i])) return 2;
            expected *= size_t(axes[i]);
        }
        auto reference = original.get_f32(name), native = candidate.get_f32(name);
        auto shape = original.shape(name);
        if (shape.size() > 4) return 2;
        shape.resize(4, 1);
        bool finite = reference.first && native.first && reference.second == native.second && reference.second > 0 &&
                      reference.second == expected && original.shape(name) == candidate.shape(name) &&
                      std::equal(shape.begin(), shape.end(), axes);
        if (finite)
            for (size_t i = 0; i < reference.second; ++i)
                finite &= std::isfinite(reference.first[i]) && std::isfinite(native.first[i]);
        const auto result = finite ? original.compare(name, native.first, native.second) : crispembed_diff::Report{};
        // Fixed raw metrics; Report::is_pass() permits an environment override
        // and is deliberately not used by this frozen protocol.
        bool passed = finite && result.found && result.n_elem == reference.second && result.max_abs <= 0.002f &&
                      result.cos_min >= 0.99999f;
        if (kind == "exact" && finite)
            passed &= std::memcmp(reference.first, native.first, reference.second * sizeof(float)) == 0;
        if (!passed && first.empty()) first = name;
        all &= passed;
        std::printf("STAGE %s passed=%d finite=%d count=%zu max_abs=%.9g mean_abs=%.9g rms=%.9g "
                    "row_cos=%.9g ref_norm=%.9g native_norm=%.9g\n",
                    name.c_str(), passed ? 1 : 0, finite ? 1 : 0, result.n_elem, result.max_abs, result.mean_abs,
                    result.rms, result.cos_min, result.ref_norm, result.mine_norm);
    }
    const auto a = original.tensor_names(), b = candidate.tensor_names();
    if (!order.eof() || !count || std::set<std::string>(a.begin(), a.end()) != names ||
        std::set<std::string>(b.begin(), b.end()) != names)
        return 2;
    std::printf("DOCUMENT_DIFF passed=%d stages=%zu first=%s inference_executed=0\n", all ? 1 : 0, count,
                first.empty() ? "none" : first.c_str());
    return all ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
