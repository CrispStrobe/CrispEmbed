// Copied from tests/minicpm-graph/diff.cpp at83eac856; same actual Ref/fixed gates.
// Test-only actual crispembed-diff::Ref comparison of lossless F32 adapters.
#include "core/clean_exit.h"
#include "core/json.h"
#include "crispembed_diff.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

static void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
static void quoted(std::ostream & out, const std::string & s) {
    out << '"' << core_json::json_escape_strict(s) << '"';
}
struct magnitude {
    double lo = 0, hi = 0, abs_lo = 0, abs_hi = 0;
    magnitude(const float * x, size_t n) {
        require(n > 0, "Empty F32 tensor");
        lo = hi = x[0];
        abs_lo = abs_hi = std::abs(double(x[0]));
        for (size_t i = 0; i < n; ++i) {
            require(std::isfinite(x[i]), "Nonfinite F32 tensor");
            lo = std::min(lo, double(x[i]));
            hi = std::max(hi, double(x[i]));
            abs_lo = std::min(abs_lo, std::abs(double(x[i])));
            abs_hi = std::max(abs_hi, std::abs(double(x[i])));
        }
    }
};
static int body(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: unlimited-q4-diff ORIGINAL.gguf REPLAY.gguf OUTPUT.json\n");
        return 2;
    }
    try {
        crispembed_diff::Ref original, native;
        require(original.load(argv[1]) && native.load(argv[2]), "Ref load failed");
        auto names = original.tensor_names(), other = native.tensor_names();
        std::sort(names.begin(), names.end());
        std::sort(other.begin(), other.end());
        require(!names.empty() && names.size() <= 512 && names == other, "Exact adapter stage coverage required");
        std::ofstream out(argv[3]);
        require(bool(out), "Comparator report open failed");
        out << std::setprecision(17) << std::boolalpha;
        // Buffer rows so the final numeric status precedes the complete results.
        std::ostringstream rows;
        rows << std::setprecision(17) << std::boolalpha;
        bool failed = false;
        size_t index = 0;
        for (const auto & name : names) {
            const auto a = original.get_f32(name), b = native.get_f32(name);
            const auto dims = original.shape(name);
            require(a.first && b.first && a.second == b.second && !dims.empty() && dims == native.shape(name),
                    "Exact F32 shape/element count required");
            require(dims[0] > 0 && a.second % size_t(dims[0]) == 0, "Invalid contiguous row width");
            const magnitude x(a.first, a.second), y(b.first, b.second);
            double signed_error = 0;
            for (size_t i = 0; i < a.second; ++i) signed_error += double(b.first[i]) - double(a.first[i]);
            // row_dim is an AXIS INDEX: ne[0] is the contiguous C-order last axis.
            const auto r = original.compare(name, b.first, b.second, 0);
            require(r.found && r.n_elem == a.second, "Ref did not compare every element");
            for (float v : { r.max_abs, r.mean_abs, r.rms, r.cos_min, r.ref_norm, r.mine_norm })
                require(std::isfinite(v), "Nonfinite Ref metric");
            // Explicit fixed comparisons: Ref::is_pass's environment override is
            // intentionally not used, and no quantization tolerance applies.
            const bool passed = double(r.max_abs) <= 0.002 && double(r.cos_min) >= 0.99999;
            failed |= !passed;
            if (index++) rows << ',';
            rows << "{\"name\":";
            quoted(rows, name);
            rows << ",\"ggml_shape\":[";
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i) rows << ',';
                rows << dims[i];
            }
            rows << "],\"row_axis\":0,\"found\":" << r.found << ",\"elements\":" << r.n_elem << ",\"passed\":" << passed
                 << ",\"max_absolute\":" << r.max_abs << ",\"mean_absolute\":" << r.mean_abs << ",\"rms\":" << r.rms
                 << ",\"mean_error\":" << signed_error / double(a.second) << ",\"min_row_cosine\":" << r.cos_min
                 << ",\"original_l2\":" << r.ref_norm << ",\"native_l2\":" << r.mine_norm
                 << ",\"original_min\":" << x.lo << ",\"original_max\":" << x.hi << ",\"original_abs_min\":" << x.abs_lo
                 << ",\"original_abs_max\":" << x.abs_hi << ",\"native_min\":" << y.lo << ",\"native_max\":" << y.hi
                 << ",\"native_abs_min\":" << y.abs_lo << ",\"native_abs_max\":" << y.abs_hi << '}';
            std::printf("%s max_abs=%.9g mean_abs=%.9g rms=%.9g cos_min=%.9g ref_l2=%.9g native_l2=%.9g %s\n",
                        name.c_str(), r.max_abs, r.mean_abs, r.rms, r.cos_min, r.ref_norm, r.mine_norm,
                        passed ? "PASS" : "FAIL");
        }
        out << "{\"schema\":\"crispembed-unlimited-q4-ref-v1\",\"status\":\""
            << (failed ? "numeric_rejection" : "passed")
            << "\",\"max_absolute_budget\":0.002,\"minimum_row_cosine\":0.99999,\"stages\":[" << rows.str() << "]}\n";
        require(bool(out), "Comparator report write failed");
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::ofstream out(argv[3]);
        out << "{\"status\":\"structural_rejection\",\"error\":";
        quoted(out, e.what());
        out << "}\n";
        std::fprintf(stderr, "Ref structural rejection: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
