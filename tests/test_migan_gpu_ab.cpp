// MI-GAN device/precision A/B against the reviewed CPU/F32 graph. Test-only;
// links the observer archive and compiles the tiling wrapper directly.
//
//   test-migan-gpu-ab stages REF_F32.gguf CAND.gguf cpu|gpu RGB.raw MASK.raw ORIG.raw|- ITERS THREADS
//     Every observed stage of CAND (device) vs the CPU/F32 reference, the
//     frozen 0.002/0.99999 float gates, observer neutrality, decoded image
//     gates vs CPU/F32 and vs the original composite, output cosine, timing.
//   test-migan-gpu-ab frame REF_F32.gguf CAND.gguf cpu|gpu WIDTH HEIGHT ITERS THREADS KIND OUT.rgb8
//     Synthetic subtitle frame through inpaint::process_rgb8 (tiles+batch):
//     CAND vs CPU/F32 pixel deltas, warm per-frame time, tile throughput.
#include "inpaint.h"
#include "migan.h"
#include "migan_test_observer.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char * checkpoint = "8b82b2e82fc8e5a2e1f06827594aac1a5d66a9ca41e24199ddd969847788097f";
constexpr const char * source_gguf = "b8fcefc3de7e04f96867b9d15bd84a2cf7e0c444d2925644a7e8b15081809c8b";

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

bool bytes(const char * path, std::vector<uint8_t> & data, size_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streamoff(expected)) return false;
    data.resize(expected);
    stream.seekg(0);
    return bool(stream.read(reinterpret_cast<char *>(data.data()), expected));
}

struct stats {
    double max_abs = 0, mae = 0, rms = 0, cos_min = 1, cos_global = 1;
};

// Same definitions as crispembed_diff::Ref::compare: rows are ne[0] vectors.
stats compare(const float * a, const float * b, size_t count, size_t row) {
    stats s;
    double sum = 0, sq = 0, dot = 0, na = 0, nb = 0;
    double worst = 2;
    for (size_t r = 0; r * row < count; ++r) {
        double rd = 0, ra = 0, rb = 0;
        for (size_t j = r * row; j < std::min(count, (r + 1) * row); ++j) {
            const double d = std::fabs(double(a[j]) - b[j]);
            s.max_abs = std::max(s.max_abs, d);
            sum += d;
            sq += d * d;
            rd += double(a[j]) * b[j];
            ra += double(a[j]) * a[j];
            rb += double(b[j]) * b[j];
        }
        const double c = ra > 1e-18 && rb > 1e-18 ? rd / std::sqrt(ra * rb) : (ra <= 1e-18 && rb <= 1e-18 ? 1 : 0);
        worst = std::min(worst, c);
        dot += rd;
        na += ra;
        nb += rb;
    }
    s.mae = sum / count;
    s.rms = std::sqrt(sq / count);
    s.cos_min = worst;
    s.cos_global = na > 1e-18 && nb > 1e-18 ? dot / std::sqrt(na * nb) : (na <= 1e-18 && nb <= 1e-18 ? 1 : 0);
    return s;
}

struct image_stats {
    size_t fill = 0, known_errors = 0;
    int max_abs = 0;
    double mae = 0, mse = 0, psnr = std::numeric_limits<double>::infinity(), cos_centered = 1;
    bool passed = false;
};

// Existing image gate: known bytes exact; fill max1/MAE0.1/PSNR50dB.
image_stats image_metrics(const uint8_t * actual, const uint8_t * reference, const uint8_t * input,
                          const uint8_t * fill, size_t pixels) {
    image_stats s;
    double sum = 0, sq = 0, dot = 0, na = 0, nb = 0;
    for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < 3; ++c) {
            const size_t i = 3 * p + c;
            if (!fill[p]) {
                if (actual[i] != input[i] || reference[i] != input[i]) ++s.known_errors;
                continue;
            }
            const int d = std::abs(int(actual[i]) - int(reference[i]));
            s.max_abs = std::max(s.max_abs, d);
            sum += d;
            sq += double(d) * d;
            const double x = actual[i] - 127.5, y = reference[i] - 127.5;
            dot += x * y;
            na += x * x;
            nb += y * y;
            ++s.fill;
        }
    if (s.fill) {
        s.mae = sum / s.fill;
        s.mse = sq / s.fill;
        if (s.mse > 0) s.psnr = 10 * std::log10(255.0 * 255.0 / s.mse);
        s.cos_centered = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 1;
    }
    s.passed = s.known_errors == 0 && (!s.fill || (s.max_abs <= 1 && s.mae <= 0.1 && s.psnr >= 50.0));
    return s;
}

void print_image(const char * label, const image_stats & s) {
    std::printf("MIGAN_AB_IMAGE {\"against\":\"%s\",\"fill_channels\":%zu,\"max_abs\":%d,\"mae\":%.9g,\"mse\":%.9g,"
                "\"psnr_db\":",
                label, s.fill, s.max_abs, s.mae, s.mse);
    if (std::isfinite(s.psnr))
        std::printf("%.6f", s.psnr);
    else
        std::printf("null");
    std::printf(",\"exact_fill\":%s,\"fill_cos_centered\":%.9f,\"known_errors\":%zu,\"passed\":%s}\n",
                s.mse == 0 ? "true" : "false", s.cos_centered, s.known_errors, s.passed ? "true" : "false");
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

migan::device_choice parse_device(const char * value) {
    return std::strcmp(value, "gpu") == 0 ? migan::device_choice::gpu : migan::device_choice::cpu;
}

int stages(int argc, char ** argv) {
    if (argc != 10) return 2;
    const int iters = std::atoi(argv[8]), threads = std::atoi(argv[9]);
    std::string error;
    std::vector<uint8_t> rgb, mask, original;
    if (!bytes(argv[5], rgb, migan::rgb_elements) || !bytes(argv[6], mask, migan::mask_elements)) return 2;
    const bool have_original = std::strcmp(argv[7], "-") != 0;
    if (have_original && !bytes(argv[7], original, migan::rgb_elements)) return 2;
    std::unique_ptr<migan::context, decltype(&migan::free)> ref(
        migan::init_ex(argv[2], checkpoint, threads, migan::device_choice::cpu, source_gguf, error), migan::free);
    if (!ref) return std::fprintf(stderr, "reference: %s\n", error.c_str()), 2;
    std::unique_ptr<migan::context, decltype(&migan::free)> cand(
        migan::init_ex(argv[3], checkpoint, threads, parse_device(argv[4]), source_gguf, error), migan::free);
    if (!cand) return std::fprintf(stderr, "candidate: %s\n", error.c_str()), 2;
    std::vector<float> input(migan::input_elements), ref_raw(migan::rgb_elements), cand_raw(migan::rgb_elements),
        observed(migan::rgb_elements);
    if (!migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 256, 256, input.data(), input.size(),
                                error))
        return 2;

    std::map<std::string, std::vector<float>> reference;
    std::map<std::string, size_t> rows;
    std::vector<std::string> order;
    auto keep = [&](const std::string & stage, const float * data, size_t count, const std::array<int64_t, 4> & shape) {
        if (reference.count(stage)) return true;
        reference[stage].assign(data, data + count);
        rows[stage] = size_t(shape[0]);
        order.push_back(stage);
        return true;
    };
    size_t buffer = 0;
    if (!migan::test::run_observed(ref.get(), input.data(), input.size(), ref_raw.data(), ref_raw.size(), keep, buffer,
                                   error))
        return std::fprintf(stderr, "reference observe: %s\n", error.c_str()), 2;

    size_t compared = 0, failed = 0;
    std::string first_failure;
    double worst_max = 0, worst_cos = 1;
    std::map<std::string, bool> seen;
    auto check = [&](const std::string & stage, const float * data, size_t count, const std::array<int64_t, 4> &) {
        if (seen[stage]) return true;
        seen[stage] = true;
        auto it = reference.find(stage);
        if (it == reference.end() || it->second.size() != count) {
            std::fprintf(stderr, "stage mismatch: %s\n", stage.c_str());
            return false;
        }
        const bool finite = std::all_of(data, data + count, [](float v) { return std::isfinite(v); });
        const stats s = compare(data, it->second.data(), count, rows[stage]);
        const bool pass = finite && s.max_abs <= 0.002 && s.cos_min >= 0.99999 && s.cos_global >= 0.99999;
        ++compared;
        worst_max = std::max(worst_max, s.max_abs);
        worst_cos = std::min(worst_cos, s.cos_min);
        if (!pass) {
            ++failed;
            if (first_failure.empty()) first_failure = stage;
        }
        std::printf("MIGAN_AB_DIFF {\"stage\":\"%s\",\"count\":%zu,\"max_abs\":%.9g,\"mae\":%.9g,\"rms\":%.9g,"
                    "\"cos_min\":%.12f,\"cos_global\":%.12f,\"finite\":%s,\"strict_passed\":%s}\n",
                    stage.c_str(), count, s.max_abs, s.mae, s.rms, s.cos_min, s.cos_global, finite ? "true" : "false",
                    pass ? "true" : "false");
        return true;
    };
    if (!migan::run(cand.get(), input.data(), input.size(), cand_raw.data(), cand_raw.size(), error))
        return std::fprintf(stderr, "candidate run: %s\n", error.c_str()), 2;
    if (!migan::test::run_observed(cand.get(), input.data(), input.size(), observed.data(), observed.size(), check,
                                   buffer, error))
        return std::fprintf(stderr, "candidate observe: %s\n", error.c_str()), 2;
    const bool neutral = std::memcmp(observed.data(), cand_raw.data(), cand_raw.size() * sizeof(float)) == 0;
    const bool coverage = compared == reference.size();

    std::vector<uint8_t> ref_image(migan::rgb_elements), cand_image(migan::rgb_elements);
    std::vector<uint8_t> ref_gen(migan::rgb_elements), cand_gen(migan::rgb_elements), full(migan::mask_elements, 255);
    migan::composite_rgb8(ref_raw.data(), ref_raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(),
                          ref_image.data(), ref_image.size(), error);
    migan::composite_rgb8(cand_raw.data(), cand_raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(),
                          cand_image.data(), cand_image.size(), error);
    migan::composite_rgb8(ref_raw.data(), ref_raw.size(), rgb.data(), rgb.size(), full.data(), full.size(),
                          ref_gen.data(), ref_gen.size(), error);
    migan::composite_rgb8(cand_raw.data(), cand_raw.size(), rgb.data(), rgb.size(), full.data(), full.size(),
                          cand_gen.data(), cand_gen.size(), error);
    const image_stats vs_cpu =
        image_metrics(cand_image.data(), ref_image.data(), rgb.data(), mask.data(), migan::mask_elements);
    print_image("cpu_f32_composite", vs_cpu);
    image_stats vs_original;
    if (have_original) {
        vs_original = image_metrics(cand_image.data(), original.data(), rgb.data(), mask.data(), migan::mask_elements);
        print_image("original_composite", vs_original);
    }
    const image_stats generated =
        image_metrics(cand_gen.data(), ref_gen.data(), rgb.data(), full.data(), migan::mask_elements);
    print_image("cpu_f32_generated_full_frame", generated);
    const stats out = compare(cand_raw.data(), ref_raw.data(), cand_raw.size(), 256);

    std::vector<double> ref_ms, cand_ms;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        migan::run(cand.get(), input.data(), input.size(), cand_raw.data(), cand_raw.size(), error);
        auto t1 = std::chrono::steady_clock::now();
        migan::run(ref.get(), input.data(), input.size(), ref_raw.data(), ref_raw.size(), error);
        auto t2 = std::chrono::steady_clock::now();
        cand_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        ref_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    const bool strict = failed == 0 && coverage && neutral && vs_cpu.passed && (!have_original || vs_original.passed);
    std::printf("MIGAN_AB_SUMMARY {\"candidate_backend\":\"%s\",\"candidate_precision\":\"%s\",\"compared\":%zu,"
                "\"expected\":%zu,\"strict_failed_stages\":%zu,\"first_strict_failure\":\"%s\",\"worst_max_abs\":%.9g,"
                "\"worst_cos_min\":%.12f,\"output_max_abs\":%.9g,\"output_cos_global\":%.12f,"
                "\"output_cos_min\":%.12f,\"observer_neutral\":%s,\"image_gate_vs_cpu\":%s,"
                "\"image_gate_vs_original\":%s,\"cosine_gate_0_94\":%s,\"strict_passed\":%s,"
                "\"warm_ms_candidate_median\":%.3f,\"warm_ms_cpu_f32_median\":%.3f,\"iters\":%d,\"threads\":%d}\n",
                migan::backend_name(cand.get()), migan::weight_precision(cand.get()), compared, reference.size(),
                failed, first_failure.c_str(), worst_max, worst_cos, out.max_abs, out.cos_global, out.cos_min,
                neutral ? "true" : "false", vs_cpu.passed ? "true" : "false",
                have_original ? (vs_original.passed ? "true" : "false") : "null",
                out.cos_global > 0.94 ? "true" : "false", strict ? "true" : "false", median(cand_ms), median(ref_ms),
                iters, threads);
    return coverage && neutral ? 0 : 1;
}

void synthetic_frame(int width, int height, const std::string & kind, std::vector<uint8_t> & rgb,
                     std::vector<uint8_t> & mask) {
    rgb.resize(size_t(width) * height * 3);
    mask.assign(size_t(width) * height, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t p = size_t(y) * width + x;
            // Smooth scene gradient plus texture; deterministic, no RNG.
            rgb[3 * p] = uint8_t((x * 255 / std::max(1, width - 1) + (y % 37) * 3) & 255);
            rgb[3 * p + 1] = uint8_t((y * 255 / std::max(1, height - 1) + (x % 23) * 2) & 255);
            rgb[3 * p + 2] = uint8_t(((x / 8 + y / 8) % 2 ? 180 : 60) + ((x * y) % 17));
            bool fill = false;
            if (kind == "subtitle2") {
                // Two centered subtitle lines in the lower sixth, glyph-like strokes.
                const int l1 = height - height / 6, l2 = height - height / 12;
                const bool line = (y >= l1 - 22 && y < l1 + 22) || (y >= l2 - 22 && y < l2 + 22);
                fill = line && x >= width / 5 && x < width - width / 5 && ((x / 9) % 4 != 3);
            } else if (kind == "bar") {
                fill = y >= height - height / 5 && y < height - height / 20 && x >= width / 8 && x < width - width / 8;
            } else if (kind == "full") {
                fill = true;
            }
            mask[p] = fill ? 255 : 0;
        }
}

int frame(int argc, char ** argv) {
    if (argc != 11) return 2;
    const int width = std::atoi(argv[5]), height = std::atoi(argv[6]), iters = std::atoi(argv[7]),
              threads = std::atoi(argv[8]);
    std::vector<uint8_t> rgb, mask;
    synthetic_frame(width, height, argv[9], rgb, mask);
    std::string error;
    set_env("CRISPEMBED_MIGAN_GPU", "0");
    const char * batch_env = std::getenv("CRISPEMBED_MIGAN_BATCH");
    std::string batch_saved = batch_env ? batch_env : "";
    set_env("CRISPEMBED_MIGAN_BATCH", "1");
    std::unique_ptr<inpaint::context, decltype(&inpaint::free)> ref(inpaint::init(argv[2], threads, error),
                                                                    inpaint::free);
    if (!ref) return std::fprintf(stderr, "reference: %s\n", error.c_str()), 2;
    set_env("CRISPEMBED_MIGAN_BATCH", batch_saved.empty() ? nullptr : batch_saved.c_str());
    set_env("CRISPEMBED_MIGAN_GPU", std::strcmp(argv[4], "gpu") == 0 ? "1" : "0");
    std::unique_ptr<inpaint::context, decltype(&inpaint::free)> cand(inpaint::init(argv[3], threads, error),
                                                                     inpaint::free);
    if (!cand) return std::fprintf(stderr, "candidate: %s\n", error.c_str()), 2;

    // Tile count: same geometry as the wrapper (stride192, fill-bearing tiles).
    auto starts = [](int extent) {
        std::vector<int> r;
        const int last = std::max(0, extent - 256);
        for (int at = 0;; at = std::min(at + 192, last)) {
            r.push_back(at);
            if (at == last) return r;
        }
    };
    size_t tiles = 0;
    for (int top : starts(height))
        for (int left : starts(width)) {
            bool any = false;
            for (int y = top; y < std::min(height, top + 256) && !any; ++y)
                for (int x = left; x < std::min(width, left + 256); ++x)
                    if (mask[size_t(y) * width + x]) {
                        any = true;
                        break;
                    }
            tiles += any;
        }

    std::vector<uint8_t> ref_out(rgb.size()), cand_out(rgb.size());
    auto t0 = std::chrono::steady_clock::now();
    if (!inpaint::process_rgb8(ref.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                               ref_out.data(), ref_out.size(), error))
        return std::fprintf(stderr, "reference frame: %s\n", error.c_str()), 2;
    const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // First candidate frame includes graph construction/allocation (and GPU
    // pipeline compilation); warm frames reuse the persistent graphs.
    t0 = std::chrono::steady_clock::now();
    if (!inpaint::process_rgb8(cand.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                               cand_out.data(), cand_out.size(), error))
        return std::fprintf(stderr, "candidate frame: %s\n", error.c_str()), 2;
    const double first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<double> warm;
    std::vector<uint8_t> repeat(rgb.size());
    bool repeat_exact = true;
    for (int i = 0; i < iters; ++i) {
        t0 = std::chrono::steady_clock::now();
        inpaint::process_rgb8(cand.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                              repeat.data(), repeat.size(), error);
        warm.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        repeat_exact &= repeat == cand_out;
    }
    const image_stats s = image_metrics(cand_out.data(), ref_out.data(), rgb.data(), mask.data(), mask.size());
    print_image("cpu_f32_frame", s);
    std::ofstream(argv[10], std::ios::binary).write(reinterpret_cast<const char *>(cand_out.data()), cand_out.size());
    const double warm_ms = median(warm);
    std::printf("MIGAN_AB_FRAME {\"width\":%d,\"height\":%d,\"kind\":\"%s\",\"tiles\":%zu,\"candidate_backend\":\"%s\","
                "\"candidate_precision\":\"%s\",\"batch\":%d,\"threads\":%d,\"cpu_f32_frame_ms\":%.3f,"
                "\"candidate_first_frame_ms\":%.3f,\"candidate_warm_frame_ms_median\":%.3f,"
                "\"candidate_tiles_per_second\":%.3f,\"cpu_f32_tiles_per_second\":%.3f,\"warm_repeat_exact\":%s,"
                "\"image_gate_vs_cpu\":%s,\"iters\":%d}\n",
                width, height, argv[9], tiles, inpaint::backend_name(cand.get()), inpaint::weight_precision(cand.get()),
                inpaint::max_batch(cand.get()), threads, ref_ms, first_ms, warm_ms,
                warm_ms > 0 ? tiles * 1000.0 / warm_ms : 0, ref_ms > 0 ? tiles * 1000.0 / ref_ms : 0,
                repeat_exact ? "true" : "false", s.passed ? "true" : "false", iters);
    return 0;
}

// profile MODEL cpu|gpu BATCH THREADS: per-op device time, aggregated.
int profile(int argc, char ** argv) {
    if (argc != 6) return 2;
    std::string error;
    const size_t batch = size_t(std::atoi(argv[4]));
    set_env("CRISPEMBED_MIGAN_BATCH", argv[4]);
    std::unique_ptr<migan::context, decltype(&migan::free)> ctx(
        migan::init_ex(argv[2], checkpoint, std::atoi(argv[5]), parse_device(argv[3]), source_gguf, error),
        migan::free);
    if (!ctx) return std::fprintf(stderr, "%s\n", error.c_str()), 2;
    std::vector<float> input(batch * migan::input_elements, 0.25f);
    std::map<std::string, std::pair<double, int>> totals;
    double all = 0;
    const bool ok = migan::test::profile_ops(
        ctx.get(), input.data(), batch,
        [&](const char * op, const int64_t * ne, const int64_t * s0, const char * type, double ms) {
            char key[256];
            std::snprintf(key, sizeof key, "%s src0=%s[%lld,%lld,%lld,%lld] out=[%lld,%lld,%lld,%lld]", op, type,
                          (long long)s0[0], (long long)s0[1], (long long)s0[2], (long long)s0[3], (long long)ne[0],
                          (long long)ne[1], (long long)ne[2], (long long)ne[3]);
            auto & t = totals[std::string(op) == "MUL_MAT" || std::string(op) == "CONV_2D_DW" ? key : op];
            t.first += ms;
            t.second += 1;
            all += ms;
        },
        error);
    if (!ok) return std::fprintf(stderr, "%s\n", error.c_str()), 2;
    std::vector<std::pair<double, std::string>> sorted;
    for (const auto & t : totals)
        sorted.push_back({ t.second.first, t.first + " x" + std::to_string(t.second.second) });
    std::sort(sorted.rbegin(), sorted.rend());
    for (const auto & t : sorted) std::printf("MIGAN_AB_PROFILE %9.3f ms  %s\n", t.first, t.second.c_str());
    std::printf("MIGAN_AB_PROFILE_TOTAL {\"backend\":\"%s\",\"batch\":%zu,\"sum_node_ms\":%.3f}\n",
                migan::backend_name(ctx.get()), batch, all);
    return 0;
}

} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    set_env("CRISPEMBED_MIGAN", "1");
    if (argc > 1 && std::strcmp(argv[1], "stages") == 0) return stages(argc, argv);
    if (argc > 1 && std::strcmp(argv[1], "frame") == 0) return frame(argc, argv);
    if (argc > 1 && std::strcmp(argv[1], "profile") == 0) return profile(argc, argv);
    std::fprintf(stderr, "usage: see tests/test_migan_gpu_ab.cpp header\n");
    return 2;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
