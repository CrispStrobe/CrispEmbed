// MI-GAN backend parity and persistent-graph benchmark. --contracts
// executes only a tiny scalar-oracle graph; full mode requires a reviewed GGUF.
// Original-source stage replay remains test-migan[-512]-diff --graph-backend.
#include "migan.h"
#include "migan_test_observer.h"
#include "core/clean_exit.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
using context_ptr = std::unique_ptr<migan::context, decltype(&migan::free)>;

double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

void environment(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

std::string json_string(const std::string & value) {
    std::string out = "\"";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out += '\\';
        if (ch < 32) {
            char escape[7];
            std::snprintf(escape, sizeof(escape), "\\u%04x", unsigned(ch));
            out += escape;
        } else
            out += char(ch);
    }
    return out + '"';
}

void print_stats(const char * name, const migan::execution_stats & s) {
    std::printf("MIGAN_GRAPH_STATS {\"kind\":%s,\"backend\":%s,\"persistent\":%s,\"fast\":%s,"
                "\"cpu_kernels\":%s,\"cpu_plan_cached\":%s,\"cpu_threadpool_reused\":%s,\"resolution\":%d,"
                "\"graph_nodes\":%zu,\"graph_buffer_bytes\":%zu,\"weight_buffer_bytes\":%zu,\"graph_builds\":%llu,"
                "\"graph_allocations\":%llu,\"runs\":%llu,\"reuses\":%llu,\"weight_uploads\":%llu,"
                "\"weight_upload_bytes\":%llu,\"input_uploads\":%llu,\"input_upload_bytes\":%llu,"
                "\"output_downloads\":%llu,\"output_download_bytes\":%llu,\"last_build_ms\":%.9g,"
                "\"last_compute_ms\":%.9g,\"total_compute_ms\":%.9g,\"last_run_ms\":%.9g}\n",
                json_string(name).c_str(), json_string(s.backend).c_str(), s.persistent ? "true" : "false",
                s.fast ? "true" : "false", s.cpu_kernels ? "true" : "false", s.cpu_plan_cached ? "true" : "false",
                s.cpu_threadpool_reused ? "true" : "false", s.resolution, s.graph_nodes, s.graph_buffer_bytes,
                s.weight_buffer_bytes, (unsigned long long)s.graph_builds, (unsigned long long)s.graph_allocations,
                (unsigned long long)s.runs, (unsigned long long)s.reuses, (unsigned long long)s.weight_uploads,
                (unsigned long long)s.weight_upload_bytes, (unsigned long long)s.input_uploads,
                (unsigned long long)s.input_upload_bytes, (unsigned long long)s.output_downloads,
                (unsigned long long)s.output_download_bytes, s.last_build_ms, s.last_compute_ms, s.total_compute_ms,
                s.last_run_ms);
}

bool read_bytes(const std::string & path, std::vector<uint8_t> & data, size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streamoff(count)) return false;
    data.resize(count);
    stream.seekg(0);
    return bool(stream.read(reinterpret_cast<char *>(data.data()), std::streamsize(count)));
}

bool compare_raw(const std::vector<float> & actual, const std::vector<float> & expected, const char * phase,
                 int iteration) {
    double max_abs = 0, error2 = 0, ref2 = 0, actual2 = 0, dot = 0;
    bool finite = actual.size() == expected.size();
    for (size_t i = 0; finite && i < actual.size(); ++i) {
        finite = std::isfinite(actual[i]) && std::isfinite(expected[i]);
        if (!finite) break;
        const double delta = double(actual[i]) - expected[i];
        max_abs = std::max(max_abs, std::abs(delta));
        error2 += delta * delta;
        actual2 += double(actual[i]) * actual[i];
        ref2 += double(expected[i]) * expected[i];
        dot += double(actual[i]) * expected[i];
    }
    if (!finite) {
        std::printf("MIGAN_GRAPH_PARITY {\"phase\":%s,\"iteration\":%d,\"max_abs\":null,\"rel_l2\":null,"
                    "\"cosine\":null,\"finite\":false,\"passed\":false}\n",
                    json_string(phase).c_str(), iteration);
        return false;
    }
    const double relative_l2 = std::sqrt(error2 / std::max(ref2, 1e-30));
    const double cosine = actual2 && ref2 ? dot / std::sqrt(actual2 * ref2) : (actual2 == ref2 ? 1 : 0);
    const bool passed = finite && max_abs <= 0.002 && relative_l2 <= 0.0003 && cosine >= 0.99999;
    std::printf("MIGAN_GRAPH_PARITY {\"phase\":%s,\"iteration\":%d,\"max_abs\":%.9g,\"rel_l2\":%.9g,"
                "\"cosine\":%.9g,\"finite\":%s,\"passed\":%s}\n",
                json_string(phase).c_str(), iteration, max_abs, relative_l2, cosine, finite ? "true" : "false",
                passed ? "true" : "false");
    return passed;
}

bool compare_image(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & expected,
                   const std::vector<uint8_t> & rgb, const std::vector<uint8_t> & mask) {
    size_t known_errors = 0, count = 0;
    double sum = 0, sum2 = 0;
    int max_abs = 0;
    for (size_t pixel = 0; pixel < mask.size(); ++pixel)
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t index = 3 * pixel + channel;
            if (!mask[pixel]) {
                known_errors += actual[index] != rgb[index] || expected[index] != rgb[index];
            } else {
                const int delta = std::abs(int(actual[index]) - expected[index]);
                max_abs = std::max(max_abs, delta);
                sum += delta;
                sum2 += delta * delta;
                ++count;
            }
        }
    const double mae = count ? sum / count : 0;
    const double mse = count ? sum2 / count : 0;
    const double psnr = mse ? 10 * std::log10(255.0 * 255.0 / mse) : 1000;
    const bool passed = known_errors == 0 && max_abs <= 1 && mae <= 0.1 && psnr >= 50;
    std::printf("MIGAN_GRAPH_RGB {\"fill_channels\":%zu,\"known_errors\":%zu,\"max_abs\":%d,\"mae\":%.9g,"
                "\"mse\":%.9g,\"psnr_db\":",
                count, known_errors, max_abs, mae, mse);
    if (mse)
        std::printf("%.9g", psnr);
    else
        std::printf("null");
    std::printf(",\"exact_fill\":%s,\"passed\":%s}\n", mse == 0 ? "true" : "false", passed ? "true" : "false");
    return passed;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

double sample_deviation(const std::vector<double> & values) {
    if (values.size() < 2) return 0;
    double mean = 0, variance = 0;
    for (double value : values) mean += value;
    mean /= values.size();
    for (double value : values) variance += (value - mean) * (value - mean);
    return std::sqrt(variance / (values.size() - 1));
}

void print_timing(const char * phase, int iteration, bool changed, double cpu_ms, double graph_ms) {
    std::printf("MIGAN_GRAPH_TIMING {\"phase\":%s,\"iteration\":%d,\"input_variant\":%s,"
                "\"cpu_ms\":%.9g,\"graph_ms\":%.9g}\n",
                json_string(phase).c_str(), iteration,
                json_string(changed ? "sign-inverted-known-rgb" : "original").c_str(), cpu_ms, graph_ms);
}

int integer(const std::string & value, int lo, int hi) {
    if (value.empty() || value.size() > 9 || value.find_first_not_of("0123456789") != std::string::npos) return -1;
    const long result = std::strtol(value.c_str(), nullptr, 10);
    return result >= lo && result <= hi ? int(result) : -1;
}
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    std::string backend = "cpu", model_path, digest, rgb_path, mask_path;
    int resolution = 256, threads = 2, warm_runs = 3;
    bool contracts = false, fast = false, cpu_kernels = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--contracts")
            contracts = true;
        else if (key == "--fast")
            fast = true;
        else if (key == "--cpu-kernels")
            cpu_kernels = true;
        else if (i + 1 < argc) {
            const std::string value = argv[++i];
            if (key == "--backend")
                backend = value;
            else if (key == "--model")
                model_path = value;
            else if (key == "--checkpoint-sha256")
                digest = value;
            else if (key == "--rgb")
                rgb_path = value;
            else if (key == "--mask")
                mask_path = value;
            else if (key == "--size")
                resolution = integer(value, 256, 512);
            else if (key == "--threads")
                threads = integer(value, 1, 64);
            else if (key == "--warm-runs")
                warm_runs = integer(value, 1, 100);
            else
                return 2;
        } else
            return 2;
    }
    if ((backend != "cpu" && backend != "cuda" && backend != "metal" && backend != "mps" && backend != "auto") ||
        (resolution != 256 && resolution != 512) || threads < 1 || warm_runs < 1 ||
        (contracts && (!model_path.empty() || !digest.empty() || !rgb_path.empty() || !mask_path.empty())) ||
        (!contracts && (model_path.empty() || digest.empty() || rgb_path.empty() != mask_path.empty()))) {
        std::fprintf(stderr,
                     "Usage: test-migan-graph --contracts [--backend cpu|cuda|metal] [--threads N] [--warm-runs N] "
                     "[--cpu-kernels]\n"
                     "or: test-migan-graph --model MODEL.gguf --checkpoint-sha256 SHA --size 256|512 "
                     "[--backend cpu|cuda|metal|auto] [--threads N] [--warm-runs N] [--rgb RGB.raw --mask MASK.raw] "
                     "[--fast] [--cpu-kernels]\n");
        return 2;
    }
    crispasr_set_gpu_backend_pref(backend.c_str());
    environment("CRISPEMBED_MIGAN", "1");
    environment("CRISPEMBED_MIGAN512", "1");
    environment("CRISPEMBED_INPAINT_GRAPH", "1");
    environment("CRISPEMBED_INPAINT_FAST", fast ? "1" : "0");
    environment("CRISPEMBED_MIGAN_CPU_KERNELS", cpu_kernels ? "1" : "0");
    std::string error;
    if (contracts) {
        migan::execution_stats stats;
        const bool passed = migan::test::graph_backend_controls(threads, warm_runs + 1, stats, error);
        if (!passed) std::fprintf(stderr, "%s\n", error.c_str());
        if (passed) print_stats("synthetic-controls", stats);
        std::printf("MIGAN_GRAPH_CONTRACT "
                    "{\"requested_backend\":%s,\"requested_fast\":%s,\"fast\":%s,\"cpu_kernels\":%s,\"passed\":%s}\n",
                    json_string(backend).c_str(), fast ? "true" : "false", stats.fast ? "true" : "false",
                    stats.cpu_kernels ? "true" : "false", passed ? "true" : "false");
        return passed ? 0 : 1;
    }
    const size_t pixels = size_t(resolution) * resolution;
    std::vector<uint8_t> rgb(3 * pixels), mask(pixels), image(3 * pixels), cpu_image(3 * pixels);
    if (!rgb_path.empty()) {
        if (!read_bytes(rgb_path, rgb, 3 * pixels) || !read_bytes(mask_path, mask, pixels)) {
            std::fprintf(stderr, "MI-GAN graph benchmark input file sizes differ\n");
            return 2;
        }
    } else {
        for (int y = 0; y < resolution; ++y)
            for (int x = 0; x < resolution; ++x) {
                const size_t p = size_t(y) * resolution + x;
                for (int c = 0; c < 3; ++c) rgb[3 * p + c] = uint8_t((x * (c + 3) + y * (c + 7) + x * y) % 256);
                mask[p] =
                    (x > resolution / 4 && x < 3 * resolution / 4 && y > resolution / 4 && y < 3 * resolution / 4) ||
                            (x + 3 * y) % 97 == 0
                        ? 255
                        : 0;
            }
    }
    std::vector<float> input(4 * pixels), changed(4 * pixels), cpu_raw(3 * pixels), raw(3 * pixels),
        base_cpu(3 * pixels), observed(3 * pixels);
    if (!migan::preprocess_rgb8_variant(resolution, rgb.data(), rgb.size(), mask.data(), mask.size(), resolution,
                                        resolution, input.data(), input.size(), error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    changed = input;
    for (size_t i = pixels; i < changed.size(); ++i) changed[i] = -changed[i];
    environment("CRISPEMBED_INPAINT_GRAPH", "0");
    auto start = clock_type::now();
    context_ptr cpu(migan::init_variant(model_path.c_str(), digest.c_str(), resolution, threads, error), migan::free);
    const double cpu_load_ms = milliseconds(start);
    if (!cpu) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    environment("CRISPEMBED_INPAINT_GRAPH", "1");
    start = clock_type::now();
    context_ptr graph(migan::init_variant(model_path.c_str(), digest.c_str(), resolution, threads, error), migan::free);
    const double graph_load_ms = milliseconds(start);
    if (!graph) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    migan::execution_stats initial;
    if (!migan::get_execution_stats(graph.get(), initial, error)) return 2;
    bool passed = initial.persistent && initial.fast == (fast && initial.backend != "CPU") &&
                  initial.cpu_kernels == cpu_kernels && initial.graph_builds == 1 && initial.graph_allocations == 1 &&
                  initial.runs == 0 && initial.weight_uploads == (resolution == 256 ? 154 : 177);
    auto execute = [&](migan::context * ctx, const std::vector<float> & tile, std::vector<float> & output) {
        const auto begin = clock_type::now();
        if (!migan::run(ctx, tile.data(), tile.size(), output.data(), output.size(), error))
            throw std::runtime_error(error);
        return milliseconds(begin);
    };
    try {
        const double cpu_cold_ms = execute(cpu.get(), input, cpu_raw);
        base_cpu = cpu_raw;
        const double graph_cold_ms = execute(graph.get(), input, raw);
        print_timing("cold", 0, false, cpu_cold_ms, graph_cold_ms);
        passed &= compare_raw(raw, cpu_raw, "cold", 0);
        std::array<std::vector<float>, 2> previous = { raw, {} };
        std::vector<double> cpu_times, graph_times;
        bool repeat_bit_exact = true;
        for (int i = 0; i < warm_runs; ++i) {
            const int which = i % 2 ? 0 : 1;
            const auto & active = which ? changed : input;
            cpu_times.push_back(execute(cpu.get(), active, cpu_raw));
            graph_times.push_back(execute(graph.get(), active, raw));
            print_timing("warm", i, which != 0, cpu_times.back(), graph_times.back());
            passed &= compare_raw(raw, cpu_raw, "warm", i);
            if (previous[which].empty())
                previous[which] = raw;
            else
                repeat_bit_exact &= std::memcmp(raw.data(), previous[which].data(), raw.size() * sizeof(float)) == 0;
        }
        execute(graph.get(), input, raw);
        repeat_bit_exact &= std::memcmp(raw.data(), previous[0].data(), raw.size() * sizeof(float)) == 0;
        passed &= repeat_bit_exact;
        if (!migan::composite_rgb8_variant(resolution, raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(),
                                           mask.size(), image.data(), image.size(), error) ||
            !migan::composite_rgb8_variant(resolution, base_cpu.data(), base_cpu.size(), rgb.data(), rgb.size(),
                                           mask.data(), mask.size(), cpu_image.data(), cpu_image.size(), error))
            throw std::runtime_error(error);
        passed &= compare_image(image, cpu_image, rgb, mask);
        migan::execution_stats before_observer, after_observer, retained, baseline;
        if (!migan::get_execution_stats(graph.get(), before_observer, error)) throw std::runtime_error(error);
        size_t observed_bytes = 0, stages = 0;
        const auto observe = [&](const std::string &, const float * data, size_t count,
                                 const std::array<int64_t, 4> &) {
            ++stages;
            return count && std::all_of(data, data + count, [](float v) { return std::isfinite(v); });
        };
        bool neutral = true;
        for (int i = 0; i < 2; ++i) {
            if (!migan::test::run_observed(graph.get(), input.data(), input.size(), observed.data(), observed.size(),
                                           observe, observed_bytes, error))
                throw std::runtime_error(error);
            neutral &= std::memcmp(raw.data(), observed.data(), raw.size() * sizeof(float)) == 0;
        }
        if (!migan::get_execution_stats(graph.get(), after_observer, error) ||
            !migan::test::get_observed_execution_stats(graph.get(), retained, error) ||
            !migan::get_execution_stats(cpu.get(), baseline, error))
            throw std::runtime_error(error);
        const bool reuse = after_observer.graph_builds == 1 && after_observer.graph_allocations == 1 &&
                           after_observer.graph_buffer_bytes == initial.graph_buffer_bytes &&
                           after_observer.graph_nodes == initial.graph_nodes &&
                           after_observer.weight_uploads == initial.weight_uploads &&
                           after_observer.weight_upload_bytes == initial.weight_upload_bytes &&
                           after_observer.runs == uint64_t(warm_runs + 2) &&
                           after_observer.reuses == uint64_t(warm_runs + 1) &&
                           after_observer.input_uploads == after_observer.runs &&
                           after_observer.output_downloads == after_observer.runs;
        const bool isolation = after_observer.runs == before_observer.runs &&
                               after_observer.graph_buffer_bytes == before_observer.graph_buffer_bytes &&
                               after_observer.output_download_bytes == before_observer.output_download_bytes &&
                               retained.graph_builds == 1 && retained.graph_allocations == 1 && retained.runs == 2 &&
                               retained.reuses == 1 && observed_bytes == retained.graph_buffer_bytes && stages > 0;
        passed &= reuse && isolation && neutral;
        print_stats("legacy-cpu", baseline);
        print_stats("production", after_observer);
        print_stats("observed", retained);
        std::printf(
            "MIGAN_GRAPH_BENCHMARK {\"requested_backend\":%s,\"selected_backend\":%s,\"fast\":%s,\"cpu_kernels\":%s,"
            "\"size\":%d,\"threads\":%d,\"warm_runs\":%d,\"cpu_load_ms\":%.9g,\"graph_load_ms\":%.9g,"
            "\"cpu_cold_ms\":%.9g,\"graph_cold_ms\":%.9g,\"cpu_warm_median_ms\":%.9g,"
            "\"graph_warm_median_ms\":%.9g,\"cpu_warm_min_ms\":%.9g,\"cpu_warm_max_ms\":%.9g,"
            "\"cpu_warm_stddev_ms\":%.9g,\"graph_warm_min_ms\":%.9g,\"graph_warm_max_ms\":%.9g,"
            "\"graph_warm_stddev_ms\":%.9g,\"warm_speedup\":%.9g,\"warm_reuse_verified\":%s,"
            "\"repeat_bit_exact\":%s,\"observer_isolated\":%s,\"observer_bit_exact\":%s,\"passed\":%s}\n",
            json_string(backend).c_str(), json_string(after_observer.backend).c_str(),
            after_observer.fast ? "true" : "false", after_observer.cpu_kernels ? "true" : "false", resolution, threads,
            warm_runs, cpu_load_ms, graph_load_ms, cpu_cold_ms, graph_cold_ms, median(cpu_times), median(graph_times),
            *std::min_element(cpu_times.begin(), cpu_times.end()),
            *std::max_element(cpu_times.begin(), cpu_times.end()), sample_deviation(cpu_times),
            *std::min_element(graph_times.begin(), graph_times.end()),
            *std::max_element(graph_times.begin(), graph_times.end()), sample_deviation(graph_times),
            median(cpu_times) / std::max(median(graph_times), 1e-9), reuse ? "true" : "false",
            repeat_bit_exact ? "true" : "false", isolation ? "true" : "false", neutral ? "true" : "false",
            passed ? "true" : "false");
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "%s\n", exception.what());
        return 2;
    }
    return passed ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
