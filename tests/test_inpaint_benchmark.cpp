// End-to-end benchmark of the ordinary image APIs. No observer/test archive.
#include "inpaint.h"
#include "lama.h"
#include "latent_inpaint.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/gpu_backend_pref.h"
#include "core/sha256.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::vector<uint8_t> read_bytes(const std::string & path, size_t count) {
    require(fs::file_size(path) == count, "raw fixture byte count differs: " + path);
    std::vector<uint8_t> bytes(count);
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(count))), "fixture read failed");
    return bytes;
}
int integer(const std::string & text, int low, int high) {
    require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos, "invalid integer");
    const int value = std::stoi(text);
    require(value >= low && value <= high, "integer outside bounds");
    return value;
}
std::FILE * exclusive_output(const fs::path & path) {
#ifdef _WIN32
    const int descriptor = _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (descriptor < 0) return nullptr;
    std::FILE * result = _fdopen(descriptor, "wb");
    if (!result) _close(descriptor);
#else
    const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    if (descriptor < 0) return nullptr;
    std::FILE * result = fdopen(descriptor, "wb");
    if (!result) close(descriptor);
#endif
    return result;
}
void write_new_output(const fs::path & path, const std::vector<uint8_t> & bytes) {
    std::FILE * file = exclusive_output(path);
    require(file != nullptr, "cannot exclusively create output; existing paths are protected");
    bool written = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    if (std::fclose(file) != 0) written = false;
    require(written, "output write failed");
}
int run(int argc, char ** argv) {
    const std::set<std::string> allowed{ "--engine", "--model",      "--model-sha256", "--checkpoint-sha256",
                                         "--vae",    "--vae-sha256", "--rgb",          "--mask",
                                         "--width",  "--height",     "--output",       "--threads",
                                         "--runs",   "--steps",      "--backend",      "--checkpoint-output" };
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        require(i + 1 < argc && allowed.count(argv[i]) && args.emplace(argv[i], argv[i + 1]).second,
                "usage: test-inpaint-benchmark --engine NAME --model FILE --model-sha256 SHA --rgb RAW --mask RAW "
                "--width N --height N --output NEW_RAW [--threads N] [--runs N] [--backend cpu|cuda|metal] "
                "[--checkpoint-sha256 SHA] [--vae FILE --vae-sha256 SHA --steps N] [--checkpoint-output NEW_RAW]");
    }
    auto required = [&](const std::string & name) -> std::string {
        require(args.count(name) && !args.at(name).empty(), "missing " + name);
        return args.at(name);
    };
    const std::string engine = required("--engine");
    require(engine == "lama" || engine == "migan256" || engine == "migan512" || engine == "moebius" ||
                engine == "pixelhacker",
            "unknown engine");
    const bool latent = engine == "moebius" || engine == "pixelhacker";
    const int width = integer(required("--width"), 1, inpaint::max_dimension);
    const int height = integer(required("--height"), 1, inpaint::max_dimension);
    require(size_t(width) * size_t(height) <= inpaint::max_pixels, "image too large");
    const int threads = args.count("--threads") ? integer(args.at("--threads"), 1, 64) : 8;
    const int repeats = args.count("--runs") ? integer(args.at("--runs"), 1, 20) : 5;
    const int steps = args.count("--steps") ? integer(args.at("--steps"), 2, 100) : 20;
    const std::string backend = args.count("--backend") ? args.at("--backend") : "cpu";
    require(backend == "cpu" || backend == "cuda" || backend == "metal" || backend == "mps", "unknown backend");
    require(backend == "cpu" || core_env::on("CRISPEMBED_INPAINT_GRAPH"), "GPU requires persistent graph gate");
    require(!latent || (width == 512 && height == 512), "latent benchmark uses native 512 geometry");
    const fs::path output = fs::absolute(required("--output"));
    require(!fs::is_symlink(fs::symlink_status(output)) && !fs::exists(output) &&
                fs::is_directory(output.parent_path()),
            "output must be a new file in an existing directory, not a symlink");
    fs::path checkpoint;
    if (args.count("--checkpoint-output")) {
        checkpoint = fs::absolute(required("--checkpoint-output"));
        require(!fs::is_symlink(fs::symlink_status(checkpoint)) && !fs::exists(checkpoint) &&
                    fs::is_directory(checkpoint.parent_path()),
                "checkpoint output must be a new file in an existing directory, not a symlink");
        require(fs::weakly_canonical(output) != fs::weakly_canonical(checkpoint),
                "checkpoint and final output must be different paths");
    }
    const auto rgb = read_bytes(required("--rgb"), size_t(width) * height * 3);
    const auto mask = read_bytes(required("--mask"), size_t(width) * height);
    require(std::all_of(mask.begin(), mask.end(), [](uint8_t value) { return value == 0 || value == 255; }),
            "mask must be binary");
    require(std::any_of(mask.begin(), mask.end(), [](uint8_t value) { return value != 0; }),
            "inference benchmark requires a nonempty fill mask");
    crispasr_set_gpu_backend_pref(backend.c_str());
    std::string error;
    std::unique_ptr<inpaint::context, decltype(&inpaint::free)> migan(nullptr, inpaint::free);
    std::unique_ptr<lama::context, decltype(&lama::free)> fourier(nullptr, lama::free);
    std::unique_ptr<latent_inpaint::context, decltype(&latent_inpaint::free)> diffusion(nullptr, latent_inpaint::free);
    const auto load_started = Clock::now();
    if (latent) {
        latent_inpaint::Options options;
        options.model = engine;
        options.denoiser_path = required("--model");
        options.denoiser_sha256 = required("--model-sha256");
        options.vae_path = required("--vae");
        options.vae_sha256 = required("--vae-sha256");
        options.threads = threads;
        options.steps = steps;
        options.guidance = engine == "moebius" ? 2.0f : 4.5f;
        diffusion.reset(latent_inpaint::init(options, error));
        require(bool(diffusion), error);
    } else if (engine == "lama") {
        fourier.reset(lama::init(required("--model"), required("--model-sha256"), threads, error));
        require(bool(fourier), error);
    } else {
        const std::string checkpoint =
            engine == "migan256" ? inpaint::checkpoint_sha256 : required("--checkpoint-sha256");
        migan.reset(inpaint::init_variant(required("--model").c_str(), checkpoint.c_str(),
                                          required("--model-sha256").c_str(), engine == "migan256" ? 256 : 512, threads,
                                          error));
        require(bool(migan), error);
    }
    const double load_ms = elapsed(load_started);
    std::vector<uint8_t> actual(rgb.size()), first;
    std::vector<double> samples;
    double cold_ms = 0;
    size_t known_errors = 0;
    bool repeat_exact = true;
    for (int iteration = 0; iteration <= repeats; ++iteration) {
        std::fill(actual.begin(), actual.end(), uint8_t(171));
        const auto started = Clock::now();
        const bool ok =
            diffusion ? latent_inpaint::process_rgb8(diffusion.get(), rgb.data(), rgb.size(), mask.data(), mask.size(),
                                                     width, height, actual.data(), actual.size(), error)
            : fourier ? lama::process_rgb8(fourier.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width,
                                           height, actual.data(), actual.size(), error)
                      : inpaint::process_rgb8(migan.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width,
                                              height, actual.data(), actual.size(), error);
        const double run_ms = elapsed(started);
        require(ok, error);
        for (size_t i = 0; i < actual.size(); ++i)
            if (!mask[i / 3]) known_errors += actual[i] != rgb[i];
        if (iteration == 0) {
            cold_ms = run_ms;
            first = actual;
            // Persist a completed image before the next long call. This I/O
            // is outside run_ms and does not mark the benchmark complete.
            if (!checkpoint.empty()) {
                require(known_errors == 0, "cold checkpoint changed known RGB pixels");
                write_new_output(checkpoint, first);
            }
        } else {
            samples.push_back(run_ms);
            repeat_exact = repeat_exact && actual == first;
        }
        std::cout << "INPAINT_RUN {\"engine\":" << std::quoted(engine) << ",\"iteration\":" << iteration
                  << ",\"milliseconds\":" << run_ms
                  << ",\"rgb_sha256\":" << std::quoted(core_sha256::bytes(actual.data(), actual.size())) << "}\n"
                  << std::flush;
    }
    write_new_output(output, first);
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median =
        sorted.size() % 2 ? sorted[sorted.size() / 2] : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2;
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double variance = 0;
    for (double value : samples) variance += (value - mean) * (value - mean);
    std::cout << std::setprecision(10)
              << "INPAINT_BENCHMARK {\"schema\":\"crispembed-image-api-benchmark-v1\",\"engine\":"
              << std::quoted(engine) << ",\"requested_backend\":" << std::quoted(backend) << ",\"width\":" << width
              << ",\"height\":" << height << ",\"threads\":" << threads
              << ",\"requested_steps\":" << (latent ? steps : 0) << ",\"load_and_validation_ms\":" << load_ms
              << ",\"cold_image_ms\":" << cold_ms << ",\"warm_median_ms\":" << median << ",\"warm_mean_ms\":" << mean
              << ",\"warm_stddev_ms\":" << std::sqrt(variance / samples.size()) << ",\"warm_samples_ms\":[";
    for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
    std::cout << "],\"input_rgb_sha256\":" << std::quoted(core_sha256::bytes(rgb.data(), rgb.size()))
              << ",\"mask_sha256\":" << std::quoted(core_sha256::bytes(mask.data(), mask.size()))
              << ",\"output_sha256\":" << std::quoted(core_sha256::bytes(first.data(), first.size()))
              << ",\"cold_checkpoint_sha256\":"
              << std::quoted(checkpoint.empty() ? std::string() : core_sha256::bytes(first.data(), first.size()))
              << ",\"model_sha256\":" << std::quoted(required("--model-sha256"))
              << ",\"vae_sha256\":" << std::quoted(latent ? required("--vae-sha256") : std::string())
              << ",\"known_pixel_errors\":" << known_errors
              << ",\"repeat_bit_exact\":" << (repeat_exact ? "true" : "false") << ",\"gates\":{";
    const char * gates[] = { "CRISPEMBED_INPAINT_GRAPH",
                             "CRISPEMBED_INPAINT_FAST",
                             "CRISPEMBED_INPAINT_GEMM_LAYOUT",
                             "CRISPEMBED_INPAINT_REFLECT_GATHER",
                             "CRISPEMBED_INPAINT_CPU_FFT",
                             "CRISPEMBED_INPAINT_CPU_GLA",
                             "CRISPEMBED_INPAINT_CPU_DEPTHWISE",
                             "CRISPEMBED_INPAINT_SHA256_ACCEL",
                             "CRISPEMBED_INPAINT_CPU_ATTENTION_TILES",
                             "CRISPEMBED_MOEBIUS_RELATIVE_REASSOC",
                             "CRISPEMBED_MOEBIUS_LOCAL_REASSOC",
                             "CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM",
                             "CRISPEMBED_MIGAN_CPU_KERNELS" };
    for (size_t i = 0; i < sizeof(gates) / sizeof(gates[0]); ++i)
        std::cout << (i ? "," : "") << std::quoted(gates[i]) << ':' << (core_env::on(gates[i]) ? "true" : "false");
    std::cout
        << "},\"ordinary_image_api\":true,\"observers\":false,\"original_quality_qualification\":false,\"passed\":"
        << (known_errors == 0 && repeat_exact ? "true" : "false") << "}\n";
    return known_errors == 0 && repeat_exact ? 0 : 1;
}
} // namespace
int main(int argc, char ** argv) {
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
    }
    std::cout.flush();
    std::cerr.flush();
    core_util::clean_exit(status);
}
