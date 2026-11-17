#include "gector.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
    std::string model, ids, output, trace;
    int threads = 1, warmup = 1, repeats = 5;
};
int number(const std::string & text, int lower, int upper) {
    int n = 0;
    const auto p = std::from_chars(text.data(), text.data() + text.size(), n);
    if (p.ec != std::errc() || p.ptr != text.data() + text.size() || n < lower || n > upper)
        throw std::runtime_error("Invalid bounded integer argument: " + text);
    return n;
}
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "CRISPEMBED_GECTOR=1 crispembed-gector-run --model model.gguf --ids ids.txt\n"
                         "  --output output-prefix [--threads 1..64] [--warmup 0..10] [--repeats 1..100]\n"
                         "  [--trace-prefix diagnostic-prefix]\n"
                         "IDs are whitespace-separated int64 tokens. Output is raw little-endian F32.\n";
            core_util::clean_exit(0);
        }
        if (++i >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--model") o.model = value;
        else if (key == "--ids") o.ids = value;
        else if (key == "--output") o.output = value;
        else if (key == "--trace-prefix") o.trace = value;
        else if (key == "--threads") o.threads = number(value, 1, 64);
        else if (key == "--warmup") o.warmup = number(value, 0, 10);
        else if (key == "--repeats") o.repeats = number(value, 1, 100);
        else throw std::runtime_error("Unknown option " + key);
    }
    if (o.model.empty() || o.ids.empty() || o.output.empty())
        throw std::runtime_error("--model, --ids and --output are required");
    return o;
}
std::vector<int64_t> read_ids(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Unable to read token IDs");
    std::vector<int64_t> ids;
    std::string word;
    while (in >> word) {
        int64_t value;
        const auto p = std::from_chars(word.data(), word.data() + word.size(), value);
        if (p.ec != std::errc() || p.ptr != word.data() + word.size())
            throw std::runtime_error("IDs must contain only int64 decimal integers");
        ids.push_back(value);
        if (ids.size() > CRISPEMBED_GECTOR_MAX_TOKENS) throw std::runtime_error("Too many IDs; maximum is 512");
    }
    if (in.bad()) throw std::runtime_error("Unable to finish reading token IDs");
    return ids;
}
void save(const std::string & path, const float * values, size_t n) {
    std::ofstream out(path, std::ios::binary);
    if (!out || !out.write(reinterpret_cast<const char *>(values), n * sizeof(float)))
        throw std::runtime_error("Unable to write output tensor");
}
struct Trace { std::string prefix, error; };
int trace(void * user, const char * stage, const float * values, size_t rows, size_t columns) {
    auto & output = *static_cast<Trace *>(user);
    try {
        save(output.prefix + "." + stage + ".f32", values, rows * columns);
        return 0;
    } catch (const std::exception & e) {
        output.error = e.what();
        return 1;
    }
}
int run(int argc, char ** argv) {
    const auto o = parse(argc, argv);
    const uint16_t byte_order = 1;
    if (*reinterpret_cast<const uint8_t *>(&byte_order) != 1)
        throw std::runtime_error("Raw driver output currently requires a little-endian host");
    const auto ids = read_ids(o.ids);
    const std::vector<int64_t> mask(ids.size(), 1);
    char error[512] = {};
    if (crispembed_gector_validate_inputs(ids.data(), ids.size(), mask.data(), mask.size(), 1, error, sizeof(error)))
        throw std::runtime_error(error);
    const auto load_start = std::chrono::steady_clock::now();
    std::unique_ptr<crispembed_gector_context, decltype(&crispembed_gector_free)> context(
        crispembed_gector_create(o.model.c_str(), o.threads, error, sizeof(error)), crispembed_gector_free);
    if (!context) throw std::runtime_error(error);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    std::vector<float> labels(ids.size() * CRISPEMBED_GECTOR_LABEL_COUNT);
    std::vector<float> detect(ids.size() * CRISPEMBED_GECTOR_DETECT_COUNT);
    std::vector<float> hidden(ids.size() * CRISPEMBED_GECTOR_HIDDEN_SIZE);
    std::vector<double> timings;
    for (int i = -o.warmup; i < o.repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const int rc = crispembed_gector_forward(context.get(), ids.data(), ids.size(), mask.data(), mask.size(), 1,
                                                labels.data(), labels.size(), detect.data(), detect.size(),
                                                hidden.data(), hidden.size(), error, sizeof(error));
        if (rc) throw std::runtime_error(error);
        if (i >= 0) timings.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    save(o.output + ".labels.f32", labels.data(), labels.size());
    save(o.output + ".detect.f32", detect.data(), detect.size());
    save(o.output + ".encoder.f32", hidden.data(), hidden.size());
    if (!o.trace.empty()) {
        Trace output{ o.trace, "" };
        const int rc = crispembed_gector_forward_trace(context.get(), ids.data(), ids.size(), mask.data(), mask.size(), 1,
                                                      labels.data(), labels.size(), detect.data(), detect.size(),
                                                      hidden.data(), hidden.size(), trace, &output, error, sizeof(error));
        if (rc) throw std::runtime_error(output.error.empty() ? error : output.error);
    }
    std::vector<double> sorted = timings;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 ? sorted[sorted.size() / 2] :
                          (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2;
    const double p95 = sorted[static_cast<size_t>(std::ceil(sorted.size() * 0.95)) - 1];
    std::ofstream report(o.output + ".json");
    if (!report) throw std::runtime_error("Unable to write timing report");
    report << "{\n  \"schema\": 1, \"backend\": \"crispembed-gector-cpu\",\n"
           << "  \"tokens\": " << ids.size() << ", \"label_count\": 5000, \"detect_count\": 2,\n"
           << "  \"hidden_size\": 768, \"threads\": " << o.threads << ", \"warmups\": " << o.warmup << ",\n"
           << "  \"load_ms\": " << load_ms << ", \"median_ms\": " << median << ", \"p95_ms\": " << p95 << ",\n"
           << "  \"raw_dtype\": \"little-endian-float32\", \"raw_layout\": \"token-major\",\n"
           << "  \"forward_ms\": [";
    for (size_t i = 0; i < timings.size(); ++i) report << (i ? ", " : "") << timings[i];
    report << "]\n}\n";
    if (!report) throw std::runtime_error("Unable to finish timing report");
    std::cout << "tokens=" << ids.size() << " threads=" << o.threads << " median_ms=" << median << " p95_ms=" << p95 << '\n';
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    int rc = 1;
    try { rc = run(argc, argv); }
    catch (const std::exception & e) { std::cerr << "gector: " << e.what() << '\n'; }
    core_util::clean_exit(rc);
}
