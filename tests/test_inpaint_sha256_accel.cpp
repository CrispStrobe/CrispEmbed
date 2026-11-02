// Model-free exact SHA-256 controls, plus explicitly requested read-only
// throughput/whole-file identity probes. No model inference is performed.
#include "core/clean_exit.h"
#include "core/sha256_accelerated.h"
#include "core/temp_file.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
size_t checks = 0, failures = 0;

void check(bool condition, const char * label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "SHA256_FAIL %s\n", label);
    }
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct TemporaryFile {
    std::string path = core_tmp::make_private(".sha-test");
    ~TemporaryFile() {
        if (!path.empty()) std::remove(path.c_str());
    }
};

std::vector<uint8_t> pattern(size_t size) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = uint8_t((i * 131 + (i >> 3) + 17) & 255);
    return data;
}

void one_vector(const std::string & input, const std::string & expected) {
    check(core_sha256::bytes(input.data(), input.size()) == expected, "existing scalar known vector");
    for (bool request : { false, true }) {
        core_sha256_accel::Stats stats;
        check(core_sha256_accel::bytes(input.data(), input.size(), request, &stats) == expected,
              "selected known vector");
        check(stats.sha_ni_selected == (request && core_sha256_accel::hardware_available()),
              "requested/available dispatch");
        check(stats.input_bytes == input.size() && stats.complete_blocks == input.size() / 64,
              "known-vector complete-block accounting");
        check(stats.accelerated_blocks == (stats.sha_ni_selected ? input.size() / 64 : 0),
              "known-vector accelerated-block accounting");
    }
}

void controls() {
    one_vector("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    one_vector("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    one_vector("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    one_vector(std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    for (unsigned mask = 0; mask < 8; ++mask) {
        const uint32_t leaf1 = ((mask & 1) ? UINT32_C(1) << 9 : 0) | ((mask & 2) ? UINT32_C(1) << 19 : 0);
        const uint32_t leaf7 = (mask & 4) ? UINT32_C(1) << 29 : 0;
        check(core_sha256_accel::required_features(leaf1, leaf7) == (mask == 7), "all required CPUID bits");
    }
    check(!core_sha256_accel::required_features(0, UINT32_MAX), "SHA alone is insufficient");
    check(!core_sha256_accel::required_features(UINT32_MAX, 0), "SSE alone is insufficient");
    if (!core_sha256_accel::compiled_sha_ni())
        check(!core_sha256_accel::hardware_available(), "uncompiled extension uses scalar fallback");

    const auto data = pattern(8195);
    const auto unchanged = data;
    const size_t lengths[] = { 0, 1, 7, 31, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 255, 1024, 8191 };
    const size_t chunks[] = { 1, 3, 7, 55, 63, 64, 65, 127, 4093 };
    for (size_t offset : { size_t(0), size_t(1), size_t(3) }) {
        for (size_t size : lengths) {
            const uint8_t * input = data.data() + offset;
            const std::string expected = core_sha256::bytes(input, size);
            for (bool request : { false, true }) {
                check(core_sha256_accel::bytes(input, size, request) == expected, "unaligned binary vector");
                for (size_t chunk : chunks) {
                    core_sha256_accel::Hasher stream(request);
                    bool accepted = true;
                    for (size_t position = 0; position < size; position += std::min(chunk, size - position)) {
                        accepted = stream.update(input + position, std::min(chunk, size - position)) && accepted;
                        accepted = stream.update(nullptr, 0) && accepted;
                    }
                    check(accepted, "all stream and empty updates accepted");
                    check(stream.finish() == expected, "split/padding boundary exact");
                    check(stream.finish() == expected, "finish is stable");
                    check(!stream.update(input, size), "finished stream rejects update");
                    check(stream.finish() == expected, "rejected post-finish update preserves digest");
                    check(stream.stats().input_bytes == size && stream.stats().complete_blocks == size / 64,
                          "stream complete-block accounting");
                    check(stream.stats().accelerated_blocks ==
                              ((request && core_sha256_accel::hardware_available()) ? size / 64 : 0),
                          "fragmented stream executes the selected compressor");
                }
            }
        }
    }
    check(data == unchanged, "source bytes were not modified");
    if (!core_sha256_accel::hardware_available()) {
        core_sha256_accel::Stats fallback;
        check(core_sha256_accel::bytes(data.data(), data.size(), true, &fallback) ==
                  core_sha256::bytes(data.data(), data.size()),
              "unavailable hardware requested-on fallback is exact");
        check(fallback.requested && !fallback.hardware_available && !fallback.sha_ni_selected &&
                  fallback.accelerated_blocks == 0 && fallback.input_bytes == data.size() &&
                  fallback.complete_blocks == data.size() / 64,
              "unavailable hardware reports actual scalar execution");
    }
    for (bool request : { false, true }) {
        check(core_sha256_accel::bytes(nullptr, 0, request) == core_sha256::bytes(nullptr, 0), "empty null buffer");
        check(core_sha256_accel::bytes(nullptr, 1, request).empty(), "nonempty null buffer rejected");
        core_sha256_accel::Hasher invalid(request);
        check(invalid.update("abc", 3), "valid prefix before rejected null");
        check(!invalid.update(nullptr, 1) && invalid.finish().empty(), "null update poisons unfinished stream");
    }

    TemporaryFile file;
    check(!file.path.empty(), "private temporary file created");
    if (!file.path.empty()) {
        for (bool request : { false, true }) {
            check(core_sha256_accel::file(file.path, request) == core_sha256::bytes(nullptr, 0), "empty file exact");
            check(core_sha256_accel::file(file.path + ".missing", request).empty(), "missing file rejected");
            check(core_sha256_accel::file(std::filesystem::path(file.path).parent_path().string(), request).empty(),
                  "file open/read failure returns empty digest");
        }
        std::ofstream output(file.path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size()));
        output.flush();
        check(bool(output), "test file bytes written");
        output.close();
        check(!output.fail(), "test file closed");
        const std::string expected = core_sha256::bytes(data.data(), data.size());
        check(core_sha256::file(file.path) == expected, "ordinary file path remains exact");
        for (bool request : { false, true }) {
            core_sha256_accel::Stats stats;
            check(core_sha256_accel::file(file.path, request, &stats) == expected, "optional file path exact");
            check(stats.requested == request &&
                      stats.sha_ni_selected == (request && core_sha256_accel::hardware_available()),
                  "file requested/actual backend accounting");
            check(stats.input_bytes == data.size() && stats.complete_blocks == data.size() / 64,
                  "file stream accounting");
            check(stats.accelerated_blocks ==
                      ((request && core_sha256_accel::hardware_available()) ? data.size() / 64 : 0),
                  "file actually used selected complete-block implementation");
        }
    }
}

void memory_benchmark(size_t mib, int iterations) {
    const auto data = pattern(mib * 1024 * 1024 + 3);
    const auto * input = data.data() + 3;
    const size_t size = data.size() - 3;
    const std::string expected = core_sha256::bytes(input, size);
    std::vector<double> scalar, accelerated;
    uint64_t observed_input_bytes = 0, observed_complete_blocks = 0, observed_blocks = 0;
    bool correct = true;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Alternate the measured order. Both arms read the same immutable
        // unaligned buffer, and every resulting digest must match exactly.
        for (int arm = 0; arm < 2; ++arm) {
            const bool request = bool((iteration + arm) & 1);
            core_sha256_accel::Stats stats;
            const auto start = Clock::now();
            const std::string actual =
                request ? core_sha256_accel::bytes(input, size, true, &stats) : core_sha256::bytes(input, size);
            const double ms = elapsed_ms(start);
            (request ? accelerated : scalar).push_back(ms);
            observed_input_bytes += stats.input_bytes;
            observed_complete_blocks += stats.complete_blocks;
            observed_blocks += stats.accelerated_blocks;
            correct = correct && actual == expected;
        }
    }
    const bool accounting =
        observed_input_bytes == uint64_t(iterations) * size &&
        observed_complete_blocks == uint64_t(iterations) * (size / 64) &&
        observed_blocks == (core_sha256_accel::hardware_available() ? uint64_t(iterations) * (size / 64) : 0);
    correct = correct && accounting;
    check(correct, "memory throughput retains exact digest and compressor accounting");
    auto samples = [](const std::vector<double> & values) {
        for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
    };
    std::cout << std::setprecision(17) << "SHA256_MEMORY {\"bytes\":" << size
              << ",\"iterations_per_arm\":" << iterations << ",\"sha256\":" << std::quoted(expected)
              << ",\"hardware_available\":" << (core_sha256_accel::hardware_available() ? "true" : "false")
              << ",\"counter_scope\":\"requested-acceleration-arm/all-samples\""
              << ",\"input_bytes\":" << observed_input_bytes << ",\"complete_blocks\":" << observed_complete_blocks
              << ",\"block_accounting_passed\":" << (accounting ? "true" : "false")
              << ",\"accelerated_blocks\":" << observed_blocks << ",\"scalar_ms\":[";
    samples(scalar);
    std::cout << "],\"requested_acceleration_ms\":[";
    samples(accelerated);
    std::cout << "],\"passed\":" << (correct ? "true" : "false") << ",\"model_inference\":false}\n";
}

void file_identity(const std::string & path, const std::string & expected) {
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("whole-file probe requires a regular file");
    const auto size_before = std::filesystem::file_size(path);
    const auto modified_before = std::filesystem::last_write_time(path);
    const auto scalar_start = Clock::now();
    const std::string scalar = core_sha256::file(path);
    const double scalar_ms = elapsed_ms(scalar_start);
    core_sha256_accel::Stats stats;
    const auto accelerated_start = Clock::now();
    const std::string accelerated = core_sha256_accel::file(path, true, &stats);
    const double accelerated_ms = elapsed_ms(accelerated_start);
    const bool stable =
        size_before == std::filesystem::file_size(path) && modified_before == std::filesystem::last_write_time(path);
    const bool accounting = stats.input_bytes == size_before && stats.complete_blocks == size_before / 64 &&
                            stats.sha_ni_selected == stats.hardware_available &&
                            stats.accelerated_blocks == (stats.sha_ni_selected ? size_before / 64 : 0);
    const bool passed = stable && accounting && scalar == expected && accelerated == expected;
    check(passed, "actual file pinned digest identity");
    std::cout << std::setprecision(17) << "SHA256_FILE {\"bytes\":" << size_before
              << ",\"expected_sha256\":" << std::quoted(expected) << ",\"scalar_sha256\":" << std::quoted(scalar)
              << ",\"accelerated_sha256\":" << std::quoted(accelerated) << ",\"scalar_ms\":" << scalar_ms
              << ",\"requested_acceleration_ms\":" << accelerated_ms
              << ",\"hardware_available\":" << (stats.hardware_available ? "true" : "false")
              << ",\"sha_ni_selected\":" << (stats.sha_ni_selected ? "true" : "false")
              << ",\"input_bytes\":" << stats.input_bytes << ",\"complete_blocks\":" << stats.complete_blocks
              << ",\"accelerated_blocks\":" << stats.accelerated_blocks
              << ",\"block_accounting_passed\":" << (accounting ? "true" : "false")
              << ",\"file_metadata_unchanged\":" << (stable ? "true" : "false")
              << ",\"passed\":" << (passed ? "true" : "false") << ",\"model_inference\":false}\n";
}

int test_main(int argc, char ** argv) try {
    bool require_acceleration = false, require_fallback = false;
    size_t mib = 0;
    int iterations = 3;
    std::string file, expected;
    auto bounded_number = [](const std::string & value, unsigned maximum) {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("benchmark counts require decimal integers");
        const unsigned long long count = std::stoull(value);
        if (count > maximum) throw std::runtime_error("benchmark count exceeds its bounded limit");
        return unsigned(count);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i == argc) throw std::runtime_error("missing argument value");
            return argv[i];
        };
        if (arg == "--require-accel")
            require_acceleration = true;
        else if (arg == "--require-fallback")
            require_fallback = true;
        else if (arg == "--benchmark-mib")
            mib = bounded_number(value(), 256);
        else if (arg == "--iterations")
            iterations = int(bounded_number(value(), 5));
        else if (arg == "--file")
            file = value();
        else if (arg == "--expected-sha256")
            expected = value();
        else
            throw std::runtime_error("unknown test argument");
    }
    if ((require_acceleration && require_fallback) || mib > 256 || iterations < 1 || iterations > 5 ||
        file.empty() != expected.empty() ||
        (!expected.empty() &&
         (expected.size() != 64 || expected.find_first_not_of("0123456789abcdef") != std::string::npos)))
        throw std::runtime_error("invalid bounded benchmark arguments");
    if (require_acceleration && !core_sha256_accel::hardware_available())
        throw std::runtime_error("actual SHA-NI execution requested but unavailable");
    if (require_fallback && core_sha256_accel::hardware_available())
        throw std::runtime_error("scalar fallback probe requested but hardware path is available");
    controls();
    if (!failures && mib) memory_benchmark(mib, iterations);
    if (!failures && !file.empty()) file_identity(file, expected);
    std::cout << "SHA256_TEST {\"checks\":" << checks << ",\"failures\":" << failures
              << ",\"sha_ni_compiled\":" << (core_sha256_accel::compiled_sha_ni() ? "true" : "false")
              << ",\"hardware_available\":" << (core_sha256_accel::hardware_available() ? "true" : "false")
              << ",\"model_inference\":false}\n";
    return failures ? 1 : 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "SHA256_TEST_ERROR %s\n", error.what());
    return 2;
}

} // namespace

int main(int argc, char ** argv) {
    core_util::clean_exit(test_main(argc, argv));
}
