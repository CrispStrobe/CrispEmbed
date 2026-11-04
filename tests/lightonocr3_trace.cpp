// Separate observer build; never link this probe into release artifacts.
#include "lightonocr_test.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/sha256.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct capture {
    std::filesystem::path path;
    bool failed = false;
};
static void observe(const char * name, const float * data, size_t size, void * opaque) {
    auto & c = *static_cast<capture *>(opaque);
    std::ofstream out(c.path / (std::string(name) + ".f32"), std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), std::streamsize(size * sizeof(float)));
    if (!out) c.failed = true;
}
static void write_ids(std::ostream & out, const int32_t * ids, size_t count) {
    out << '[';
    for (size_t i = 0; i < count; ++i) out << (i ? "," : "") << ids[i];
    out << ']';
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s model.gguf image output-dir max-tokens [forced-ids.txt|-] [observe|plain]\n",
                argv[0]);
        return 2;
    }
    size_t parsed = 0;
    const int cap = std::stoi(argv[4], &parsed);
    if (parsed != std::string(argv[4]).size() || cap <= 0 || cap > 256) return 2;
    const std::string arm = argc < 7 ? "observe" : argv[6];
    if (arm != "observe" && arm != "plain") return 2;
    capture c{ argv[3] };
    if (std::filesystem::exists(c.path) && !std::filesystem::is_empty(c.path)) return 2;
    std::filesystem::create_directories(c.path);
    const auto image_sha = core_sha256::file(argv[2]);
    if (image_sha.empty()) return 2;
    std::vector<int32_t> forced;
    if (argc >= 6 && std::string(argv[5]) != "-") {
        std::ifstream in(argv[5]);
        int id;
        while (in >> id) forced.push_back(id);
        if (!in.eof() || forced.size() != size_t(cap) || arm == "plain") return 2;
    }
    auto * ctx = lightonocr_init(argv[1], 4);
    if (!ctx) return 1;
    if (!lightonocr_test_is_profile3(ctx)) {
        lightonocr_free(ctx);
        return 2;
    }
    if (arm == "observe") lightonocr_test_set_observer(ctx, observe, &c);
    if (!forced.empty()) lightonocr_test_force_tokens(ctx, forced.data(), forced.size());
    lightonocr_set_max_tokens(ctx, cap);
    const auto start = std::chrono::steady_clock::now();
    int len = 0;
    const char * text = lightonocr_recognize_file(ctx, argv[2], &len);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    size_t count = 0, argmax_count = 0;
    const auto * ids = lightonocr_test_generated_ids(ctx, &count, false);
    const auto * argmax = lightonocr_test_generated_ids(ctx, &argmax_count, true);
    bool ids_valid = count > 0 && count <= size_t(cap) && count == argmax_count;
    for (size_t i = 0; i < count; ++i) {
        ids_valid = ids_valid && ids[i] >= 0 && ids[i] < 151936;
        if (i + 1 < count && (ids[i] == 151645 || ids[i] == 151643)) ids_valid = false;
    }
    ids_valid = ids_valid && (count == size_t(cap) || ids[count - 1] == 151645 || ids[count - 1] == 151643);
    std::ofstream decoded(c.path / "decoded.txt", std::ios::binary);
    if (text) decoded.write(text, len);
    decoded.close();
    std::ofstream(c.path / "seconds.txt") << seconds << '\n';
    std::ofstream receipt(c.path / "trace.json");
    receipt << "{\"schema_version\":2,\"profile\":\"lightonocr3-1b\",\"image_sha256\":\"" << image_sha
            << "\",\"mode\":\"" << (core_env::on("CRISPEMBED_LIGHTONOCR3_GROUNDING") ? "grounding" : "transcription")
            << "\",\"observed\":" << (arm == "observe" ? "true" : "false")
            << ",\"model_bytes\":" << std::filesystem::file_size(argv[1]) << ",\"max_new_tokens\":" << cap
            << ",\"seconds\":" << seconds << ",\"forced_ids\":";
    write_ids(receipt, forced.data(), forced.size());
    receipt << ",\"generated_ids\":";
    write_ids(receipt, ids, count);
    receipt << ",\"argmax_ids\":";
    write_ids(receipt, argmax, argmax_count);
    receipt << "}\n";
    receipt.close();
    const bool ok = text && decoded && receipt && ids_valid && !c.failed;
    lightonocr_free(ctx);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(crispembed_test_main(argc, argv));
    } catch (const std::exception & e) {
        fprintf(stderr, "lightonocr3-trace: %s\n", e.what());
        core_util::clean_exit(2);
    }
}
