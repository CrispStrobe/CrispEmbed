// Separate observer build; never link this probe into release artifacts.
#include "lightonocr_test.h"
#include "core/clean_exit.h"
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
static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s model.gguf image output-dir max-tokens [forced-ids.txt|-] [observe|plain]\n",
                argv[0]);
        return 2;
    }
    const int cap = std::stoi(argv[4]);
    if (cap <= 0 || cap > 256) return 2;
    capture c{ argv[3] };
    std::filesystem::create_directories(c.path);
    auto * ctx = lightonocr_init(argv[1], 4);
    if (!ctx) return 1;
    if (argc < 7 || std::string(argv[6]) != "plain") lightonocr_test_set_observer(ctx, observe, &c);
    std::vector<int32_t> forced;
    if (argc >= 6 && std::string(argv[5]) != "-") {
        std::ifstream in(argv[5]);
        int id;
        while (in >> id) forced.push_back(id);
        if (forced.size() != size_t(cap)) {
            lightonocr_free(ctx);
            return 2;
        }
        lightonocr_test_force_tokens(ctx, forced.data(), forced.size());
    }
    lightonocr_set_max_tokens(ctx, cap);
    const auto start = std::chrono::steady_clock::now();
    int len = 0;
    const char * text = lightonocr_recognize_file(ctx, argv[2], &len);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream decoded(c.path / "decoded.txt", std::ios::binary);
    if (text) decoded.write(text, len);
    std::ofstream(c.path / "seconds.txt") << seconds << '\n';
    const bool ok = text && decoded && !c.failed;
    lightonocr_free(ctx);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
