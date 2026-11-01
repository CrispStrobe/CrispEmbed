// Native OXR stage dump for backend/precision A/B (Metal vs CPU, quantized vs
// F32). Runs one observed prefill (max_tokens forced to 1) and writes every
// observed stage as raw little-endian F32 plus index.json into --dump-dir.
// Compare two dumps with tools/oxr_stage_compare.py. Diagnostic only: the
// original-Python stage protocol remains tests/oxr_diff.cpp.
//
//   test-oxr-stage-dump --dump-dir DIR --model m.gguf --image page.png --prompt "..." [--threads 4]
#include "oxr_test_common.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <sys/stat.h>

namespace {
struct dump_state {
    std::string dir;
    std::string index = "{";
    bool first = true;
    std::set<std::string> skip, only;
};
void observe(void * user, const char * name, const float * values, size_t count, const int64_t * shape, int ndim) {
    auto & s = *static_cast<dump_state *>(user);
    if (s.skip.count(name) || (!s.only.empty() && !s.only.count(name))) return;
    FILE * f = std::fopen((s.dir + "/" + name + ".f32").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write dump");
    std::fwrite(values, sizeof(float), count, f);
    std::fclose(f);
    s.index += std::string(s.first ? "" : ",") + "\"" + name + "\":[";
    for (int d = 0; d < ndim; ++d) s.index += (d ? "," : "") + std::to_string(shape[d]);
    s.index += "]";
    s.first = false;
}
int execute(int argc, char ** argv) {
    dump_state state;
    std::vector<char *> rest{ argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dump-dir" && i + 1 < argc)
            state.dir = argv[++i];
        else if (a == "--skip-large") {
            // The visual logits/probs are 65532 x tokens each (hundreds of MB).
            state.skip = { "visual.logits", "visual.probs" };
        } else
            rest.push_back(argv[i]);
    }
    if (state.dir.empty()) throw std::invalid_argument("--dump-dir is required");
    mkdir(state.dir.c_str(), 0755);
    auto opt = oxr_test::parse(int(rest.size()), rest.data());
    opt.max_tokens = 1;
    const auto img = oxr_test::load_image(opt.image);
    auto ctx = oxr_test::load_model(opt);
    oxr_ocr_set_stage_observer(ctx.get(), observe, &state);
    int length = 0;
    const char * text = oxr_ocr_recognize_raw(ctx.get(), img.rgb.data(), img.width, img.height, 3, &length);
    if (!text || !oxr_ocr_last_ok(ctx.get())) throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
    int count = 0;
    const auto * ids = oxr_ocr_completion_ids(ctx.get(), &count);
    state.index += std::string(state.first ? "" : ",") + "\"first_token\":" + std::to_string(count ? ids[0] : -1) + "}";
    FILE * f = std::fopen((state.dir + "/index.json").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write index");
    std::fputs(state.index.c_str(), f);
    std::fclose(f);
    std::cout << state.index << '\n';
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = execute(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "OXR stage dump: %s\n", error.what());
    }
    core_util::clean_exit(code);
}
