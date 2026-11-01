// Resident-model OXR batch runner for local backend/precision A/B: one model
// load, many cases. Each line of --batch is "<out-prefix>\t<image>\t<prompt>".
// For every case it writes <out-prefix>.json (decoded output and warm timings
// in the test-oxr-run schema subset) and, with --stages LIST, a prefill stage
// dump into <out-prefix>.stages/ (tests/oxr_stage_dump.cpp format, observer
// run with max_tokens 1, separate from the timed runs). Diagnostic only.
//
//   test-oxr-batch --batch cases.tsv --model m.gguf --image any.png [--repeat 2]
//       [--max-tokens 1024] [--hf-generation 1] [--stages llm.norm,llm.logits]
#include "oxr_test_common.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/stat.h>

namespace {
struct dump_state {
    std::string dir, index = "{";
    bool first = true;
    std::set<std::string> only;
};
void observe(void * user, const char * name, const float * values, size_t count, const int64_t * shape, int ndim) {
    auto & s = *static_cast<dump_state *>(user);
    if (!s.only.empty() && !s.only.count(name)) return;
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
    std::string batch, stages;
    std::vector<char *> rest{ argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--batch" && i + 1 < argc)
            batch = argv[++i];
        else if (a == "--stages" && i + 1 < argc)
            stages = argv[++i];
        else
            rest.push_back(argv[i]);
    }
    if (batch.empty()) throw std::invalid_argument("--batch is required");
    auto opt = oxr_test::parse(int(rest.size()), rest.data());
    using clock = std::chrono::steady_clock;
    const auto load_started = clock::now();
    auto ctx = oxr_test::load_model(opt);
    const double load_ms = std::chrono::duration<double, std::milli>(clock::now() - load_started).count();
    std::ifstream in(batch);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) throw std::invalid_argument("bad batch line");
        const std::string prefix = line.substr(0, t1), image = line.substr(t1 + 1, t2 - t1 - 1);
        opt.prompt = line.substr(t2 + 1);
        opt.image = image;
        if (!oxr_ocr_set_prompt(ctx.get(), opt.prompt.c_str())) throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
        const auto img = oxr_test::load_image(image);
        oxr_ocr_set_max_tokens(ctx.get(), opt.max_tokens);
        std::vector<double> times;
        std::vector<std::string> runs;
        for (int r = 0; r < opt.repeat; ++r) {
            int length = 0;
            const auto started = clock::now();
            const char * text = oxr_ocr_recognize_raw(ctx.get(), img.rgb.data(), img.width, img.height, 3, &length);
            const double ms = std::chrono::duration<double, std::milli>(clock::now() - started).count();
            if (!text || !oxr_ocr_last_ok(ctx.get())) throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
            times.push_back(ms);
            runs.push_back(oxr_test::result_json(ctx.get(), opt, img, text, length, ms));
        }
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(12) << runs.front().substr(0, runs.front().size() - 1)
            << ",\"schema\":\"crispembed.oxr.native-batch.v1\",\"device\":"
            << oxr_test::quote(core_env::on("CRISPEMBED_OXR_GPU") && !core_env::on("CRISPEMBED_OXR_FORCE_CPU") ? "gpu"
                                                                                                               : "cpu")
            << ",\"threads\":" << opt.threads << ",\"timing\":{\"load_ms\":" << load_ms << ",\"runs_ms\":";
        oxr_test::array(out, times.data(), int(times.size()));
        out << "},\"runs\":[";
        for (size_t i = 0; i < runs.size(); ++i) out << (i ? "," : "") << runs[i];
        out << "]}";
        std::ofstream(prefix + ".json") << out.str() << '\n';
        if (!stages.empty()) {
            dump_state state;
            state.dir = prefix + ".stages";
            mkdir(state.dir.c_str(), 0755);
            for (size_t p = 0; p <= stages.size();) {
                const size_t q = std::min(stages.find(',', p), stages.size());
                if (q > p) state.only.insert(stages.substr(p, q - p));
                p = q + 1;
            }
            oxr_ocr_set_max_tokens(ctx.get(), 1);
            oxr_ocr_set_stage_observer(ctx.get(), observe, &state);
            int length = 0;
            const char * text = oxr_ocr_recognize_raw(ctx.get(), img.rgb.data(), img.width, img.height, 3, &length);
            oxr_ocr_set_stage_observer(ctx.get(), nullptr, nullptr);
            if (!text) throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
            int count = 0;
            const auto * ids = oxr_ocr_completion_ids(ctx.get(), &count);
            state.index +=
                std::string(state.first ? "" : ",") + "\"first_token\":" + std::to_string(count ? ids[0] : -1) + "}";
            std::ofstream(state.dir + "/index.json") << state.index;
        }
        std::fprintf(stderr, "[oxr-batch] %s done (%zu runs)\n", prefix.c_str(), times.size());
    }
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = execute(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "OXR batch: %s\n", error.what());
    }
    core_util::clean_exit(code);
}
