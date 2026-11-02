// Ordinary, unobserved, resident-context native OXR timing/output runner.
#include "oxr_test_common.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"

#include <chrono>
#include <cstdio>
#include <iostream>

static int execute(int argc, char ** argv) {
    const auto opt = oxr_test::parse(argc, argv);
    const auto img = oxr_test::load_image(opt.image);
    using clock = std::chrono::steady_clock;
    const auto load_started = clock::now();
    auto ctx = oxr_test::load_model(opt);
    const double load_ms = std::chrono::duration<double, std::milli>(clock::now() - load_started).count();
    auto * metadata = core_gguf::open_metadata(opt.model.c_str());
    if (!metadata) throw std::runtime_error("could not read native conversion recipe");
    const std::string recipe = core_gguf::kv_str(metadata, "oxr.conversion.recipe", "");
    core_gguf::free_metadata(metadata);
    std::vector<double> times;
    std::vector<std::string> runs;
    for (int i = 0; i < opt.repeat; ++i) {
        int length = 0;
        const auto started = clock::now();
        const char * text = oxr_ocr_recognize_raw(ctx.get(), img.rgb.data(), img.width, img.height, 3, &length);
        const double elapsed = std::chrono::duration<double, std::milli>(clock::now() - started).count();
        if (!text || !oxr_ocr_last_ok(ctx.get())) throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
        times.push_back(elapsed);
        runs.push_back(oxr_test::result_json(ctx.get(), opt, img, text, length, elapsed));
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12) << runs.front().substr(0, runs.front().size() - 1)
        << ",\"schema\":\"crispembed.oxr.native.v1\",\"device\":"
        << oxr_test::quote(core_env::on("CRISPEMBED_OXR_FORCE_CPU") ? "cpu" : "auto") << ",\"threads\":" << opt.threads
        << ",\"conversion_recipe\":" << oxr_test::quote(recipe) << ",\"timing\":{\"load_ms\":" << load_ms
        << ",\"runs_ms\":";
    oxr_test::array(out, times.data(), int(times.size()));
    out << "},\"runs\":[";
    for (size_t i = 0; i < runs.size(); ++i) out << (i ? "," : "") << runs[i];
    out << "]}";
    std::cout << out.str() << '\n';
    return 0;
}

int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = execute(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "OXR runner: %s\n", error.what());
    }
    core_util::clean_exit(code);
}
