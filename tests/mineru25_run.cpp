// Resident-context, machine-readable native MinerU qualification runner.
#include "crispembed.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "image_preprocess.h"
#include "mineru_layout.h"
#include "stb_image.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void prepare_image(std::vector<uint8_t> & rgb, int & w, int & h, const std::string & task) {
    if (task != "page") {
        auto plan = mineru::normalize_crop_geometry(w, h);
        if (task == "layout") {
            plan.padded_width = w;
            plan.padded_height = h;
            plan.paste_x = 0;
            plan.paste_y = 0;
            plan.output_width = 1036;
            plan.output_height = 1036;
        }
        if (w != plan.padded_width || h != plan.padded_height) {
            std::vector<uint8_t> padded(size_t(plan.padded_width) * plan.padded_height * 3, 255);
            for (int y = 0; y < h; ++y)
                std::memcpy(padded.data() + (size_t(y + plan.paste_y) * plan.padded_width + plan.paste_x) * 3,
                            rgb.data() + size_t(y) * w * 3, size_t(w) * 3);
            rgb.swap(padded);
            w = plan.padded_width;
            h = plan.padded_height;
        }
        if (w != plan.output_width || h != plan.output_height) {
            std::vector<float> resized(size_t(plan.output_width) * plan.output_height * 3);
            image_preproc::resize_bicubic_pil_u8_hwc(rgb.data(), h, w, resized.data(), plan.output_height,
                                                     plan.output_width, 3);
            rgb.resize(resized.size());
            for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t(resized[i]);
            w = plan.output_width;
            h = plan.output_height;
        }
    }
}

static int crispembed_test_main(int argc, char ** argv) {
    std::string model, image, task = "text";
    int max_tokens = 0, threads = 4, repeat = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) {
            std::cerr << "missing option value\n";
            return 2;
        }
        const std::string value = argv[++i];
        try {
            if (key == "--model")
                model = value;
            else if (key == "--image")
                image = value;
            else if (key == "--task")
                task = value;
            else if (key == "--max-tokens")
                max_tokens = std::stoi(value);
            else if (key == "--threads")
                threads = std::stoi(value);
            else if (key == "--repeat")
                repeat = std::stoi(value);
            else {
                std::cerr << "unknown option: " << key << '\n';
                return 2;
            }
        } catch (const std::exception &) {
            std::cerr << "invalid value for " << key << '\n';
            return 2;
        }
    }
    if (model.empty() || image.empty() || repeat < 1 || repeat > 100 || threads < 1 || max_tokens < 0 ||
        max_tokens > 8192 ||
        (task != "text" && task != "table" && task != "formula" && task != "layout" && task != "image" &&
         task != "page")) {
        std::cerr << "Usage: test-mineru25-run --model MODEL --image IMAGE --task text|table|formula|layout|image|page "
                     "--max-tokens N --threads N --repeat N\n";
        return 2;
    }
    int w = 0, h = 0, channels = 0;
    unsigned char * raw = stbi_load(image.c_str(), &w, &h, &channels, 4);
    if (!raw || w <= 0 || h <= 0 || uint64_t(w) * h > 100000000ULL) {
        if (raw) stbi_image_free(raw);
        std::cerr << "image decode failed\n";
        return 2;
    }
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < size_t(w) * h; ++i)
        for (int c = 0; c < 3; ++c) {
            rgb[i * 3 + c] = raw[i * 4 + c]; // PIL.convert("RGB") discards alpha.
        }
    stbi_image_free(raw);
    const int original_w = w, original_h = h;
    const auto started = std::chrono::steady_clock::now();
    auto * ctx = crispembed_mineru_init(model.c_str(), threads);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (!ctx) {
        std::cerr << "model/profile/gate initialization failed\n";
        return 1;
    }
    const char * prompt = mineru::prompt_for_type(task == "formula"  ? "equation"
                                                  : task == "layout" ? "[layout]"
                                                                     : task);
    std::vector<double> times;
    std::vector<std::string> results;
    for (int run = 0; run < repeat; ++run) {
        const auto t = std::chrono::steady_clock::now();
        auto prepared = rgb;
        int pw = w, ph = h;
        prepare_image(prepared, pw, ph, task);
        const char * json = task == "page"
                                ? crispembed_mineru_extract(ctx, prepared.data(), pw, ph, 3, 0, max_tokens)
                                : crispembed_mineru_predict(ctx, prepared.data(), pw, ph, 3, prompt, max_tokens);
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
        if (!json) {
            std::cerr << crispembed_mineru_last_error(ctx) << '\n';
            crispembed_mineru_free(ctx);
            return 1;
        }
        results.emplace_back(json);
    }
    crispembed_mineru_free(ctx);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    // Repeat0 carries cold inference; runs1..N-1 are warm. Outputs from EVERY
    // run are retained so timing never hides drift/stale context state.
    out << std::setprecision(12) << results.front().substr(0, results.front().size() - 1)
        << ",\"device\":" << mineru::json_quote(core_env::on("QWEN2VL_OCR_FORCE_CPU") ? "cpu" : "auto")
        << ",\"threads\":" << threads << ",\"task\":" << mineru::json_quote(task)
        << ",\"original_image_width\":" << original_w << ",\"original_image_height\":" << original_h
        << ",\"timing\":{\"load_ms\":" << load_ms << ",\"runs_ms\":[";
    for (size_t i = 0; i < times.size(); ++i) {
        if (i) out << ',';
        out << times[i];
    }
    out << "]},\"runs\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i) out << ',';
        out << results[i];
    }
    out << "]}";
    std::cout << out.str() << '\n';
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
