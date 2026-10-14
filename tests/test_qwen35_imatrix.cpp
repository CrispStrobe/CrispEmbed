#include "core/clean_exit.h"
#include "qwen35_ocr.h"
#include "qwen35_math.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
struct stats {
    uint64_t count = 0;
    std::vector<double> sum;
};
using observations = std::map<std::string, stats>;
static bool collect(const char * name, const float * x, int width, int rows, void * user) {
    if (!name || std::string(name).find("l.layers.") != 0 || std::string(name).find(".mlp.") == std::string::npos)
        return true;
    if (!x || !user || (width != 1024 && width != 3584) || rows < 1 || rows > 4096) return false;
    auto & entry = (*static_cast<observations *>(user))[name];
    if (entry.sum.empty()) entry.sum.resize(width);
    if (entry.sum.size() != size_t(width) || entry.count + rows > 1000000000) return false;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < width; ++c) {
            const double value = x[size_t(r) * width + c];
            if (!std::isfinite(value)) return false;
            entry.sum[c] += value * value;
            if (!std::isfinite(entry.sum[c])) return false;
        }
    entry.count += rows;
    return true;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 4) return 2; // model, output stats, independent calibration PNGs
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_QWEN35_IMATRIX", "0");
    observations data;
    if (qwen35_ocr::set_projection_observer(collect, &data)) return 1;
    gate("CRISPEMBED_QWEN35_IMATRIX", "1");
    if (!qwen35_ocr::set_projection_observer(collect, &data)) return 1;
    using owner = std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)>;
    owner context(qwen35_ocr_init(argv[1], 4), qwen35_ocr_free);
    if (!context || !qwen35_ocr_set_max_tokens(context.get(), 64)) return 2;
    gate("CRISPEMBED_QWEN35_IMATRIX", "0");
    int disabled_length = 0, disabled_count = 0;
    const char * disabled = qwen35_ocr_recognize_file(context.get(), argv[3], nullptr, &disabled_length);
    const auto * disabled_ids = qwen35_ocr_token_ids(context.get(), &disabled_count);
    if (!disabled || !disabled_ids || disabled_count < 1 || !data.empty()) return 1;
    const std::string disabled_text(disabled, size_t(disabled_length));
    const std::vector<int> control_ids(disabled_ids, disabled_ids + disabled_count);
    gate("CRISPEMBED_QWEN35_IMATRIX", "1");
    for (int i = 3; i < argc; ++i) {
        int length = 0, count = 0;
        const char * text = qwen35_ocr_recognize_file(context.get(), argv[i], nullptr, &length);
        const auto * ids = qwen35_ocr_token_ids(context.get(), &count);
        if (!text || !ids || count < 1 || length < 0) return 2;
        if (i == 3 &&
            (std::string(text, size_t(length)) != disabled_text || std::vector<int>(ids, ids + count) != control_ids))
            return 1;
        std::printf("CALIB index=%d stop=%d ids:", i - 3, qwen35_ocr_last_stop(context.get()));
        for (int j = 0; j < count; ++j) std::printf(" %d", ids[j]);
        std::printf("\nCALIB utf8_hex:");
        for (int j = 0; j < length; ++j) std::printf("%02x", static_cast<unsigned char>(text[j]));
        std::printf("\n");
    }
    if (!qwen35_ocr::set_projection_observer(nullptr, nullptr) || data.size() != 72) return 1;
    gate("CRISPEMBED_QWEN35_IMATRIX", "0");
    std::ofstream output(argv[2]);
    output << std::setprecision(17);
    uint64_t count = 0;
    for (const auto & entry : data) {
        if (!entry.second.count) return 1;
        if (!count) count = entry.second.count;
        if (entry.second.count != count) return 1;
        output << entry.first << ' ' << count << ' ' << entry.second.sum.size();
        for (double value : entry.second.sum) output << ' ' << value;
        output << '\n';
    }
    output.close();
    if (!output) return 1;
    std::printf("PASS native FFN importance: 72 matrices, rows=%llu each, independent image/greedy inputs, literal0 "
                "registration rejected\n",
                static_cast<unsigned long long>(count));
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
