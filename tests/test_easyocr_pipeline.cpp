#include "easyocr_pipeline.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

static std::string json_escape(const std::string & text) {
    std::string out;
    for (const char c : text) {
        if (c == '\\')
            out += "\\\\";
        else if (c == '"')
            out += "\\\"";
        else if (c == '\n')
            out += "\\n";
        else if (c == '\r')
            out += "\\r";
        else if (c == '\t')
            out += "\\t";
        else
            out += c;
    }
    return out;
}

static bool write_manifest(const char * path, const char * image, easyocr_layout::ordering_mode mode,
                           const std::vector<easyocr_pipeline::result> & results) {
    std::ofstream out(path);
    if (!out) return false;
    out << "{\n  \"schema\": \"crispembed.easyocr.postprocess.v1\",\n"
           "  \"source\": \"CrispEmbed native easyocr_pipeline\",\n"
        << "  \"image\": \"" << json_escape(image ? image : "") << "\",\n"
        << "  \"mode\": \"" << (mode == easyocr_layout::ordering_mode::words ? "words" : "lines")
        << "\",\n"
           "  \"records\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & item = results[i];
        out << "    {\"index\": " << i << ", \"line\": " << item.word.line << ", \"text\": \""
            << json_escape(item.word.text) << "\", \"confidence\": " << item.word.confidence
            << ", \"detector_confidence\": " << item.detector_confidence << ", \"box\": [" << item.word.x << ", "
            << item.word.y << ", " << item.word.w << ", " << item.word.h << "], \"crop\": [" << item.crop_x << ", "
            << item.crop_y << ", " << item.crop_w << ", " << item.crop_h << "], \"normalized_box\": ["
            << item.normalized.x0 << ", " << item.normalized.y0 << ", " << item.normalized.x1 << ", "
            << item.normalized.y1 << "]}" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return out.good();
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: %s <dbnet.gguf> <easyocr.gguf> <image> <lines|words> [manifest.json]\n", argv[0]);
        return 2;
    }
    easyocr_pipeline::context * ctx = nullptr;
    if (!easyocr_pipeline::load(&ctx, argv[1], argv[2], 1)) return 3;
    const auto mode = std::strcmp(argv[4], "words") == 0 ? easyocr_layout::ordering_mode::words
                                                         : easyocr_layout::ordering_mode::lines;
    easyocr_pipeline::set_ordering_mode(ctx, mode);
    const auto results = easyocr_pipeline::run_file(ctx, argv[3]);
    std::printf("easyocr-pipeline mode=%s results=%zu\n",
                mode == easyocr_layout::ordering_mode::words ? "words" : "lines", results.size());
    if (results.empty()) {
        easyocr_pipeline::free(ctx);
        return 4;
    }
    int previous_line = -1;
    float previous_x = -1.0f;
    for (const auto & item : results) {
        const auto & box = item.normalized;
        if (box.x0 < 0 || box.y0 < 0 || box.x1 < box.x0 || box.y1 < box.y0 || box.x1 > 1000 || box.y1 > 1000) {
            easyocr_pipeline::free(ctx);
            return 6;
        }
        if (mode == easyocr_layout::ordering_mode::words) {
            if (item.word.line < previous_line || (item.word.line == previous_line && item.word.x < previous_x)) {
                easyocr_pipeline::free(ctx);
                return 7;
            }
            previous_line = item.word.line;
            previous_x = item.word.x;
        }
    }
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & item = results[i];
        std::printf("result=%zu line=%d box=%.1f,%.1f %.1fx%.1f det_conf=%.4f rec_conf=%.4f norm=%d,%d,%d,%d text=%s\n",
                    i, item.word.line, item.word.x, item.word.y, item.word.w, item.word.h, item.detector_confidence,
                    item.word.confidence, item.normalized.x0, item.normalized.y0, item.normalized.x1,
                    item.normalized.y1, item.word.text.c_str());
    }
    if (argc == 6 && !write_manifest(argv[5], argv[3], mode, results)) {
        easyocr_pipeline::free(ctx);
        return 8;
    }
    easyocr_pipeline::free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
