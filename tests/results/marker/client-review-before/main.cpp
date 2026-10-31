// SPDX-License-Identifier: Apache-2.0
#include "crispembed.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
void usage() {
    std::cerr << "Usage: crispembed-marker INPUT [--output FILE] [--format markdown|html|json|chunks]\n"
                 "  --mode auto|text|ocr  --model TELEOCR.gguf  --pdfium LIBRARY\n"
                 "  --threads N --first-page N --last-page N --dpi N\n"
                 "  --max-tokens N --max-layout-tokens N --max-blocks N --max-pages N\n"
                 "  --max-pixels N --max-asset-bytes N --include-headers --no-images\n"
                 "Requires CRISPEMBED_MARKER=1; auto/ocr also require CRISPEMBED_TELEOCR=1.\n"
                 "Native Marker-style processing with TeleOCR, not original Surya model parity.\n";
}
int integer(const std::string & value) {
    size_t end = 0; int n = std::stoi(value, &end);
    if (end != value.size()) throw std::runtime_error("invalid integer: " + value);
    return n;
}
uint64_t unsigned_integer(const std::string & value) {
    if (value.empty() || value[0] == '-') throw std::runtime_error("invalid byte/pixel limit");
    size_t end = 0; uint64_t n = std::stoull(value, &end);
    if (end != value.size()) throw std::runtime_error("invalid integer: " + value);
    return n;
}
void write_file(const std::filesystem::path & p, const void * bytes, size_t count) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create output: " + p.string());
    f.write(static_cast<const char *>(bytes), static_cast<std::streamsize>(count));
    f.close();
    if (!f) throw std::runtime_error("cannot finish output: " + p.string());
}
}
int main(int argc, char ** argv) {
    try {
        auto cfg = crispembed_marker_default_config();
        std::string input, output, format = "markdown", model, pdfium;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") { usage(); return 0; }
            if (a == "--include-headers") { cfg.include_page_headers = 1; continue; }
            if (a == "--no-images") { cfg.extract_images = 0; continue; }
            if (a.rfind("--", 0) != 0) {
                if (!input.empty()) throw std::runtime_error("exactly one input document is required");
                input = a; continue;
            }
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            const std::string v = argv[++i];
            if (a == "--model") model = v;
            else if (a == "--pdfium") pdfium = v;
            else if (a == "--output" || a == "--output-file") output = v;
            else if (a == "--format") format = v;
            else if (a == "--mode") {
                if (v == "auto") cfg.mode = 0;
                else if (v == "text") cfg.mode = 1;
                else if (v == "ocr") cfg.mode = 2;
                else throw std::runtime_error("mode must be auto, text or ocr");
            } else if (a == "--threads") cfg.n_threads = integer(v);
            else if (a == "--first-page") cfg.first_page = integer(v);
            else if (a == "--last-page") cfg.last_page = integer(v);
            else if (a == "--dpi") cfg.dpi = integer(v);
            else if (a == "--max-tokens") cfg.max_tokens = integer(v);
            else if (a == "--max-layout-tokens") cfg.max_layout_tokens = integer(v);
            else if (a == "--max-blocks") cfg.max_blocks = integer(v);
            else if (a == "--max-pages") cfg.max_pages = integer(v);
            else if (a == "--max-pixels") cfg.max_pixels = unsigned_integer(v);
            else if (a == "--max-asset-bytes") cfg.max_asset_bytes = unsigned_integer(v);
            else throw std::runtime_error("unknown argument: " + a);
        }
        if (input.empty()) { usage(); return 2; }
        if (format != "markdown" && format != "html" && format != "json" && format != "chunks")
            throw std::runtime_error("format must be markdown, html, json or chunks");
        cfg.model_path = model.c_str(); cfg.pdfium_path = pdfium.c_str();
        std::unique_ptr<crispembed_marker_context, decltype(&crispembed_marker_free)> ctx(
            crispembed_marker_init(&cfg), crispembed_marker_free);
        if (!ctx) throw std::runtime_error(crispembed_marker_error(nullptr));
        if (!crispembed_marker_convert(ctx.get(), input.c_str())) throw std::runtime_error(crispembed_marker_error(ctx.get()));
        const char * rendered = crispembed_marker_render(ctx.get(), format.c_str());
        if (!rendered) throw std::runtime_error(crispembed_marker_error(ctx.get()));
        const std::string text(rendered);
        const int assets = crispembed_marker_asset_count(ctx.get());
        if (output.empty()) {
            if (assets) throw std::runtime_error("--output FILE is required to save extracted figure assets");
            std::cout << text;
            if (text.empty() || text.back() != '\n') std::cout << '\n';
        } else {
            const std::filesystem::path path(output);
            const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
            std::filesystem::create_directories(parent);
            write_file(path, text.data(), text.size());
            const std::string meta(crispembed_marker_metadata(ctx.get()));
            write_file(parent / (path.stem().string() + "_meta.json"), meta.data(), meta.size());
            for (int i = 0; i < assets; ++i) {
                size_t n = 0; const auto * data = crispembed_marker_asset_data(ctx.get(), i, &n);
                write_file(parent / crispembed_marker_asset_name(ctx.get(), i), data, n);
            }
            std::cerr << "Converted " << input << " -> " << output << " (" << assets << " image assets)\n";
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "crispembed-marker: " << e.what() << '\n'; return 1;
    }
}
