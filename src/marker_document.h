// SPDX-License-Identifier: Apache-2.0
// Native document processing based on Marker 2.1.0. See NOTICE.marker.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace crisp_marker {

struct point {
    double x = 0, y = 0;
};
struct box {
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    double width() const { return x1 - x0; }
    double height() const { return y1 - y0; }
};
struct image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};
struct span {
    std::string text;
    box bbox;
    double font_size = 0;
    bool bold = false, italic = false, monospace = false;
    std::string url;
};
struct line {
    box bbox;
    std::vector<span> spans;
};
struct block {
    std::string id, type = "Text", text, html, raw, raw_label, source;
    box bbox;
    std::vector<point> polygon;
    std::vector<line> lines;
    std::vector<block> children;
    std::map<int, std::string> section_hierarchy;
    std::vector<std::string> warnings;
    std::string image_name;
    int order = 0, heading_level = 0, angle = 0, generated_tokens = 0;
    double confidence = 0;
    bool token_cap_reached = false, ignored = false, continuation = false;
    // Additive (2026-10-31): provider-specific exact metadata, e.g. the MinerU
    // upstream-profile fields ("mineru.*"). Ignored by the Marker renderers.
    std::map<std::string, std::string> attributes;
};
struct page {
    int index = 0;
    double width = 0, height = 0;
    std::vector<line> text_lines;
    std::vector<block> blocks;
    std::vector<std::string> warnings;
    std::string extraction_method;
    std::string raw_layout;
    int layout_generated_tokens = 0;
    bool layout_token_cap_reached = false;
};
struct asset {
    std::string name, mime_type = "image/png";
    std::vector<uint8_t> bytes;
};
struct document {
    std::string source_name, backend;
    std::vector<page> pages;
    std::vector<asset> assets;
    std::vector<std::string> warnings;
    std::map<std::string, std::string> metadata;
};
struct options {
    // 0: use PDF text where sound, TeleOCR layout and selective OCR.
    // 1: model-free digital text with heuristic layout. 2: force TeleOCR OCR.
    int mode = 0, n_threads = 4, first_page = 0, last_page = -1;
    int dpi = 144, max_tokens = 1024, max_layout_tokens = 2048;
    int max_blocks = 512, max_pages = 1000;
    size_t max_pixels = 16000000, max_asset_bytes = 128 * 1024 * 1024;
    bool include_page_headers = false, extract_images = true;
    std::string model_path, pdfium_path, password;
};

// Deterministic structure/text processing and four renderers. Renderers never
// execute raw model HTML; table HTML is generated from parsed OTSL cells.
std::string escape_html(const std::string & text);
std::string text_of(const line & value);
bool text_is_usable(const std::string & value);
void build_text_blocks(page & value);
void process_document(document & value, const options & opts);
std::string render_document(const document & value, const std::string & format);
std::string render_metadata(const document & value);
std::string otsl_to_html(const std::string & text, std::vector<std::string> & warnings);
std::string pdf_table_html(const std::vector<line> & lines, std::vector<std::string> & warnings);
std::vector<block> parse_teleocr_layout(const std::string & text, double width, double height, int max_blocks,
                                        std::vector<std::string> & warnings);

} // namespace crisp_marker
