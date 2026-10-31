// mineru_document.cpp - see mineru_document.h. Layout parsing and extraction
// preparation translated from mineru-vl-utils mineru_client.py
// (c9797cde8f54cc48306196fc6aebe73ea31dd4a3, Apache-2.0, MinerU Team);
// modified: C++, integrated with CrispEmbed's layout/router/document model.
#include "mineru_document.h"

#include "image_preprocess.h"
#include "mineru_layout.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace mineru_doc {

using mineru_img::rgb_image;
using mineru_pp::block;
using mineru_pp::block_list;
using mineru_pp::block_ptr;
using pyre::ustring;

const char * profile_name() {
    return "mineru-vl-utils-c9797cde-full-v1";
}

uint64_t fnv1a(const std::vector<uint8_t> & bytes) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 1099511628211ULL;
    }
    return h;
}

namespace {
const std::set<std::string> & block_types() {
    static const std::set<std::string> t = { "text",          "title",          "doc_title",      "paragraph_title",
                                             "table",         "equation",       "formula_number", "code",
                                             "algorithm",     "aside_text",     "ref_text",       "index",
                                             "phonetic",      "list_item",      "table_caption",  "image_caption",
                                             "code_caption",  "caption",        "table_footnote", "image_footnote",
                                             "footnote",      "header",         "footer",         "page_number",
                                             "page_footnote", "image",          "chart",          "list",
                                             "image_block",   "equation_block", "unknown" };
    return t;
}

// Python int() of a \d+ match, bounded: returns -1 when > 1000.
long bounded_int(const ustring & digits) {
    long v = 0;
    for (char32_t c : digits) {
        long d = -1;
        if (c >= '0' && c <= '9')
            d = static_cast<long>(c - '0');
        else {
            // Unicode Nd blocks are contiguous runs of ten starting at zero.
            const uint32_t * r = pyunicode_tables::decimal_ranges;
            const size_t n = sizeof(pyunicode_tables::decimal_ranges) / 8;
            for (size_t i = 0; i < n; ++i)
                if (c >= r[i * 2] && c <= r[i * 2 + 1]) d = static_cast<long>((c - r[i * 2]) % 10);
        }
        if (d < 0) throw std::invalid_argument("non-decimal digit");
        v = v * 10 + d;
        if (v > 1000) return -1;
    }
    return v;
}

double inter_area(const std::array<double, 4> & a, const std::array<double, 4> & b) {
    const double x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]), x2 = std::min(a[2], b[2]),
                 y2 = std::min(a[3], b[3]);
    if (x2 <= x1 || y2 <= y1) return 0.0;
    return (x2 - x1) * (y2 - y1);
}
double cover_ratio(const std::array<double, 4> & inner, const std::array<double, 4> & outer) {
    const double a = std::max(0.0, inner[2] - inner[0]) * std::max(0.0, inner[3] - inner[1]);
    if (a == 0) return 0.0;
    return inter_area(inner, outer) / a;
}
std::set<size_t> covered_indices(const block_list & blocks, const std::set<std::string> & candidates,
                                 const std::set<std::string> & containers, double threshold = 0.9) {
    std::vector<size_t> cont;
    for (size_t i = 0; i < blocks.size(); ++i)
        if (containers.count(blocks[i]->type)) cont.push_back(i);
    std::set<size_t> out;
    if (cont.empty()) return out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (!candidates.count(blocks[i]->type)) continue;
        for (size_t c : cont) {
            if (c == i) continue;
            if (cover_ratio(blocks[i]->bbox, blocks[c]->bbox) >= threshold) {
                out.insert(i);
                break;
            }
        }
    }
    return out;
}

rgb_image resize_bicubic(const rgb_image & src, int nw, int nh) {
    if (src.width == nw && src.height == nh) return src;
    std::vector<float> tmp(static_cast<size_t>(nw) * nh * 3);
    image_preproc::resize_bicubic_pil_u8_hwc(src.px.data(), src.height, src.width, tmp.data(), nh, nw, 3);
    rgb_image out;
    out.width = nw;
    out.height = nh;
    out.px.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) out.px[i] = static_cast<uint8_t>(std::clamp(tmp[i], 0.f, 255.f));
    return out;
}

int round_half_even(double x) {
    return mineru::round_even(x);
}
} // namespace

layout_detect::label_id label_for(const std::string & t) {
    using L = layout_detect::label_id;
    if (t == "table") return L::table;
    if (t == "equation" || t == "equation_block") return L::formula;
    if (t == "image" || t == "chart" || t == "image_block" || t == "unknown") return L::picture;
    if (t == "title" || t == "doc_title") return L::title;
    if (t == "paragraph_title") return L::section_header;
    if (t == "code" || t == "algorithm") return L::code;
    if (t == "list_item" || t == "list") return L::list_item;
    if (t == "index") return L::document_index;
    if (t == "header") return L::page_header;
    if (t == "footer" || t == "page_number") return L::page_footer;
    if (t == "table_caption" || t == "image_caption" || t == "code_caption" || t == "caption") return L::caption;
    if (t == "table_footnote" || t == "image_footnote" || t == "footnote" || t == "page_footnote") return L::footnote;
    return L::text;
}

namespace {
const char * label_string(layout_detect::label_id id) {
    // Same strings as layout_detect::label_name (heron class names).
    static const char * names[] = { "caption",
                                    "footnote",
                                    "formula",
                                    "list_item",
                                    "page_footer",
                                    "page_header",
                                    "picture",
                                    "section_header",
                                    "table",
                                    "text",
                                    "title",
                                    "document_index",
                                    "code",
                                    "checkbox_selected",
                                    "checkbox_unselected",
                                    "form",
                                    "key_value_region" };
    const int i = static_cast<int>(id);
    return i >= 0 && i < static_cast<int>(layout_detect::label_id::NUM_CLASSES) ? names[i] : "unknown";
}
} // namespace

block_list parse_layout_output(const std::string & text) {
    static const pyre::regex layout_re(
        R"re(<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?(.*?)(?=<\|box_start\|>|$))re",
        pyre::DOTALL);
    const ustring out = pyre::from_utf8(text);
    block_list blocks;
    for (const auto & m : layout_re.finditer(out)) {
        long c[4];
        bool ok = true;
        for (int k = 0; k < 4; ++k) {
            c[k] = bounded_int(m.str(k + 1));
            if (c[k] < 0) ok = false;
        }
        if (!ok) continue;
        long x1 = c[0], y1 = c[1], x2 = c[2], y2 = c[3];
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);
        if (x1 == x2 || y1 == y2) continue;
        const std::array<double, 4> bbox = { { x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0 } };
        std::string type = pyre::to_utf8(pyre::lower(m.str(5)));
        if (type == "unknown") type = "image";
        if (type == "inline_formula") continue;
        if (!block_types().count(type)) continue;
        int angle = -1;
        if (m.has(6)) {
            const ustring r = m.str(6);
            if (pyre::contains(r, U"<|rotate_up|>"))
                angle = 0;
            else if (pyre::contains(r, U"<|rotate_right|>"))
                angle = 90;
            else if (pyre::contains(r, U"<|rotate_down|>"))
                angle = 180;
            else if (pyre::contains(r, U"<|rotate_left|>"))
                angle = 270;
        }
        const bool merge_prev = type == "text" && pyre::contains(m.str(7), U"txt_contd_tgt");
        blocks.push_back(mineru_pp::make_block(type, bbox, angle, merge_prev));
    }
    // _deduplicate_layout_blocks
    block_list kept;
    std::map<std::pair<std::string, int>, std::vector<std::pair<block_ptr, double>>> groups;
    for (auto & b : blocks) {
        auto & g = groups[{ b->type, b->angle }];
        const double area = (b->bbox[2] - b->bbox[0]) * (b->bbox[3] - b->bbox[1]);
        bool dup = false;
        for (auto & kv : g) {
            const double inter = inter_area(b->bbox, kv.first->bbox);
            const double uni = mineru::fsum({ area, kv.second, -inter });
            if (uni > 0 && inter / uni > 0.9) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            kept.push_back(b);
            g.push_back({ b, area });
        }
    }
    // _filter_table_internal_layout_blocks
    const auto internal = covered_indices(kept, { "text", "equation", "equation_block" }, { "table" });
    if (internal.empty()) return kept;
    block_list out_blocks;
    for (size_t i = 0; i < kept.size(); ++i)
        if (!internal.count(i)) out_blocks.push_back(kept[i]);
    return out_blocks;
}

std::vector<region> layout_regions(const block_list & blocks, int width, int height) {
    std::vector<region> out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const block & b = *blocks[i];
        region r;
        r.raw_type = b.type;
        r.angle = b.angle;
        r.merge_prev = b.has_key("merge_prev") && b.merge_prev;
        r.norm = b.bbox;
        r.index = static_cast<int>(i);
        r.base.x1 = static_cast<float>(b.bbox[0] * width);
        r.base.y1 = static_cast<float>(b.bbox[1] * height);
        r.base.x2 = static_cast<float>(b.bbox[2] * width);
        r.base.y2 = static_cast<float>(b.bbox[3] * height);
        r.base.score = 1.0f;
        r.base.label = label_for(b.type);
        r.base.label_name = label_string(r.base.label);
        out.push_back(r);
    }
    return out;
}

rgb_image prepare_for_layout(const rgb_image & page) {
    return resize_bicubic(page, 1036, 1036);
}

rgb_image resize_by_need(const rgb_image & img, int min_edge, double max_ratio) {
    const auto plan = mineru::normalize_crop_geometry(img.width, img.height, -1, min_edge, max_ratio);
    rgb_image cur = img;
    if (plan.padded_width != img.width || plan.padded_height != img.height) {
        rgb_image padded;
        padded.width = plan.padded_width;
        padded.height = plan.padded_height;
        padded.px.assign(static_cast<size_t>(padded.width) * padded.height * 3, 255);
        for (int y = 0; y < img.height; ++y)
            std::memcpy(padded.at(plan.paste_x, y + plan.paste_y), img.at(0, y), static_cast<size_t>(img.width) * 3);
        cur = padded;
    }
    if (plan.output_width != cur.width || plan.output_height != cur.height)
        cur = resize_bicubic(cur, plan.output_width, plan.output_height);
    return cur;
}

namespace {
const char * prompt_for(const std::string & type) {
    if (type == "table") return "\nTable Recognition:";
    if (type == "equation") return "\nFormula Recognition:";
    if (type == "image" || type == "chart") return "\nImage Analysis:";
    return "\nText Recognition:";
}
const char * destination_name(ocr_region_router::destination d) {
    switch (d) {
    case ocr_region_router::destination::table:
        return "table";
    case ocr_region_router::destination::formula:
        return "formula";
    case ocr_region_router::destination::fallback:
        return "image-analysis";
    default:
        return "text";
    }
}
void require_complete(const recognition & r, bool allow) {
    if ((r.capped || !r.eos) && !allow)
        throw std::runtime_error("MinerU generation did not reach EOS; partial content is not accepted");
}
} // namespace

page_result extract_page(const rgb_image & page, const predictor & predict, const options & opts,
                         mineru_img::python_random & rng) {
    if (page.width <= 0 || page.height <= 0) throw std::invalid_argument("MinerU page image must be nonempty");
    page_result res;
    res.layout = predict(prepare_for_layout(page), "\nLayout Detection:");
    require_complete(res.layout, opts.allow_truncated);
    res.capped = res.layout.capped;
    block_list blocks = parse_layout_output(res.layout.text);

    // prepare_for_extract
    {
        const auto internal = covered_indices(blocks, { "image_caption" }, { "image", "chart", "image_block" });
        if (!internal.empty()) {
            block_list kept;
            for (size_t i = 0; i < blocks.size(); ++i)
                if (!internal.count(i)) kept.push_back(blocks[i]);
            blocks.swap(kept);
        }
    }
    res.regions = layout_regions(blocks, page.width, page.height);
    {
        std::vector<layout_detect::region> base;
        for (const auto & r : res.regions) base.push_back(r.base);
        ocr_region_router::request_policy policy;
        policy.want_tables = policy.want_formulas = true;
        policy.image_text_fallback = opts.image_analysis;
        ocr_region_router::build(base, {}, policy, res.plan);
    }
    const auto non_standalone = covered_indices(blocks, { "image", "chart" }, { "image_block" });
    std::set<std::string> skip = { "list", "equation_block", "image_block" };
    if (!opts.image_analysis) skip.insert({ "image", "chart" });
    for (const auto & t : opts.not_extract)
        if (block_types().count(t)) skip.insert(t);
    std::vector<size_t> tables;
    for (size_t i = 0; i < blocks.size(); ++i)
        if (blocks[i]->type == "table" && !skip.count("table")) tables.push_back(i);
    const auto table_map = mineru_img::build_table_image_map(blocks, tables, 0.9);
    std::set<size_t> absorbed;
    for (const auto & kv : table_map) absorbed.insert(kv.second.begin(), kv.second.end());
    for (size_t i : absorbed) blocks[i]->absorbed = true;

    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        block & b = *blocks[idx];
        if (skip.count(b.type)) continue;
        if (b.type == "image" && b.absorbed) continue;
        if (b.type == "image" || b.type == "chart") {
            if (non_standalone.count(idx)) continue;
            const double w = b.bbox[2] - b.bbox[0], h = b.bbox[3] - b.bbox[1];
            if (!((w > 0.1 && h > 0.1) || w * h > 0.01)) continue;
        }
        const int x0 = round_half_even(b.bbox[0] * page.width), y0 = round_half_even(b.bbox[1] * page.height);
        const int x1 = round_half_even(b.bbox[2] * page.width), y1 = round_half_even(b.bbox[3] * page.height);
        rgb_image crop = mineru_img::crop(page, x0, y0, x1, y1);
        if (crop.width < 1 || crop.height < 1) continue;
        bool table_prepared = false, masked = false;
        if (b.type == "table") {
            std::vector<const block *> images;
            for (const auto & kv : table_map)
                if (kv.first == idx)
                    for (size_t im : kv.second) images.push_back(blocks[im].get());
            std::vector<std::pair<ustring, ustring>> tokens;
            crop = mineru_img::mask_and_encode_table_image(page, b, images, crop, rng, tokens);
            table_prepared = true;
            if (!tokens.empty()) {
                b.table_image_tokens = tokens;
                b.has_token_map = true;
                masked = true;
            }
        }
        if (!table_prepared && (b.angle == 90 || b.angle == 180 || b.angle == 270))
            crop = mineru_img::rotate(crop, b.angle);
        crop = resize_by_need(crop);
        crop_record rec;
        rec.block_index = idx;
        rec.type = b.type;
        rec.prompt = prompt_for(b.type);
        rec.width = crop.width;
        rec.height = crop.height;
        rec.fnv1a = fnv1a(crop.px);
        rec.masked_table = masked;
        rec.result = predict(crop, rec.prompt);
        require_complete(rec.result, opts.allow_truncated);
        res.capped = res.capped || rec.result.capped;
        b.set_content(pyre::from_utf8(rec.result.text));
        res.crops.push_back(rec);
    }
    res.blocks = mineru_pp::post_process_helper(blocks, opts.post);
    res.json = mineru_pp::blocks_json(res.blocks);
    res.markdown = mineru_pp::json2md(res.blocks);
    return res;
}

// --------------------------------------------------- document model --------
namespace {
std::string marker_type(const std::string & t) {
    static const std::map<std::string, std::string> m = {
        { "text", "Text" },
        { "title", "SectionHeader" },
        { "doc_title", "SectionHeader" },
        { "paragraph_title", "SectionHeader" },
        { "table", "Table" },
        { "equation", "Equation" },
        { "formula_number", "Text" },
        { "code", "Code" },
        { "algorithm", "Code" },
        { "aside_text", "Text" },
        { "ref_text", "Bibliography" },
        { "index", "TableOfContents" },
        { "phonetic", "Text" },
        { "list_item", "ListItem" },
        { "table_caption", "Caption" },
        { "image_caption", "Caption" },
        { "code_caption", "Caption" },
        { "caption", "Caption" },
        { "table_footnote", "Footnote" },
        { "image_footnote", "Footnote" },
        { "footnote", "Footnote" },
        { "header", "PageHeader" },
        { "footer", "PageFooter" },
        { "page_number", "PageFooter" },
        { "page_footnote", "Footnote" },
        { "image", "Picture" },
        { "chart", "Figure" },
        { "list", "ListGroup" },
        { "image_block", "Figure" },
        { "equation_block", "Equation" },
        { "unknown", "Text" },
    };
    auto it = m.find(t);
    return it == m.end() ? "Text" : it->second;
}
std::string bbox_repr(const std::array<double, 4> & b) {
    std::string s;
    for (int k = 0; k < 4; ++k) s += (k ? "," : "") + mineru_pp::python_float_repr(b[static_cast<size_t>(k)]);
    return s;
}
} // namespace

void append_page(crisp_marker::document & doc, const page_result & r, int page_index, int width, int height) {
    crisp_marker::page p;
    p.index = page_index;
    p.width = width;
    p.height = height;
    p.extraction_method = "mineru25-pro-vlm";
    p.raw_layout = r.layout.text;
    p.layout_token_cap_reached = r.layout.capped;
    for (size_t i = 0; i < r.blocks.size(); ++i) {
        const block & b = *r.blocks[i];
        crisp_marker::block mb;
        mb.id = "/page/" + std::to_string(page_index) + "/" + marker_type(b.type) + "/" + std::to_string(i);
        mb.type = marker_type(b.type);
        mb.order = static_cast<int>(i);
        mb.bbox = { b.bbox[0] * width, b.bbox[1] * height, b.bbox[2] * width, b.bbox[3] * height };
        mb.angle = b.angle < 0 ? 0 : b.angle;
        mb.continuation = b.has_key("merge_prev") && b.merge_prev;
        const std::string content = b.has_content ? pyre::to_utf8(b.content) : std::string();
        if (b.type == "table")
            mb.html = content;
        else
            mb.text = content;
        mb.raw_label = b.type;
        mb.source = "mineru25-pro";
        auto & a = mb.attributes;
        a["mineru.type"] = b.type;
        a["mineru.bbox"] = bbox_repr(b.bbox);
        a["mineru.angle"] = b.angle < 0 ? "null" : std::to_string(b.angle);
        a["mineru.content_null"] = b.has_content ? "0" : "1";
        a["mineru.content"] = content;
        std::string keys;
        for (const auto & k : b.extra_keys) keys += (keys.empty() ? "" : ",") + k;
        a["mineru.keys"] = keys;
        if (b.has_key("merge_prev")) a["mineru.merge_prev"] = b.merge_prev ? "true" : "false";
        if (b.has_key("sub_type")) a["mineru.sub_type"] = pyre::to_utf8(b.sub_type);
        p.blocks.push_back(std::move(mb));
    }
    for (size_t k = 0; k < r.crops.size(); ++k) {
        const auto & c = r.crops[k];
        if (c.result.capped)
            p.warnings.push_back("Region " + std::to_string(c.block_index) + " reached its token cap.");
    }
    doc.pages.push_back(std::move(p));
    doc.backend = "mineru25-pro";
    doc.metadata["mineru_profile"] = profile_name();
    doc.metadata["mineru_blueprint"] = "opendatalab/mineru-vl-utils@c9797cde8f54cc48306196fc6aebe73ea31dd4a3";
    doc.metadata["mineru_token_text_rendering"] = "embedded DejaVu Sans atlas (approximates Pillow/FreeType pixels)";
}

namespace {
const std::string & attr(const crisp_marker::block & b, const char * key) {
    auto it = b.attributes.find(key);
    if (it == b.attributes.end()) throw std::invalid_argument(std::string("block lacks MinerU attribute ") + key);
    return it->second;
}
} // namespace

std::string render_upstream_json(const crisp_marker::page & page) {
    std::string out = "[";
    for (size_t i = 0; i < page.blocks.size(); ++i) {
        const auto & b = page.blocks[i];
        if (i) out += ",";
        out +=
            "{\"type\":" + mineru_pp::json_quote(pyre::from_utf8(attr(b, "mineru.type"))) + ",\"bbox\":[" +
            attr(b, "mineru.bbox") + "],\"angle\":" + attr(b, "mineru.angle") + ",\"content\":" +
            (attr(b, "mineru.content_null") == "1" ? std::string("null")
                                                   : mineru_pp::json_quote(pyre::from_utf8(attr(b, "mineru.content"))));
        std::stringstream keys(attr(b, "mineru.keys"));
        std::string k;
        while (std::getline(keys, k, ',')) {
            if (k == "merge_prev")
                out += ",\"merge_prev\":" + attr(b, "mineru.merge_prev");
            else if (k == "sub_type")
                out += ",\"sub_type\":" + mineru_pp::json_quote(pyre::from_utf8(attr(b, "mineru.sub_type")));
        }
        out += "}";
    }
    return out + "]";
}

std::string render_upstream_markdown(const crisp_marker::page & page) {
    block_list blocks;
    for (const auto & b : page.blocks) {
        auto nb = std::make_shared<block>();
        nb->type = attr(b, "mineru.type");
        if (attr(b, "mineru.content_null") != "1") nb->set_content(pyre::from_utf8(attr(b, "mineru.content")));
        auto it = b.attributes.find("mineru.merge_prev");
        if (it != b.attributes.end()) {
            nb->extra_keys.push_back("merge_prev");
            nb->merge_prev = it->second == "true";
        }
        blocks.push_back(nb);
    }
    return mineru_pp::json2md(blocks);
}

// ------------------------------------------------------------ mini JSON ----
namespace {
struct jv {
    enum kind_t { null_, bool_, num_, str_, arr_, obj_ } kind = null_;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<jv> a;
    std::vector<std::pair<std::string, jv>> o;
    const jv * get(const std::string & k) const {
        for (auto & kv : o)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
struct json_parser {
    const std::string & t;
    size_t i = 0;
    void ws() {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
    }
    [[noreturn]] void fail() { throw std::invalid_argument("invalid JSON at offset " + std::to_string(i)); }
    std::string str() {
        if (t[i] != '"') fail();
        ++i;
        std::string out;
        while (i < t.size() && t[i] != '"') {
            char c = t[i++];
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= t.size()) fail();
            const char e = t[i++];
            switch (e) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                auto hex4 = [&]() {
                    if (i + 4 > t.size()) fail();
                    unsigned v = static_cast<unsigned>(std::stoul(t.substr(i, 4), nullptr, 16));
                    i += 4;
                    return v;
                };
                unsigned cp = hex4();
                if (cp >= 0xd800 && cp < 0xdc00 && i + 1 < t.size() && t[i] == '\\' && t[i + 1] == 'u') {
                    i += 2;
                    const unsigned lo = hex4();
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                }
                pyre::append_utf8(out, cp);
                break;
            }
            default:
                fail();
            }
        }
        if (i >= t.size()) fail();
        ++i;
        return out;
    }
    jv value() {
        ws();
        if (i >= t.size()) fail();
        jv v;
        const char c = t[i];
        if (c == '{') {
            v.kind = jv::obj_;
            ++i;
            ws();
            if (t[i] == '}') {
                ++i;
                return v;
            }
            while (true) {
                ws();
                std::string k = str();
                ws();
                if (t[i++] != ':') fail();
                v.o.push_back({ k, value() });
                ws();
                if (t[i] == ',') {
                    ++i;
                    continue;
                }
                if (t[i++] != '}') fail();
                return v;
            }
        }
        if (c == '[') {
            v.kind = jv::arr_;
            ++i;
            ws();
            if (t[i] == ']') {
                ++i;
                return v;
            }
            while (true) {
                v.a.push_back(value());
                ws();
                if (t[i] == ',') {
                    ++i;
                    continue;
                }
                if (t[i++] != ']') fail();
                return v;
            }
        }
        if (c == '"') {
            v.kind = jv::str_;
            v.s = str();
            return v;
        }
        if (t.compare(i, 4, "true") == 0) {
            i += 4;
            v.kind = jv::bool_;
            v.b = true;
            return v;
        }
        if (t.compare(i, 5, "false") == 0) {
            i += 5;
            v.kind = jv::bool_;
            return v;
        }
        if (t.compare(i, 4, "null") == 0) {
            i += 4;
            return v;
        }
        size_t end = i;
        while (end < t.size() && (std::isdigit(static_cast<unsigned char>(t[end])) || t[end] == '-' || t[end] == '+' ||
                                  t[end] == '.' || t[end] == 'e' || t[end] == 'E'))
            ++end;
        if (end == i) fail();
        v.kind = jv::num_;
        v.n = std::stod(t.substr(i, end - i));
        i = end;
        return v;
    }
};
jv parse_json(const std::string & text) {
    json_parser p{ text };
    jv v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail();
    return v;
}
bool jbool(const jv * v, bool def) {
    if (!v) return def;
    if (v->kind != jv::bool_) throw std::invalid_argument("expected JSON boolean");
    return v->b;
}
options options_from(const jv & o) {
    if (o.kind != jv::obj_) throw std::invalid_argument("MinerU options must be a JSON object");
    static const std::set<std::string> known = { "image_analysis",        "simple_post_process",
                                                 "handle_equation_block", "abandon_list",
                                                 "abandon_paratext",      "enable_table_formula_eq_wrap",
                                                 "not_extract_list",      "allow_truncated_content",
                                                 "table_token_seed" };
    for (auto & kv : o.o)
        if (!known.count(kv.first)) throw std::invalid_argument("unknown MinerU option " + kv.first);
    options r;
    r.image_analysis = jbool(o.get("image_analysis"), false);
    r.post.simple_post_process = jbool(o.get("simple_post_process"), false);
    r.post.handle_equation_block = jbool(o.get("handle_equation_block"), true);
    r.post.abandon_list = jbool(o.get("abandon_list"), false);
    r.post.abandon_paratext = jbool(o.get("abandon_paratext"), false);
    r.post.enable_table_formula_eq_wrap = jbool(o.get("enable_table_formula_eq_wrap"), false);
    r.allow_truncated = jbool(o.get("allow_truncated_content"), false);
    if (const jv * n = o.get("not_extract_list")) {
        if (n->kind != jv::arr_ && n->kind != jv::null_) throw std::invalid_argument("not_extract_list must be a list");
        for (auto & x : n->a) {
            if (x.kind != jv::str_) throw std::invalid_argument("not_extract_list entries must be strings");
            r.not_extract.push_back(x.s);
        }
    }
    if (const jv * s = o.get("table_token_seed")) {
        if (s->kind == jv::num_) {
            if (s->n < 0 || s->n != std::floor(s->n) || s->n > 9007199254740992.0)
                throw std::invalid_argument("table_token_seed must be a non-negative integer");
            r.has_seed = true;
            r.seed = static_cast<uint64_t>(s->n);
        } else if (s->kind != jv::null_)
            throw std::invalid_argument("table_token_seed must be an integer or null");
    }
    return r;
}
std::string hex64(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return b;
}
} // namespace

options parse_options_json(const std::string & json) {
    if (json.empty()) return options{};
    return options_from(parse_json(json));
}

std::string layout_blocks_json(const std::string & layout_text) {
    return mineru_pp::blocks_json(parse_layout_output(layout_text));
}

extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);

std::string replay_case_json(const std::string & case_json) {
    const jv c = parse_json(case_json);
    const jv * path = c.get("image_path");
    if (!path || path->kind != jv::str_) throw std::invalid_argument("case lacks image_path");
    int w = 0, h = 0, n = 0;
    unsigned char * px = stbi_load(path->s.c_str(), &w, &h, &n, 3);
    if (!px) throw std::runtime_error("cannot decode " + path->s);
    rgb_image page;
    page.width = w;
    page.height = h;
    page.px.assign(px, px + static_cast<size_t>(w) * h * 3);
    stbi_image_free(px);
    options opts;
    if (const jv * o = c.get("options")) opts = options_from(*o);
    const jv * layout = c.get("layout");
    const jv * recs = c.get("recognitions");
    if (!layout || !recs || recs->kind != jv::arr_) throw std::invalid_argument("case lacks layout/recognitions");
    size_t next = 0;
    std::string mismatch;
    auto to_rec = [](const jv & v) {
        recognition r;
        const jv * t = v.get("text");
        if (!t || t->kind != jv::str_) throw std::invalid_argument("recognition lacks text");
        r.text = t->s;
        r.eos = jbool(v.get("eos"), true);
        r.capped = jbool(v.get("capped"), false);
        return r;
    };
    bool first = true;
    predictor predict = [&](const rgb_image & img, const std::string & prompt) {
        if (first) {
            first = false;
            if (prompt != "\nLayout Detection:" || img.width != 1036 || img.height != 1036)
                mismatch += "layout request differs;";
            return to_rec(*layout);
        }
        if (next >= recs->a.size()) throw std::runtime_error("native route requested more regions than recorded");
        const jv & r = recs->a[next++];
        const jv * p = r.get("prompt");
        const jv * rw = r.get("width");
        const jv * rh = r.get("height");
        if (p && p->kind == jv::str_ && p->s != prompt)
            mismatch += "prompt differs at " + std::to_string(next - 1) + ";";
        if (rw && rh && (static_cast<int>(rw->n) != img.width || static_cast<int>(rh->n) != img.height))
            mismatch += "size differs at " + std::to_string(next - 1) + ";";
        return to_rec(r);
    };
    mineru_img::python_random rng = opts.has_seed ? mineru_img::python_random(opts.seed) : mineru_img::python_random();
    const page_result res = extract_page(page, predict, opts, rng);
    if (next != recs->a.size()) mismatch += "native route requested fewer regions than recorded;";
    crisp_marker::document doc;
    append_page(doc, res, 0, w, h);
    const std::string doc_json = render_upstream_json(doc.pages[0]);
    const std::string doc_md = render_upstream_markdown(doc.pages[0]);
    std::string out = "{\"profile\":" + mineru_pp::json_quote(pyre::from_utf8(profile_name())) +
                      ",\"blocks_json\":" + mineru_pp::json_quote(pyre::from_utf8(res.json)) +
                      ",\"markdown\":" + mineru_pp::json_quote(pyre::from_utf8(res.markdown)) +
                      ",\"document_blocks_json\":" + mineru_pp::json_quote(pyre::from_utf8(doc_json)) +
                      ",\"document_markdown\":" + mineru_pp::json_quote(pyre::from_utf8(doc_md)) +
                      ",\"generic_markdown\":" +
                      mineru_pp::json_quote(pyre::from_utf8(crisp_marker::render_document(doc, "markdown"))) +
                      ",\"request_mismatch\":" + mineru_pp::json_quote(pyre::from_utf8(mismatch)) + ",\"crops\":[";
    for (size_t i = 0; i < res.crops.size(); ++i) {
        const auto & cr = res.crops[i];
        out += (i ? "," : "") + std::string("{\"block_index\":") + std::to_string(cr.block_index) +
               ",\"prompt\":" + mineru_pp::json_quote(pyre::from_utf8(cr.prompt)) +
               ",\"width\":" + std::to_string(cr.width) + ",\"height\":" + std::to_string(cr.height) + ",\"fnv1a\":\"" +
               hex64(cr.fnv1a) + "\",\"masked_table\":" + (cr.masked_table ? "true" : "false") + "}";
    }
    out += "],\"regions\":[";
    for (size_t i = 0; i < res.regions.size(); ++i) {
        const auto & r = res.regions[i];
        out += (i ? "," : "") + std::string("{\"raw_type\":") + mineru_pp::json_quote(pyre::from_utf8(r.raw_type)) +
               ",\"label\":" + mineru_pp::json_quote(pyre::from_utf8(r.base.label_name)) + ",\"destination\":\"" +
               destination_name(res.plan.layout_decisions[i].dest) + "\"}";
    }
    return out + "]}";
}

bool self_test(std::string & error) {
    // Layout parse: unknown->image, inline_formula dropped, rotation, merge flag.
    const std::string layout = "<|box_start|>100 100 900 200<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>"
                               "<|box_start|>100 300 900 400<|box_end|><|ref_start|>Unknown<|ref_end|><|rotate_right|>"
                               "<|box_start|>1 1 1 5<|box_end|><|ref_start|>text<|ref_end|>"
                               "<|box_start|>100 500 900 600<|box_end|><|ref_start|>text<|ref_end|>txt_contd_tgt";
    const std::string got = layout_blocks_json(layout);
    const std::string want =
        "[{\"type\":\"text\",\"bbox\":[0.1,0.1,0.9,0.2],\"angle\":0,\"content\":null,\"merge_prev\":false},"
        "{\"type\":\"image\",\"bbox\":[0.1,0.3,0.9,0.4],\"angle\":90,\"content\":null},"
        "{\"type\":\"text\",\"bbox\":[0.1,0.5,0.9,0.6],\"angle\":null,\"content\":null,\"merge_prev\":true}]";
    if (got != want) {
        error = "layout parse: " + got;
        return false;
    }
    // Python random compatibility (random.seed(0); random.random()).
    mineru_img::python_random r(0);
    if (std::fabs(r.random() - 0.8444218515250481) > 0) {
        error = "python_random seed(0) mismatch";
        return false;
    }
    return true;
}

} // namespace mineru_doc
