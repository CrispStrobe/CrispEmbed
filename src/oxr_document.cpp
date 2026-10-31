// oxr_document.cpp — native OXR-1.0 document route. See oxr_document.h.
//
// Translated from SoMarkAI/OXR @ c1a1dcb312ee9f691404f127785510453c0d9350:
// pipeline/orchestrator.py, layout.py, postprocess/layout_postprocess.py,
// recognition.py, reading_order.py, xycut_order.py, model/prompts.py,
// postprocess/{assembly,json,cross_category_dedupe,overlap_dedupe,
// exact_text_dedupe,full_overlap_dedupe,picture}_postprocess.py,
// utils/image.py. Apache License 2.0 with the additional OXR terms
// (licenses/oxr-LICENSE, licenses/oxr-NOTICE). Modified for CrispEmbed.
#include "oxr_document.h"

#include "core/json.h"
#include "cv_compat.h"
#include "layout_detect.h"
#include "oxr_document_internal.h"
#include "xycut_plus_plus.h"

#include "../ggml/examples/stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace oxr_document {

using core_json_value::value;
using py_str::u32;

static const char * k_blueprint_revision = "c1a1dcb312ee9f691404f127785510453c0d9350";
static const char * k_prompt_text = "Recognize the content of the image.";
static const char * k_prompt_formula = "Recognize the formula of the image into KaTex format.";
static const char * k_prompt_table = "Recognize the table of the image into html format.";
static const char * k_prompt_code = "Recognize the code of the image into markdown code format.";
static const char * k_prompt_chemical = "Recognize the chemical structure of the image into SMILES.";
static const char * k_formula_numbering_role = "formula_numbering";
static const char * k_table_image_role = "table_embedded_image";

const char * type_name(block_type type) {
    static const char * names[] = {
        "Caption", "Footnote", "Formula", "Text",       "Page-footer", "Page-header",
        "Picture", "Title",    "Table",   "Code Block", "Stamp",       "Chemical Structure"
    };
    const int i = int(type);
    return i >= 0 && i < k_num_layout_classes ? names[i] : "Text";
}
bool type_from_name(const std::string & name, block_type & out) {
    for (int i = 0; i < k_num_layout_classes; ++i)
        if (name == type_name(block_type(i))) {
            out = block_type(i);
            return true;
        }
    return false;
}
const char * format_name(content_format f) {
    switch (f) {
    case content_format::html:
        return "html";
    case content_format::latex:
        return "latex";
    case content_format::markdown:
        return "markdown";
    default:
        return "text";
    }
}

namespace {
std::string quote(const std::string & s) {
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
std::string utf8(const u32 & s) {
    return py_str::to_utf8(s);
}
u32 cps(const std::string & s) {
    return py_str::from_utf8(s);
}
uint64_t next_uid() {
    static thread_local uint64_t counter = 0;
    return ++counter;
}
double area(const std::array<double, 4> & b) {
    return std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
}
double intersection(const std::array<double, 4> & a, const std::array<double, 4> & b) {
    return area({ std::max(a[0], b[0]), std::max(a[1], b[1]), std::min(a[2], b[2]), std::min(a[3], b[3]) });
}
bool inside(const block & child, const block & parent) {
    return intersection(child.bbox, parent.bbox) / std::max(area(child.bbox), 1e-9) >= 0.95;
}
bool is_numbering(const block & b) {
    return b.generated_role == k_formula_numbering_role;
}
} // namespace

// ---------------------------------------------------------------- layout input
page_layout from_layout_detect(const std::vector<layout_detect::region> & regions) {
    page_layout out;
    for (const auto & r : regions) {
        block_type t;
        if (!r.label_name || !type_from_name(r.label_name, t))
            throw std::invalid_argument(std::string("layout region label is not an OXR class: ") +
                                        (r.label_name ? r.label_name : "(null)"));
        out.push_back({ int(t), double(r.x1), double(r.y1), double(r.x2), double(r.y2), double(r.score) });
    }
    return out;
}

value layout_response(const std::vector<page_layout> & pages) {
    auto arr = [] {
        value v;
        v.kind = value::array_t;
        return v;
    };
    value labels = arr(), boxes = arr(), scores = arr();
    for (const auto & page : pages) {
        value pl = arr(), pb = arr(), ps = arr();
        for (const auto & r : page) {
            value l;
            l.kind = value::int_t;
            l.i = r.label;
            pl.a.push_back(l);
            value b = arr();
            for (double c : { r.x1, r.y1, r.x2, r.y2 }) {
                value n;
                n.kind = value::float_t;
                n.f = c;
                b.a.push_back(n);
            }
            pb.a.push_back(b);
            value s;
            s.kind = value::float_t;
            s.f = r.score;
            ps.a.push_back(s);
        }
        labels.a.push_back(pl);
        boxes.a.push_back(pb);
        scores.a.push_back(ps);
    }
    value data;
    data.kind = value::object_t;
    data.o = { { "labels", labels }, { "boxes", boxes }, { "scores", scores } };
    value status;
    status.kind = value::string_t;
    status.s = "success";
    value params;
    params.kind = value::object_t;
    value root;
    root.kind = value::object_t;
    root.o = { { "status", status }, { "params", params }, { "data", data } };
    return root;
}

namespace helpers {

namespace {
[[noreturn]] void layout_error(const std::string & message) {
    throw std::invalid_argument("Invalid layout response: " + message);
}
const value & require_list(const value * v, const std::string & field) {
    if (!v || v->kind != value::array_t) layout_error(field + " must be a list");
    return *v;
}
double require_number(const value & v, const std::string & field) {
    if (!v.is_number()) layout_error(field + " must be numeric");
    const double n = v.number();
    if (!std::isfinite(n)) layout_error(field + " must be finite");
    return n;
}
double iou(const std::array<double, 4> & l, const std::array<double, 4> & r) {
    const double w = std::max(0.0, std::min(l[2], r[2]) - std::max(l[0], r[0]));
    const double h = std::max(0.0, std::min(l[3], r[3]) - std::max(l[1], r[1]));
    const double inter = w * h;
    if (inter == 0.0) return 0.0;
    const double la = (l[2] - l[0]) * (l[3] - l[1]);
    const double ra = (r[2] - r[0]) * (r[3] - r[1]);
    return inter / (la + ra - inter);
}
} // namespace

std::vector<std::vector<block>> parse_layout_batch(const value & payload,
                                                   const std::vector<std::pair<int, int>> & sizes,
                                                   double score_threshold, std::optional<double> nms) {
    if (payload.kind != value::object_t) layout_error("response must be an object");
    const value * status = payload.get("status");
    if (!status || status->kind != value::string_t || status->s != "success")
        layout_error("status must equal 'success'");
    const value * params = payload.get("params");
    if (!params || params->kind != value::object_t) layout_error("params must be an object");
    const value * data = payload.get("data");
    if (!data || data->kind != value::object_t) layout_error("data must be an object");
    const value & labels = require_list(data->get("labels"), "data.labels");
    const value & boxes = require_list(data->get("boxes"), "data.boxes");
    const value & scores = require_list(data->get("scores"), "data.scores");
    if (!std::isfinite(score_threshold)) layout_error("score threshold must be finite");
    if (!(0.0 <= score_threshold && score_threshold <= 1.0)) layout_error("score threshold must be between 0 and 1");
    if (nms) {
        if (!std::isfinite(*nms)) layout_error("NMS IoU threshold must be finite");
        if (!(0.0 <= *nms && *nms <= 1.0)) layout_error("NMS IoU threshold must be between 0 and 1");
    }
    if (!(labels.a.size() == boxes.a.size() && boxes.a.size() == scores.a.size() && scores.a.size() == sizes.size()))
        layout_error("batch count must match image count");
    std::vector<std::vector<block>> pages;
    for (size_t pi = 0; pi < sizes.size(); ++pi) {
        const int width = sizes[pi].first, height = sizes[pi].second;
        const std::string ps = "page " + std::to_string(pi);
        if (width <= 0 || height <= 0) layout_error(ps + " image dimensions must be positive integers");
        const value & pl = require_list(&labels.a[pi], ps + " labels");
        const value & pb = require_list(&boxes.a[pi], ps + " boxes");
        const value & psc = require_list(&scores.a[pi], ps + " scores");
        if (!(pl.a.size() == pb.a.size() && pb.a.size() == psc.a.size()))
            layout_error(ps + " candidate count mismatch");
        struct cand {
            int position;
            block_type type;
            std::array<double, 4> bbox;
            double score;
        };
        std::vector<cand> candidates;
        for (size_t ci = 0; ci < pl.a.size(); ++ci) {
            const std::string field = ps + " candidate " + std::to_string(ci);
            const value & label = pl.a[ci];
            if (label.kind != value::int_t) layout_error(field + " label must be an integer");
            if (!(0 <= label.i && label.i < k_num_layout_classes))
                layout_error(field + " label must be between 0 and 11");
            const double score = require_number(psc.a[ci], field + " score");
            if (!(0.0 <= score && score <= 1.0)) layout_error(field + " score must be between 0 and 1");
            const value & box = require_list(&pb.a[ci], field + " bbox");
            if (box.a.size() != 4) layout_error(field + " bbox must contain four coordinates");
            double c[4];
            for (int k = 0; k < 4; ++k)
                c[k] = require_number(box.a[size_t(k)], field + " bbox[" + std::to_string(k) + "]");
            if (score < score_threshold) continue;
            if (c[0] >= c[2] || c[1] >= c[3]) layout_error(field + " bbox must be nondegenerate");
            const double x1 = std::min(std::max(c[0], 0.0), double(width));
            const double y1 = std::min(std::max(c[1], 0.0), double(height));
            const double x2 = std::min(std::max(c[2], 0.0), double(width));
            const double y2 = std::min(std::max(c[3], 0.0), double(height));
            if (x1 >= x2 || y1 >= y2) continue;
            candidates.push_back({ int(ci), block_type(label.i), { x1, y1, x2, y2 }, score });
        }
        std::vector<cand> retained = candidates;
        if (nms) {
            std::vector<size_t> order(candidates.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                if (candidates[a].score != candidates[b].score) return candidates[a].score > candidates[b].score;
                return candidates[a].position < candidates[b].position;
            });
            std::vector<size_t> kept;
            for (size_t i : order) {
                bool drop = false;
                for (size_t k : kept)
                    if (candidates[k].type == candidates[i].type &&
                        iou(candidates[i].bbox, candidates[k].bbox) > *nms) {
                        drop = true;
                        break;
                    }
                if (!drop) kept.push_back(i);
            }
            std::set<int> kept_positions;
            for (size_t k : kept) kept_positions.insert(candidates[k].position);
            retained.clear();
            for (const auto & c : candidates)
                if (kept_positions.count(c.position)) retained.push_back(c);
        }
        std::vector<block> page;
        for (const auto & c : retained) {
            block b;
            b.idx = int(page.size());
            b.type = c.type;
            b.bbox = c.bbox;
            b.format = content_format::text;
            b.layout_score = c.score;
            b.uid = next_uid();
            page.push_back(b);
        }
        pages.push_back(page);
    }
    return pages;
}

std::vector<uint8_t> crop_rgb(const rgb_page & image, const std::array<double, 4> & bbox, int & width, int & height) {
    long long x1 = py_str::round_half_even(bbox[0]), y1 = py_str::round_half_even(bbox[1]);
    long long x2 = py_str::round_half_even(bbox[2]), y2 = py_str::round_half_even(bbox[3]);
    x1 = std::max(0LL, x1);
    y1 = std::max(0LL, y1);
    x2 = std::min((long long)image.width, x2);
    y2 = std::min((long long)image.height, y2);
    // numpy slicing image[y1:y2, x1:x2] (bounds are nonnegative here).
    auto norm = [](long long v, long long n) { return std::min(std::max(v < 0 ? v + n : v, 0LL), n); };
    const long long ys = norm(y1, image.height), ye = norm(y2, image.height);
    const long long xs = norm(x1, image.width), xe = norm(x2, image.width);
    width = int(std::max(0LL, xe - xs));
    height = int(std::max(0LL, ye - ys));
    std::vector<uint8_t> out(size_t(width) * size_t(height) * 3);
    for (int y = 0; y < height; ++y)
        std::copy_n(image.rgb + (size_t(ys + y) * image.width + size_t(xs)) * 3, size_t(width) * 3,
                    out.data() + size_t(y) * width * 3);
    return out;
}

// ------------------------------------------------------------- reading order
std::string reading_order_prompt(const std::vector<block> & blocks, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("image dimensions must be positive");
    const double sx = 1000.0 / width, sy = 1000.0 / height;
    std::string lines;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto & b = blocks[i].bbox;
        for (double v : b)
            if (!std::isfinite(v)) throw std::invalid_argument("objects.bbox must contain only finite numbers");
        const long long c0 = py_str::round_half_even(b[0] * sx), c1 = py_str::round_half_even(b[1] * sy);
        const long long c2 = py_str::round_half_even(b[2] * sx), c3 = py_str::round_half_even(b[3] * sy);
        if (i) lines += "\n";
        lines += std::to_string(i) + " | <|object_ref_start|>" + type_name(blocks[i].type) + "<|object_ref_end|> | " +
                 "<|box_start|>(" + std::to_string(c0) + "," + std::to_string(c1) + "),(" + std::to_string(c2) + "," +
                 std::to_string(c3) + ")<|box_end|>";
    }
    return "\nGiven the detected layout boxes below, output the reading order as a whitespace-separated permutation "
           "of 0-based box IDs only.\n\nBoxes:\n" +
           lines + "\n\nOutput only the box IDs.";
}

namespace {
// _READING_ORDER_PATTERN.fullmatch + [int(v) for v in raw.split()]. Values
// beyond 2^62 are clamped (they are out of range for any page anyway).
bool order_tokens(const std::string & raw, std::vector<long long> & out) {
    u32 text;
    try {
        text = cps(raw);
    } catch (const std::exception &) {
        return false;
    }
    out.clear();
    size_t i = 0;
    while (i < text.size()) {
        if (py_str::isspace(text[i])) {
            ++i;
            continue;
        }
        long long v = 0;
        size_t j = i;
        while (j < text.size() && !py_str::isspace(text[j])) {
            if (!py_str::is_ascii_digit(text[j])) return false;
            v = v > (1LL << 58) ? (1LL << 62) : v * 10 + (text[j] - U'0');
            ++j;
        }
        out.push_back(v);
        i = j;
    }
    return true;
}
struct partial_order {
    std::vector<int> order;
    std::set<int> missing;
    bool duplicates = false;
};
std::optional<partial_order> partial_reading_order(const std::string & raw, int count) {
    std::vector<long long> tokens;
    if (!order_tokens(raw, tokens)) return std::nullopt;
    for (long long v : tokens)
        if (v < 0 || v >= count) return std::nullopt;
    partial_order p;
    std::set<int> seen;
    for (long long v : tokens)
        if (seen.insert(int(v)).second) p.order.push_back(int(v));
    for (int i = 0; i < count; ++i)
        if (!seen.count(i)) p.missing.insert(i);
    p.duplicates = p.order.size() != tokens.size();
    return p;
}
std::vector<int> insert_missing(const std::vector<int> & order, const std::set<int> & missing,
                                const std::vector<int> & reference) {
    std::map<int, int> rank;
    for (size_t r = 0; r < reference.size(); ++r) rank[reference[r]] = int(r);
    std::vector<int> repaired = order;
    std::vector<int> ids(missing.begin(), missing.end());
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return rank.at(a) < rank.at(b); });
    for (int id : ids) {
        const int mr = rank.at(id);
        size_t best = 0;
        long long best_cost = std::numeric_limits<long long>::max();
        for (size_t index = 0; index <= repaired.size(); ++index) {
            long long cost = 0;
            for (size_t k = 0; k < index; ++k) cost += rank.at(repaired[k]) > mr;
            for (size_t k = index; k < repaired.size(); ++k) cost += rank.at(repaired[k]) < mr;
            if (cost < best_cost) {
                best_cost = cost;
                best = index;
            }
        }
        repaired.insert(repaired.begin() + long(best), id);
    }
    return repaired;
}
} // namespace

bool parse_reading_order(const std::string & raw, int count, std::vector<int> & order) {
    std::vector<long long> tokens;
    if (!order_tokens(raw, tokens)) return false;
    if ((long long)tokens.size() != count) return false;
    std::vector<long long> sorted = tokens;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < count; ++i)
        if (sorted[size_t(i)] != i) return false;
    order.assign(tokens.begin(), tokens.end());
    return true;
}

std::vector<int> xycut_plus_order(const std::vector<block> & blocks) {
    std::vector<int> identity(blocks.size());
    for (size_t i = 0; i < identity.size(); ++i) identity[i] = int(i);
    if (blocks.size() <= 1) return identity;
    static const char * label_map[] = { "figure_title", "footnote",        "formula", "text",      "text", "text",
                                        "image",        "paragraph_title", "table",   "algorithm", "seal", "image" };
    std::vector<xycut_plus_plus::entry> entries;
    for (const auto & b : blocks) {
        for (double v : b.bbox)
            if (!std::isfinite(v))
                throw std::invalid_argument("invalid bbox for XY-Cut++ block " + std::to_string(b.idx));
        xycut_plus_plus::entry e;
        e.bbox = { (long long)std::floor(b.bbox[0]), (long long)std::floor(b.bbox[1]), (long long)std::ceil(b.bbox[2]),
                   (long long)std::ceil(b.bbox[3]) };
        if (e.bbox[2] <= e.bbox[0] || e.bbox[3] <= e.bbox[1])
            throw std::invalid_argument("degenerate bbox for XY-Cut++ block " + std::to_string(b.idx));
        e.label = label_map[int(b.type)];
        entries.push_back(e);
    }
    const std::vector<std::string> no_mask = { "text", "formula", "algorithm", "reference", "content", "abstract" };
    const auto order = xycut_plus_plus::get_layout_ordering(entries, no_mask, false);
    std::vector<int> sorted = order;
    std::sort(sorted.begin(), sorted.end());
    if (sorted != identity) throw std::invalid_argument("XY-Cut++ did not return a complete unique permutation");
    return order;
}

std::optional<std::vector<int>> repair_incomplete_reading_order(const std::string & raw,
                                                                const std::vector<block> & blocks) {
    const auto p = partial_reading_order(raw, int(blocks.size()));
    if (!p) return std::nullopt;
    const int n = int(blocks.size());
    const int missing = int(p->missing.size());
    const int allowed = std::min(2, std::max(1, n / 10));
    if (missing < 1 || missing > allowed || p->duplicates) return std::nullopt;
    return insert_missing(p->order, p->missing, xycut_plus_order(blocks));
}

std::optional<std::vector<int>> supplement_incomplete_reading_order(const std::string & raw,
                                                                    const std::vector<block> & blocks) {
    const auto p = partial_reading_order(raw, int(blocks.size()));
    if (!p) return std::nullopt;
    const int n = int(blocks.size());
    if (p->missing.empty() && !p->duplicates) return std::nullopt;
    if (n && int(p->order.size()) * 2 < n) return std::nullopt;
    auto out = insert_missing(p->order, p->missing, xycut_plus_order(blocks));
    std::vector<int> sorted = out;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < n; ++i)
        if (sorted[size_t(i)] != i)
            throw std::invalid_argument("XY-Cut++ supplementation did not produce an exact permutation");
    return out;
}

// ------------------------------------------------------------ postprocessing
void postprocess(block & b, const std::string & raw_utf8, const std::string & table_format) {
    const u32 raw = cps(raw_utf8);
    switch (b.type) {
    case block_type::table: {
        if (table_format == "markdown") throw std::invalid_argument("element_formats.table=markdown is not ported");
        b.content = utf8(py_str::strip(text::strip_table_code_fence(raw)));
        b.format = content_format::html;
        return;
    }
    case block_type::formula: {
        const auto parts = text::postprocess_formula(raw);
        b.content = utf8(parts[0]);
        b.format = content_format::latex;
        b.text_before = utf8(parts[1]);
        b.text_after = utf8(parts[2]);
        b.generated_role.clear();
        return;
    }
    case block_type::code_block:
        b.content = utf8(text::postprocess_code(raw));
        b.format = content_format::markdown;
        return;
    case block_type::chemical_structure:
        b.content = utf8(text::postprocess_chemical(raw));
        b.format = content_format::markdown;
        return;
    default:
        b.content = utf8(text::postprocess_text(raw));
        b.format = content_format::text;
        return;
    }
}
std::string move_formula_numbering_outside_math(const std::string & t) {
    return utf8(text::move_formula_numbering_outside_math(cps(t)));
}
std::string space_markdown_math(const std::string & t) {
    return utf8(text::space_markdown_math(cps(t)));
}
std::string clean_formula_markdown_delimiters(const std::string & t) {
    return utf8(text::clean_formula_markdown_delimiters(cps(t)));
}

// ------------------------------------------------------------------ cleanup
namespace {
u32 normalize_cross_category(const u32 & content) {
    u32 n = py_str::strip(content);
    const std::pair<const char *, const char *> wrappers[] = {
        { "$$", "$$" }, { "$", "$" }, { "\\[", "\\]" }, { "\\(", "\\)" }
    };
    for (const auto & w : wrappers) {
        const u32 o = py_str::U(w.first), c = py_str::U(w.second);
        if (py_str::startswith(n, o) && py_str::endswith(n, c) && n.size() >= o.size() + c.size()) {
            n = py_str::strip(n.substr(o.size(), n.size() - o.size() - c.size()));
            break;
        }
    }
    return py_str::remove_space(n);
}
} // namespace

std::vector<block> dedupe_systematic_cross_category(const std::vector<block> & blocks) {
    auto cross = [](block_type t) { return t == block_type::text || t == block_type::formula; };
    std::vector<std::pair<std::array<double, 4>, std::vector<size_t>>> groups;
    for (size_t p = 0; p < blocks.size(); ++p) {
        if (!cross(blocks[p].type)) continue;
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto & g) { return g.first == blocks[p].bbox; });
        if (it == groups.end())
            groups.push_back({ blocks[p].bbox, { p } });
        else
            it->second.push_back(p);
    }
    std::set<size_t> remove;
    int duplicate_groups = 0;
    auto both = [&](const std::vector<size_t> & ps) {
        bool t = false, f = false, other = false;
        for (size_t p : ps) {
            if (blocks[p].type == block_type::text)
                t = true;
            else if (blocks[p].type == block_type::formula)
                f = true;
            else
                other = true;
        }
        return t && f && !other;
    };
    for (const auto & g : groups) {
        if (!both(g.second)) continue;
        std::vector<std::pair<u32, std::vector<size_t>>> by_content;
        for (size_t p : g.second) {
            const u32 n = normalize_cross_category(cps(blocks[p].content));
            if (n.empty()) continue;
            auto it = std::find_if(by_content.begin(), by_content.end(), [&](const auto & c) { return c.first == n; });
            if (it == by_content.end())
                by_content.push_back({ n, { p } });
            else
                it->second.push_back(p);
        }
        for (const auto & c : by_content) {
            if (c.second.size() < 2 || !both(c.second)) continue;
            size_t kept = c.second[0];
            auto score = [&](size_t p) { return blocks[p].layout_score ? *blocks[p].layout_score : 0.0; };
            for (size_t p : c.second)
                if (std::make_pair(score(p), -(long long)p) > std::make_pair(score(kept), -(long long)kept)) kept = p;
            for (size_t p : c.second)
                if (p != kept) remove.insert(p);
            duplicate_groups += 1;
        }
    }
    size_t content_count = 0;
    for (const auto & b : blocks) content_count += cross(b.type);
    const bool systematic = duplicate_groups >= 2 && remove.size() * 100 >= content_count * 10;
    if (remove.empty() || !systematic) return blocks;
    std::vector<block> out;
    for (size_t p = 0; p < blocks.size(); ++p)
        if (!remove.count(p)) out.push_back(blocks[p]);
    return out;
}

std::vector<block> expand_formula_numbering(const std::vector<block> & blocks) {
    std::vector<block> expanded;
    auto numbering = [](const block & source, const std::string & content) {
        block b;
        b.idx = source.idx;
        b.type = block_type::text;
        b.bbox = source.bbox;
        b.content = content;
        b.format = content_format::text;
        b.generated_role = k_formula_numbering_role;
        b.uid = next_uid();
        return b;
    };
    for (const auto & b : blocks) {
        if (b.type != block_type::formula) {
            expanded.push_back(b);
            continue;
        }
        const std::string before = utf8(py_str::strip(cps(b.text_before)));
        const std::string after = utf8(py_str::strip(cps(b.text_after)));
        if (!before.empty()) expanded.push_back(numbering(b, before));
        expanded.push_back(b);
        if (!after.empty()) expanded.push_back(numbering(b, after));
    }
    for (size_t i = 0; i < expanded.size(); ++i) expanded[i].idx = int(i);
    return expanded;
}

namespace {
// round(v) then numpy slice bounds for the ink checks.
struct ibox {
    long long x, y, u, v;
};
ibox rounded(const std::array<double, 4> & b) {
    return { py_str::round_half_even(b[0]), py_str::round_half_even(b[1]), py_str::round_half_even(b[2]),
             py_str::round_half_even(b[3]) };
}
long long slice_norm(long long v, long long n) {
    if (v < 0) v += n;
    return std::min(std::max(v, 0LL), n);
}
struct ink_mask {
    long long w = 0, h = 0;
    std::vector<uint8_t> ink, covered;
    void cover(long long r0, long long r1, long long c0, long long c1) {
        r0 = slice_norm(r0, h), r1 = slice_norm(r1, h), c0 = slice_norm(c0, w), c1 = slice_norm(c1, w);
        for (long long r = r0; r < r1; ++r)
            for (long long c = c0; c < c1; ++c) covered[size_t(r * w + c)] = 1;
    }
    bool sufficient(double threshold) const {
        long long n = 0, both = 0;
        for (size_t i = 0; i < ink.size(); ++i) {
            n += ink[i];
            both += ink[i] && covered[i];
        }
        if (!n) return false;
        return double(both) / double(n) >= threshold;
    }
};
ink_mask make_mask(const std::vector<uint8_t> & gray, int width, long long x, long long y, long long u, long long v) {
    ink_mask m;
    m.w = u - x;
    m.h = v - y;
    m.ink.resize(size_t(m.w * m.h));
    m.covered.assign(size_t(m.w * m.h), 0);
    for (long long r = 0; r < m.h; ++r)
        for (long long c = 0; c < m.w; ++c) m.ink[size_t(r * m.w + c)] = gray[size_t((y + r) * width + (x + c))] < 200;
    return m;
}
bool key_equal(const block & a, const block & b) {
    return a.type == b.type && a.bbox == b.bbox;
}
} // namespace

std::vector<block> dedupe_visual_parents(const std::vector<block> & raw, const std::vector<block> & blocks,
                                         const rgb_page & image, bool allow_mixed) {
    const std::vector<uint8_t> gray = cv_compat::rgb_to_gray_u8(image.rgb, image.width, image.height);
    auto visual = [](block_type t) { return t == block_type::picture || t == block_type::chemical_structure; };
    struct decision {
        const block * parent;
        std::vector<const block *> children, captions;
    };
    std::vector<decision> decisions;
    for (const auto & parent : raw) {
        if (!visual(parent.type) || !parent.layout_score) continue;
        std::vector<const block *> children;
        for (const auto & child : raw) {
            if (child.idx == parent.idx) continue;
            if (!(child.type == parent.type || (allow_mixed && visual(child.type)))) continue;
            if (!child.layout_score) continue;
            if (!(allow_mixed || *child.layout_score >= *parent.layout_score + 0.1)) continue;
            if (!(area(child.bbox) < 0.65 * area(parent.bbox))) continue;
            if (!inside(child, parent)) continue;
            children.push_back(&child);
        }
        if (children.size() < 2) continue;
        if (allow_mixed) {
            int stronger = 0;
            for (const block * c : children) stronger += *c->layout_score >= *parent.layout_score + 0.1;
            if (stronger < 2) continue;
        }
        bool overlap = false;
        for (size_t i = 0; i < children.size() && !overlap; ++i)
            for (size_t j = i + 1; j < children.size(); ++j) {
                const double inter = intersection(children[i]->bbox, children[j]->bbox);
                if (inter / std::max(area(children[i]->bbox) + area(children[j]->bbox) - inter, 1e-9) > 0.1) {
                    overlap = true;
                    break;
                }
            }
        if (overlap) continue;
        ibox pb = rounded(parent.bbox);
        long long x = std::max(0LL, pb.x), y = std::max(0LL, pb.y);
        long long u = std::min((long long)image.width, pb.u), v = std::min((long long)image.height, pb.v);
        if (x >= u || y >= v) continue;
        ink_mask mask = make_mask(gray, image.width, x, y, u, v);
        std::vector<const block *> captions;
        for (const auto & b : raw)
            if ((b.type == block_type::caption || (allow_mixed && b.type == block_type::title)) && inside(b, parent))
                captions.push_back(&b);
        std::vector<const block *> covering = children;
        covering.insert(covering.end(), captions.begin(), captions.end());
        for (const block * b : covering) {
            const ibox r = rounded(b->bbox);
            mask.cover(std::max(0LL, r.y - y), std::min(v - y, r.v - y), std::max(0LL, r.x - x),
                       std::min(u - x, r.u - x));
        }
        if (!mask.sufficient(0.999)) continue;
        decisions.push_back({ &parent, children, captions });
    }
    std::map<int, int> raw_ids;
    for (const auto & b : raw) raw_ids[b.idx] += 1;
    auto final_count = [&](const block & k) {
        int n = 0;
        for (const auto & b : blocks) n += key_equal(b, k);
        return n;
    };
    std::set<int> planned;
    for (const auto & d : decisions) planned.insert(d.parent->idx);
    std::vector<const block *> removed;
    for (const auto & d : decisions) {
        bool skip = raw_ids[d.parent->idx] != 1;
        for (const block * c : d.children) skip = skip || raw_ids[c->idx] != 1;
        if (skip) continue;
        if (final_count(*d.parent) != 1) continue;
        for (const block * c : d.children) skip = skip || final_count(*c) != 1;
        for (const block * c : d.captions) skip = skip || final_count(*c) != 1;
        if (skip) continue;
        for (const block * c : d.children) skip = skip || planned.count(c->idx);
        if (skip) continue;
        removed.push_back(d.parent);
    }
    std::vector<block> out;
    for (const auto & b : blocks) {
        bool drop = false;
        for (const block * r : removed) drop = drop || key_equal(b, *r);
        if (!drop) out.push_back(b);
    }
    return out;
}

namespace {
bool structured_text(const u32 & raw) {
    for (char32_t c : raw)
        if (c == U'$' || c == U'`' || c == U'\\' || c == U'<' || c == U'>' || c == U'[' || c == U']' || c == U'*' ||
            c == U'_' || c == U'~' || c == U'|')
            return true;
    // r" {2,}\r?\n"
    for (size_t i = 0; i + 1 < raw.size(); ++i) {
        if (raw[i] != U' ' || raw[i + 1] != U' ') continue;
        size_t j = i + 2;
        while (j < raw.size() && raw[j] == U' ') ++j;
        if (j < raw.size() && raw[j] == U'\r') ++j;
        if (j < raw.size() && raw[j] == U'\n') return true;
    }
    std::vector<size_t> starts = { 0 };
    for (size_t i = 0; i < raw.size(); ++i)
        if (raw[i] == U'\n') starts.push_back(i + 1);
    for (size_t s : starts) {
        // (?m:^[ ]{0,3}(?:=+|-+)[ \t]*$)
        size_t j = s, sp = 0;
        while (j < raw.size() && raw[j] == U' ' && sp < 3) ++j, ++sp;
        if (j < raw.size() && (raw[j] == U'=' || raw[j] == U'-')) {
            const char32_t ch = raw[j];
            while (j < raw.size() && raw[j] == ch) ++j;
            while (j < raw.size() && (raw[j] == U' ' || raw[j] == U'\t')) ++j;
            if (j == raw.size() || raw[j] == U'\n') return true;
        }
        // (?m)^(?: {4}|\t| {0,3}(?:#{1,6}\s|(?:[-+*]|\d+[.)])\s))
        if (py_str::startswith(raw, py_str::U("    "), s)) return true;
        if (s < raw.size() && raw[s] == U'\t') return true;
        j = s, sp = 0;
        while (j < raw.size() && raw[j] == U' ' && sp < 3) ++j, ++sp;
        if (j < raw.size() && raw[j] == U'#') {
            size_t h = j;
            while (h < raw.size() && raw[h] == U'#' && h - j < 6) ++h;
            if (h < raw.size() && py_str::isspace(raw[h])) return true;
        } else if (j < raw.size() && (raw[j] == U'-' || raw[j] == U'+' || raw[j] == U'*')) {
            if (j + 1 < raw.size() && py_str::isspace(raw[j + 1])) return true;
        } else if (j < raw.size() && py_str::isdecimal(raw[j])) {
            size_t d = j;
            while (d < raw.size() && py_str::isdecimal(raw[d])) ++d;
            if (d < raw.size() && (raw[d] == U'.' || raw[d] == U')') && d + 1 < raw.size() &&
                py_str::isspace(raw[d + 1]))
                return true;
        }
    }
    return false;
}
} // namespace

std::vector<block> dedupe_complete_text(const std::vector<block> & blocks) {
    std::vector<size_t> text_positions;
    for (size_t i = 0; i < blocks.size(); ++i)
        if (blocks[i].type == block_type::text) text_positions.push_back(i);
    std::vector<u32> norm(blocks.size());
    for (size_t i : text_positions) norm[i] = py_str::remove_space(cps(blocks[i].content));
    std::set<size_t> removed;
    std::vector<std::pair<size_t, std::vector<size_t>>> groups;
    std::vector<size_t> by_area = text_positions;
    std::stable_sort(by_area.begin(), by_area.end(), [&](size_t a, size_t b) {
        return std::make_pair(-area(blocks[a].bbox), a) < std::make_pair(-area(blocks[b].bbox), b);
    });
    for (size_t parent_pos : by_area) {
        if (removed.count(parent_pos)) continue;
        const block & parent = blocks[parent_pos];
        const u32 & parent_text = norm[parent_pos];
        std::vector<size_t> eligible;
        for (size_t child_pos : text_positions) {
            if (child_pos == parent_pos || removed.count(child_pos)) continue;
            const block & child = blocks[child_pos];
            if (area(child.bbox) > 0.8 * area(parent.bbox) || !inside(child, parent)) continue;
            const u32 & child_text = norm[child_pos];
            if (child_text.size() < 5 || parent_text.find(child_text) == u32::npos) continue;
            eligible.push_back(child_pos);
        }
        if (eligible.empty()) continue;
        std::vector<size_t> positions = eligible;
        positions.push_back(parent_pos);
        std::sort(positions.begin(), positions.end());
        if (positions.size() != positions.back() - positions.front() + 1) continue;
        size_t cursor = 0;
        bool complete = true;
        for (size_t child_pos : eligible) {
            const size_t found = parent_text.find(norm[child_pos], cursor);
            if (found == u32::npos) {
                complete = false;
                break;
            }
            cursor = found + norm[child_pos].size();
        }
        if (complete) {
            removed.insert(eligible.begin(), eligible.end());
            groups.push_back({ parent_pos, eligible });
        }
    }
    std::set<size_t> actual_removed;
    std::map<size_t, block> replacements;
    for (const auto & [parent_pos, children] : groups) {
        const u32 raw = cps(blocks[parent_pos].content);
        u32 joined;
        for (size_t c : children) joined += norm[c];
        if (py_str::remove_space(raw) != joined) continue;
        std::vector<size_t> chars;
        for (size_t i = 0; i < raw.size(); ++i)
            if (!py_str::isspace(raw[i])) chars.push_back(i);
        const bool structured = structured_text(raw);
        size_t cursor = 0;
        u32 pieces;
        for (size_t i = 0; i < children.size(); ++i) {
            if (i) {
                u32 sep = raw.substr(chars[cursor - 1] + 1, chars[cursor] - chars[cursor - 1] - 1);
                if (!structured && (sep.find(U'\n') != u32::npos || sep.find(U'\r') != u32::npos))
                    sep = py_str::U("\n\n");
                pieces += sep;
            }
            const size_t length = norm[children[i]].size();
            pieces += raw.substr(chars[cursor], chars[cursor + length - 1] + 1 - chars[cursor]);
            cursor += length;
        }
        const u32 content = raw.substr(0, chars.front()) + pieces + raw.substr(chars.back() + 1);
        if (py_str::remove_space(content) != py_str::remove_space(raw))
            throw std::logic_error("complete-text assertion");
        block replacement = blocks[parent_pos];
        replacement.content = utf8(content);
        replacement.uid = next_uid(); // dataclasses.replace creates a new object
        replacements[parent_pos] = replacement;
        actual_removed.insert(children.begin(), children.end());
    }
    std::vector<block> out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (actual_removed.count(i)) continue;
        auto it = replacements.find(i);
        out.push_back(it == replacements.end() ? blocks[i] : it->second);
    }
    return out;
}

std::vector<block> dedupe_exact_textual(const std::vector<block> & blocks) {
    auto textual = [](block_type t) {
        return t == block_type::text || t == block_type::caption || t == block_type::title;
    };
    std::vector<std::vector<size_t>> groups;
    std::vector<size_t> heads;
    for (size_t p = 0; p < blocks.size(); ++p) {
        const block & b = blocks[p];
        if (!textual(b.type)) continue;
        if (py_str::strip(cps(b.content)).empty()) continue;
        bool finite = true;
        for (double v : b.bbox) finite = finite && std::isfinite(v);
        if (!finite || b.bbox[2] <= b.bbox[0] || b.bbox[3] <= b.bbox[1]) continue;
        bool placed = false;
        for (size_t g = 0; g < groups.size(); ++g) {
            const block & h = blocks[heads[g]];
            if (h.bbox == b.bbox && h.content == b.content && h.format == b.format && h.text_before == b.text_before &&
                h.text_after == b.text_after && h.generated_role == b.generated_role) {
                groups[g].push_back(p);
                placed = true;
                break;
            }
        }
        if (!placed) {
            groups.push_back({ p });
            heads.push_back(p);
        }
    }
    std::set<size_t> removed;
    for (const auto & ps : groups) {
        std::set<int> categories;
        for (size_t p : ps) categories.insert(int(blocks[p].type));
        if (categories.size() < 2 || categories.size() != ps.size()) continue;
        bool scores_ok = true;
        for (size_t p : ps) scores_ok = scores_ok && blocks[p].layout_score && std::isfinite(*blocks[p].layout_score);
        if (!scores_ok) continue;
        size_t winner = ps[0];
        for (size_t p : ps)
            if (std::make_pair(*blocks[p].layout_score, -(long long)p) >
                std::make_pair(*blocks[winner].layout_score, -(long long)winner))
                winner = p;
        for (size_t p : ps)
            if (p != winner) removed.insert(p);
    }
    if (removed.empty()) return blocks;
    std::vector<block> out;
    for (size_t p = 0; p < blocks.size(); ++p)
        if (!removed.count(p)) out.push_back(blocks[p]);
    return out;
}

// ---- full_overlap_dedupe.py (opt-in upstream setting)
namespace {
bool textual_type(block_type t) {
    return t != block_type::picture && t != block_type::table;
}
bool picture_text_type(block_type t) {
    return textual_type(t) && t != block_type::chemical_structure && t != block_type::stamp;
}
bool unprotected_textual_pair(const block & a, const block & b) {
    return textual_type(a.type) && textual_type(b.type) && a.type != block_type::code_block &&
           b.type != block_type::code_block && a.type != block_type::chemical_structure &&
           b.type != block_type::chemical_structure && !is_numbering(a) && !is_numbering(b);
}
bool formula_pair(const block & a, const block & b) {
    return unprotected_textual_pair(a, b) && (a.type == block_type::formula || b.type == block_type::formula);
}
bool valid_box(const std::array<double, 4> & b) {
    for (double v : b)
        if (!std::isfinite(v)) return false;
    return b[0] < b[2] && b[1] < b[3];
}
bool contains_box(const std::array<double, 4> & o, const std::array<double, 4> & i) {
    return o != i && o[0] <= i[0] && o[1] <= i[1] && o[2] >= i[2] && o[3] >= i[3];
}
bool overlaps(const std::array<double, 4> & a, const std::array<double, 4> & b) {
    const double inter = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                         std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const double aa = (a[2] - a[0]) * (a[3] - a[1]), ab = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (aa + ab - inter) >= 0.80 && inter / std::min(aa, ab) >= 0.95;
}
bool all_class(const u32 & s, bool allow_space) {
    if (s.empty()) return false;
    for (char32_t c : s)
        if (!(py_str::is_ascii_alpha(c) || py_str::is_ascii_digit(c) || c == U'(' || c == U')' || c == U'+' ||
              c == U'-' || c == U'=' || c == U'>' || (allow_space && c == U' ')))
            return false;
    return true;
}
u32 full_content(const block & b) {
    const u32 text = cps(b.text_before + b.content + b.text_after);
    if (b.type == block_type::code_block) return text;
    const u32 compact = py_str::remove_space(text);
    if (b.type != block_type::formula && b.type != block_type::text) return compact;
    const u32 normalized = normalize_cross_category(compact);
    u32 chemical;
    const u32 ce = py_str::U("\\ce{");
    if (py_str::startswith(normalized, ce) && py_str::endswith(normalized, py_str::U("}")) &&
        all_class(normalized.substr(4, normalized.size() - 5), true)) {
        chemical = normalized.substr(4, normalized.size() - 5);
    } else {
        // re.sub(r"\\mathrm\{([A-Za-z]+)\}", r"\1", ...)
        u32 s1;
        const u32 mr = py_str::U("\\mathrm{");
        for (size_t i = 0; i < normalized.size();) {
            if (py_str::startswith(normalized, mr, i)) {
                size_t j = i + mr.size();
                while (j < normalized.size() && py_str::is_ascii_alpha(normalized[j])) ++j;
                if (j > i + mr.size() && j < normalized.size() && normalized[j] == U'}') {
                    s1 += normalized.substr(i + mr.size(), j - i - mr.size());
                    i = j + 1;
                    continue;
                }
            }
            s1.push_back(normalized[i++]);
        }
        // re.sub(r"_\{([0-9]+)\}", r"\1", ...)
        u32 s2;
        for (size_t i = 0; i < s1.size();) {
            if (s1[i] == U'_' && i + 1 < s1.size() && s1[i + 1] == U'{') {
                size_t j = i + 2;
                while (j < s1.size() && py_str::is_ascii_digit(s1[j])) ++j;
                if (j > i + 2 && j < s1.size() && s1[j] == U'}') {
                    s2 += s1.substr(i + 2, j - i - 2);
                    i = j + 1;
                    continue;
                }
            }
            s2.push_back(s1[i++]);
        }
        // re.sub(r"\\(?:to|rightarrow)(?![A-Za-z])", "->", ...)
        u32 s3;
        for (size_t i = 0; i < s2.size();) {
            if (s2[i] == U'\\') {
                bool done = false;
                for (const char * w : { "to", "rightarrow" }) {
                    const u32 ww = py_str::U(w);
                    if (py_str::startswith(s2, ww, i + 1)) {
                        const size_t e = i + 1 + ww.size();
                        if (e >= s2.size() || !py_str::is_ascii_alpha(s2[e])) {
                            s3 += py_str::U("->");
                            i = e;
                            done = true;
                            break;
                        }
                    }
                }
                if (done) continue;
            }
            s3.push_back(s2[i++]);
        }
        chemical = s3;
    }
    if (py_str::contains(chemical, py_str::U("->")) && all_class(chemical, true)) return chemical;
    return normalized;
}
bool preserves_content(const block & winner, const block & loser, const rgb_page * image) {
    if (loser.type == block_type::table)
        return winner.type == block_type::table && !py_str::strip(cps(loser.content)).empty() &&
               winner.text_before == loser.text_before && winner.content == loser.content &&
               winner.text_after == loser.text_after;
    if (loser.type == block_type::picture) {
        if (winner.type != block_type::picture) return false;
        const auto &a = winner.bbox, &b = loser.bbox;
        if (a[0] <= b[0] && a[1] <= b[1] && a[2] >= b[2] && a[3] >= b[3]) return true;
        if (!image) return false;
        const auto gray = cv_compat::rgb_to_gray_u8(image->rgb, image->width, image->height);
        const ibox rb = rounded(b);
        const long long x = std::max(0LL, rb.x), y = std::max(0LL, rb.y);
        const long long u = std::min((long long)image->width, rb.u), v = std::min((long long)image->height, rb.v);
        if (x >= u || y >= v) return false;
        ink_mask mask = make_mask(gray, image->width, x, y, u, v);
        const ibox ra = rounded(a);
        const long long left = std::max(x, ra.x), top = std::max(y, ra.y);
        const long long right = std::min(u, ra.u), bottom = std::min(v, ra.v);
        if (left < right && top < bottom) mask.cover(top - y, bottom - y, left - x, right - x);
        return mask.sufficient(0.999);
    }
    if (!textual_type(winner.type)) return false;
    if (loser.type == block_type::code_block) {
        const u32 lc = full_content(loser);
        return winner.type == block_type::code_block && !lc.empty() && py_str::contains(full_content(winner), lc);
    }
    const u32 content = full_content(loser);
    return !content.empty() && py_str::contains(full_content(winner), content);
}
bool absorbs(const block & parent, const block & child, const rgb_page * image) {
    if (contains_box(parent.bbox, child.bbox))
        return unprotected_textual_pair(parent, child) || preserves_content(parent, child, image);
    if (!unprotected_textual_pair(parent, child)) return false;
    const auto &a = parent.bbox, &b = child.bbox;
    const double tol = (b[3] - b[1]) * 0.20;
    const double inter = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                         std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const double child_area = (b[2] - b[0]) * (b[3] - b[1]);
    return (a[2] - a[0]) * (a[3] - a[1]) > child_area && inter / child_area >= 0.80 && a[0] - tol <= b[0] &&
           a[1] - tol <= b[1] && a[2] + tol >= b[2] && a[3] + tol >= b[3] &&
           (formula_pair(parent, child) || preserves_content(parent, child, image));
}
} // namespace

std::vector<block> dedupe_full_overlaps(const std::vector<block> & blocks, const rgb_page * image) {
    std::set<size_t> valid;
    for (size_t p = 0; p < blocks.size(); ++p)
        if (valid_box(blocks[p].bbox)) valid.insert(p);
    std::set<size_t> remaining = valid;
    while (!remaining.empty()) {
        std::set<size_t> children;
        for (size_t c : remaining)
            for (size_t p : remaining)
                if (absorbs(blocks[p], blocks[c], image)) {
                    children.insert(c);
                    break;
                }
        std::set<size_t> roots;
        for (size_t r : remaining)
            if (!children.count(r)) roots.insert(r);
        std::vector<size_t> scored;
        for (size_t r : roots)
            if (blocks[r].layout_score && std::isfinite(*blocks[r].layout_score)) scored.push_back(r);
        std::stable_sort(scored.begin(), scored.end(), [&](size_t a, size_t b) {
            return std::make_pair(-*blocks[a].layout_score, a) < std::make_pair(-*blocks[b].layout_score, b);
        });
        std::vector<size_t> kept;
        std::set<size_t> losers;
        for (size_t p : scored) {
            bool lose = false;
            for (size_t w : kept)
                if (overlaps(blocks[w].bbox, blocks[p].bbox) &&
                    (formula_pair(blocks[w], blocks[p]) || preserves_content(blocks[w], blocks[p], image))) {
                    lose = true;
                    break;
                }
            if (lose)
                losers.insert(p);
            else
                kept.push_back(p);
        }
        if (losers.empty()) {
            remaining = roots;
            break;
        }
        for (size_t l : losers) remaining.erase(l);
    }
    std::set<size_t> aliases;
    for (size_t p : remaining) {
        if (blocks[p].type != block_type::picture) continue;
        for (size_t t : remaining)
            if (picture_text_type(blocks[t].type) && !is_numbering(blocks[t]) &&
                !py_str::strip(cps(blocks[t].content)).empty() && blocks[t].bbox == blocks[p].bbox) {
                aliases.insert(p);
                break;
            }
    }
    for (size_t a : aliases) remaining.erase(a);
    if (remaining.size() == valid.size()) return blocks;
    std::vector<block> out;
    for (size_t p = 0; p < blocks.size(); ++p)
        if (!valid.count(p) || remaining.count(p)) out.push_back(blocks[p]);
    return out;
}

// ------------------------------------------------------------------ output
namespace {
std::string block_markdown_part(const block & b, const std::string & content) {
    switch (b.type) {
    case block_type::title:
        return "# " + content + "\n";
    case block_type::formula:
        return "$$\n" + clean_formula_markdown_delimiters(content) + "\n$$\n";
    case block_type::table:
    case block_type::picture:
    case block_type::code_block:
    case block_type::chemical_structure:
        return content + "\n";
    case block_type::page_header:
    case block_type::page_footer:
    case block_type::caption:
        return "*" + content + "*\n";
    case block_type::footnote:
        return "[^" + std::to_string(b.idx) + "]: " + content + "\n";
    default:
        return content + "\n";
    }
}
struct serialized_region {
    int idx;
    std::string type;
    std::array<double, 4> bbox;
    std::string content, format;
};
struct numbering_note {
    int idx;
    std::string content;
};
void serialize_final_blocks(const std::vector<block> & blocks, std::vector<serialized_region> & out,
                            std::vector<numbering_note> & notes) {
    std::map<int, const block *> formulas;
    for (const auto & b : blocks)
        if (b.type == block_type::formula) formulas[b.idx] = &b;
    std::map<int, std::map<std::string, std::string>> numbering;
    for (const auto & b : blocks) {
        if (!is_numbering(b)) continue;
        const block * owner = nullptr;
        for (const auto & [delta, field] : { std::pair<int, const char *>{ 1, "text_before" }, { -1, "text_after" } }) {
            auto it = formulas.find(b.idx + delta);
            if (it == formulas.end()) continue;
            const block & f = *it->second;
            const std::string & fv = std::string(field) == "text_before" ? f.text_before : f.text_after;
            if (f.bbox == b.bbox && py_str::strip(cps(fv)) == py_str::strip(cps(b.content))) {
                owner = &f;
                numbering[f.idx][field] = b.content;
                break;
            }
        }
        if (!owner) notes.push_back({ b.idx, b.content });
    }
    for (const auto & b : blocks) {
        if (is_numbering(b)) continue;
        std::string content = b.content;
        if (b.type == block_type::formula) {
            const auto & nums = numbering[b.idx];
            auto before = nums.count("text_before") ? nums.at("text_before") : "";
            auto after = nums.count("text_after") ? nums.at("text_after") : "";
            if (!before.empty()) content += "\\leqno " + before;
            if (!after.empty()) {
                const u32 a = cps(after);
                if (a.size() >= 2 && a.front() == U'(' && a.back() == U')')
                    content += "\\tag{" + utf8(a.substr(1, a.size() - 2)) + "}";
                else
                    content += "\\tag*{" + after + "}";
            }
        }
        const bool chem_eq = b.type == block_type::formula && b.content.find("\\ce") != std::string::npos;
        out.push_back(
            { b.idx, chem_eq ? "Chemical Equation" : type_name(b.type), b.bbox, content, format_name(b.format) });
    }
}
std::string bbox_json(const std::array<double, 4> & b) {
    return "[" + py_str::float_repr(b[0]) + "," + py_str::float_repr(b[1]) + "," + py_str::float_repr(b[2]) + "," +
           py_str::float_repr(b[3]) + "]";
}
} // namespace

std::string blocks_to_markdown(const std::vector<block> & blocks) {
    std::vector<const block *> sorted;
    for (const auto & b : blocks) sorted.push_back(&b);
    std::stable_sort(sorted.begin(), sorted.end(), [](const block * a, const block * b) { return a->idx < b->idx; });
    std::string md;
    const block * previous = nullptr;
    for (const block * b : sorted) {
        const std::string content = utf8(py_str::strip(cps(b->content)));
        if (content.empty()) continue;
        const std::string part = block_markdown_part(*b, content);
        const bool tight = previous && ((previous->type == block_type::formula && is_numbering(*b)) ||
                                        (is_numbering(*previous) && b->type == block_type::formula));
        if (previous && !tight) md += "\n";
        md += part;
        previous = b;
    }
    return md;
}

std::string serialize_page_json(const std::vector<block> & blocks, int page_num, int width, int height) {
    std::vector<serialized_region> regions;
    std::vector<numbering_note> notes;
    serialize_final_blocks(blocks, regions, notes);
    std::string out = "{\"page_num\":" + std::to_string(page_num) + ",\"page_size\":{\"w\":" + std::to_string(width) +
                      ",\"h\":" + std::to_string(height) + "},\"merge_content_from_pre_page\":false,\"blocks\":[";
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto & r = regions[i];
        if (i) out += ",";
        out += "{\"idx\":" + std::to_string(r.idx) + ",\"type\":" + quote(r.type) + ",\"bbox\":" + bbox_json(r.bbox) +
               ",\"content\":" + quote(r.content) + ",\"format\":" + quote(r.format) + "}";
    }
    out += "]";
    if (!notes.empty()) {
        out += ",\"unattached_formula_numbering\":[";
        for (size_t i = 0; i < notes.size(); ++i) {
            if (i) out += ",";
            out += "{\"idx\":" + std::to_string(notes[i].idx) + ",\"content\":" + quote(notes[i].content) + "}";
        }
        out += "]";
    }
    return out + "}";
}

} // namespace helpers

// ------------------------------------------------------------------ options
options parse_options(const std::string & json) {
    options o;
    if (json.empty()) return o;
    const value root = core_json_value::parse(json);
    if (root.kind != value::object_t) throw std::invalid_argument("OXR document options must be a JSON object");
    auto boolean = [](const value & v, const std::string & k) {
        if (v.kind != value::bool_t) throw std::invalid_argument(k + " must be bool");
        return v.b;
    };
    auto number = [](const value & v, const std::string & k) {
        if (!v.is_number() || !std::isfinite(v.number())) throw std::invalid_argument(k + " must be a finite number");
        return v.number();
    };
    auto unit = [&](const value & v, const std::string & k) {
        const double d = number(v, k);
        if (d < 0 || d > 1) throw std::invalid_argument(k + " must be a finite number in 0..1");
        return d;
    };
    for (const auto & [k, v] : root.o) {
        if (k == "file_name") {
            if (v.kind != value::string_t || v.s.empty() || v.s.find('\0') != std::string::npos)
                throw std::invalid_argument("file_name must be a nonempty NUL-free string");
            o.file_name = v.s;
        } else if (k == "output_formats") {
            if (v.kind != value::array_t || v.a.empty())
                throw std::invalid_argument("output_formats supports json and markdown");
            o.want_json = o.want_markdown = false;
            for (const auto & f : v.a) {
                if (f.kind != value::string_t || (f.s != "json" && f.s != "markdown"))
                    throw std::invalid_argument("output_formats supports json and markdown");
                (f.s == "json" ? o.want_json : o.want_markdown) = true;
            }
        } else if (k == "keep_header_footer") {
            o.keep_header_footer = boolean(v, k);
        } else if (k == "model_options") {
            if (v.kind == value::null_t) continue;
            if (v.kind != value::object_t) throw std::invalid_argument("Unsupported OXR generation option");
            for (const auto & [mk, mv] : v.o) {
                if (mk != "temperature" && mk != "top_p" && mk != "repetition_penalty")
                    throw std::invalid_argument("Unsupported OXR generation option");
                if (mv.kind == value::null_t) continue; // resolve_model_options ignores None
                if (mv.kind == value::bool_t) {
                    const double d = mv.b ? 1.0 : 0.0; // float(True) == 1.0
                    (mk == "temperature" ? o.temperature : mk == "top_p" ? o.top_p : o.repetition_penalty) = d;
                    continue;
                }
                const double d = number(mv, mk);
                (mk == "temperature" ? o.temperature : mk == "top_p" ? o.top_p : o.repetition_penalty) = d;
            }
        } else if (k == "element_formats") {
            if (v.kind == value::null_t) continue;
            if (v.kind != value::object_t) throw std::invalid_argument("element_formats must be an object");
            for (const auto & [ek, ev] : v.o) {
                if (ev.kind != value::string_t) throw std::invalid_argument("element_formats values must be strings");
                o.element_formats[ek] = ev.s;
            }
        } else if (k == "score_threshold" || k == "layout_score_threshold") {
            o.score_threshold = unit(v, k);
        } else if (k == "nms_iou_threshold" || k == "layout_nms_iou_threshold") {
            if (v.kind == value::null_t)
                o.nms_iou_threshold.reset();
            else
                o.nms_iou_threshold = unit(v, k);
        } else if (k == "provides_reading_order" || k == "layout_provides_reading_order") {
            o.provides_reading_order = boolean(v, k);
        } else if (k == "image_output_dir") {
            if (v.kind != value::string_t || v.s.find('\0') != std::string::npos)
                throw std::invalid_argument("image_output_dir must be a string");
            o.image_output_dir = v.s;
        } else if (k == "write_images") {
            o.write_images = boolean(v, k);
        } else if (k == "max_new_tokens") {
            if (v.kind != value::int_t || v.i <= 0 || v.i > 32768)
                throw std::invalid_argument("OXR document max_new_tokens must be an integer in 1..32768");
            o.max_new_tokens = int(v.i);
        } else if (k == "allow_truncated_content") {
            o.allow_truncated_content = boolean(v, k);
        } else if (k == "full_overlap_dedupe") {
            o.full_overlap_dedupe = boolean(v, k);
        } else if (k == "table_image_placeholders") {
            o.table_image_placeholders = boolean(v, k);
        } else {
            throw std::invalid_argument("Unknown OXR document option: " + k);
        }
    }
    return o;
}

// ------------------------------------------------------------- the pipeline
namespace {
struct runner {
    const std::vector<rgb_page> & pages;
    const options & opts;
    const recognizer & recognize;
    result & out;
    std::map<std::pair<int, int>, rec_result> outcomes; // last attempt per logical request

    rec_result call(const char * kind, int page, int block_pos, const uint8_t * rgb, int w, int h,
                    const std::string & prompt, int max_tokens, double penalty) {
        if (w <= 0 || h <= 0) throw std::invalid_argument("OXR image must be nonempty");
        if (prompt.empty() || prompt.find('\0') != std::string::npos)
            throw std::invalid_argument("OXR prompt must be a nonempty NUL-free string");
        if (opts.temperature != 0.0 || opts.top_p != 1.0)
            throw std::invalid_argument("Native OXR supports the deterministic temperature=0, top_p=1 profile");
        trace_entry t;
        t.kind = kind;
        t.page = page;
        t.block = block_pos;
        t.prompt = prompt;
        t.width = w;
        t.height = h;
        t.rgb_sha256 = py_str::sha256_hex(rgb, size_t(w) * size_t(h) * 3);
        t.max_tokens = max_tokens;
        t.repetition_penalty = penalty;
        rec_request req;
        req.rgb = rgb;
        req.width = w;
        req.height = h;
        req.prompt = prompt;
        req.max_tokens = max_tokens;
        req.repetition_penalty = penalty;
        out.native_requests += 1;
        rec_result r = recognize(req);
        if (r.finish_reason != "stop" && r.finish_reason != "length")
            r.finish_reason = r.eos ? "stop" : r.capped ? "length" : "unknown";
        t.finish_reason = r.finish_reason;
        t.native_json = r.native_json;
        out.trace.push_back(t);
        outcomes[{ page, block_pos }] = r;
        return r;
    }

    std::vector<int> reading_order(const std::vector<block> & blocks, int page) {
        const int n = int(blocks.size());
        if (n <= 1) {
            std::vector<int> id(size_t(std::max(n, 0)));
            for (int i = 0; i < n; ++i) id[size_t(i)] = i;
            return id;
        }
        const rgb_page & img = pages[size_t(page)];
        const std::string prompt = helpers::reading_order_prompt(blocks, img.width, img.height);
        const int max_tokens = std::min(opts.max_new_tokens, std::max(16, 4 * n));
        for (int attempt = 0; attempt < opts.retry_times; ++attempt) {
            try {
                const rec_result r =
                    call("reading_order", page, -1, img.rgb, img.width, img.height, prompt, max_tokens, 1.0);
                std::vector<int> order;
                if (helpers::parse_reading_order(r.text, n, order)) return order;
                if (auto repaired = helpers::repair_incomplete_reading_order(r.text, blocks)) return *repaired;
                if (attempt == opts.retry_times - 1) {
                    if (auto sup = helpers::supplement_incomplete_reading_order(r.text, blocks)) return *sup;
                    return helpers::xycut_plus_order(blocks);
                }
                throw std::invalid_argument("Invalid reading order: expected an exact permutation of IDs 0 through " +
                                            std::to_string(n - 1));
            } catch (const std::exception & e) {
                if (attempt == opts.retry_times - 1)
                    throw std::runtime_error("Reading order failed for page " + std::to_string(page) + " after " +
                                             std::to_string(opts.retry_times) + " attempts: " + e.what());
            }
        }
        throw std::logic_error("unreachable");
    }

    rec_result request_block(const block & b, int page, int pos, const std::vector<uint8_t> & rgb, int w, int h,
                             double penalty) {
        const char * prompt = k_prompt_text;
        if (b.type == block_type::table)
            prompt = k_prompt_table;
        else if (b.type == block_type::formula)
            prompt = k_prompt_formula;
        else if (b.type == block_type::code_block)
            prompt = k_prompt_code;
        else if (b.type == block_type::chemical_structure)
            prompt = k_prompt_chemical;
        for (int attempt = 0; attempt < opts.retry_times; ++attempt) {
            try {
                return call("block", page, pos, rgb.data(), w, h, prompt, opts.max_new_tokens, penalty);
            } catch (const std::exception & e) {
                if (attempt == opts.retry_times - 1) {
                    std::string message = e.what();
                    if (message.empty()) message = "Timeout after 120s";
                    throw std::runtime_error("Recognition failed for block " + std::to_string(b.idx) + " after " +
                                             std::to_string(opts.retry_times) + " attempts: " + message);
                }
            }
        }
        throw std::logic_error("unreachable");
    }

    void picture(block & b, const std::vector<uint8_t> & rgb, int w, int h, int page) {
        if (w <= 0 || h <= 0) throw std::invalid_argument("OpenCV: cannot write an empty Picture crop");
        const std::string base = py_str::splitext(py_str::basename(opts.file_name)).first;
        const std::string dir = py_str::join(py_str::join(opts.image_output_dir, base), "imgs");
        const std::string name = std::to_string(page) + "_" + std::to_string(b.idx) + ".png";
        const std::string path = py_str::join(dir, name);
        std::vector<uint8_t> bytes;
        auto sink = [](void * ctx, void * data, int size) {
            auto * v = static_cast<std::vector<uint8_t> *>(ctx);
            v->insert(v->end(), static_cast<uint8_t *>(data), static_cast<uint8_t *>(data) + size);
        };
        if (!stbi_write_png_to_func(sink, &bytes, w, h, 3, rgb.data(), w * 3) || bytes.empty())
            throw std::runtime_error("Unable to encode a Picture crop as PNG");
        if (opts.write_images) {
            std::error_code ec;
            std::filesystem::create_directories(dir.empty() ? "." : dir, ec);
            if (ec) throw std::runtime_error("Unable to create image directory " + dir + ": " + ec.message());
            FILE * f = std::fopen(path.c_str(), "wb");
            if (f) { // cv2.imwrite reports failure by return value only
                std::fwrite(bytes.data(), 1, bytes.size(), f);
                std::fclose(f);
            }
        }
        crisp_marker::asset a;
        a.name = "oxr/" + std::to_string(page) + "_" + std::to_string(b.idx) + ".png";
        a.bytes = std::move(bytes);
        out.picture_assets[b.uid] = a.name;
        out.assets.push_back(std::move(a));
        b.content = "![](" + path + ")";
        b.format = content_format::markdown;
    }

    block recognize_block(block b, int page, int pos) {
        const rgb_page & img = pages[size_t(page)];
        int w = 0, h = 0;
        std::vector<uint8_t> crop = helpers::crop_rgb(img, b.bbox, w, h);
        const auto cs = opts.element_formats.find("cs");
        if (b.type == block_type::picture ||
            (b.type == block_type::chemical_structure && cs != opts.element_formats.end() && cs->second == "image")) {
            picture(b, crop, w, h, page);
            return b;
        }
        rec_result r = request_block(b, page, pos, crop, w, h, opts.repetition_penalty);
        if (r.finish_reason == "length") {
            const bool text_like = b.type == block_type::text || b.type == block_type::title ||
                                   b.type == block_type::caption || b.type == block_type::footnote ||
                                   b.type == block_type::page_header || b.type == block_type::page_footer ||
                                   b.type == block_type::stamp;
            if (b.type == block_type::table) {
                const int tw = int(std::max(1LL, py_str::round_half_even(w * 0.75)));
                const int th = int(std::max(1LL, py_str::round_half_even(h * 0.75)));
                const std::vector<uint8_t> small = cv_compat::resize_area_u8(crop.data(), w, h, 3, tw, th);
                r = request_block(b, page, pos, small, tw, th, opts.repetition_penalty);
            } else if (text_like) {
                r = request_block(b, page, pos, crop, w, h, std::max(opts.repetition_penalty, 1.1));
            }
        }
        const auto tf = opts.element_formats.find("table");
        helpers::postprocess(b, r.text, tf == opts.element_formats.end() ? "html" : tf->second);
        return b;
    }
};

std::string ascii_lower(std::string s) {
    for (char & c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}
} // namespace

static result process_pages_impl(const std::vector<rgb_page> & pages, const value & layout, const options & opts,
                                 const recognizer & recognize, result & out);

result process_pages(const std::vector<rgb_page> & pages, const value & layout, const options & opts,
                     const recognizer & recognize, std::vector<trace_entry> * trace_out) {
    result out;
    try {
        process_pages_impl(pages, layout, opts, recognize, out);
    } catch (...) {
        if (trace_out) *trace_out = out.trace;
        throw;
    }
    if (trace_out) *trace_out = out.trace;
    return out;
}

static result process_pages_impl(const std::vector<rgb_page> & pages, const value & layout, const options & opts,
                                 const recognizer & recognize, result & out) {
    const auto started = std::chrono::steady_clock::now();
    if (pages.empty()) throw std::invalid_argument("OXR requires at least one page");
    if (opts.table_image_placeholders)
        throw std::invalid_argument("table_image_placeholders is not ported to the native route");
    const auto tf = opts.element_formats.find("table");
    if (tf != opts.element_formats.end() && tf->second == "markdown")
        throw std::invalid_argument("element_formats.table=markdown is not ported to the native route");
    if (opts.max_new_tokens <= 0 || opts.max_new_tokens > 32768)
        throw std::invalid_argument("OXR document max_new_tokens must be positive");
    if (!std::isfinite(opts.repetition_penalty) || opts.repetition_penalty <= 0)
        throw std::invalid_argument("repetition_penalty must be finite and positive");
    if (!recognize) throw std::invalid_argument("OXR document route requires a recognizer");
    for (const auto & p : pages)
        if (!p.rgb || p.width <= 0 || p.height <= 0) throw std::invalid_argument("OXR page image must be nonempty RGB");

    runner run{ pages, opts, recognize, out, {} };
    std::vector<std::pair<int, int>> sizes;
    for (const auto & p : pages) sizes.push_back({ p.width, p.height });
    auto pages_blocks = helpers::parse_layout_batch(layout, sizes, opts.score_threshold, opts.nms_iou_threshold);
    const auto raw_pages = pages_blocks;
    if (!opts.keep_header_footer)
        for (auto & blocks : pages_blocks)
            blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
                                        [](const block & b) {
                                            return b.type == block_type::page_header ||
                                                   b.type == block_type::page_footer;
                                        }),
                         blocks.end());

    std::vector<std::vector<int>> orders;
    if (!opts.provides_reading_order)
        for (size_t p = 0; p < pages_blocks.size(); ++p) orders.push_back(run.reading_order(pages_blocks[p], int(p)));
    for (size_t p = 0; p < pages_blocks.size(); ++p) {
        std::vector<block> recognized;
        for (size_t i = 0; i < pages_blocks[p].size(); ++i)
            recognized.push_back(run.recognize_block(pages_blocks[p][i], int(p), int(i)));
        if (!opts.provides_reading_order) {
            std::vector<block> ordered;
            for (int id : orders[p]) ordered.push_back(recognized[size_t(id)]);
            recognized = ordered;
        }
        pages_blocks[p] = recognized;
    }

    for (size_t p = 0; p < pages_blocks.size(); ++p) {
        auto & blocks = pages_blocks[p];
        blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
                                    [](const block & b) { return b.generated_role == k_table_image_role; }),
                     blocks.end());
        if (!opts.full_overlap_dedupe) blocks = helpers::dedupe_systematic_cross_category(blocks);
        blocks = helpers::expand_formula_numbering(blocks);
        if (opts.full_overlap_dedupe) {
            const auto visual = helpers::dedupe_visual_parents(raw_pages[p], blocks, pages[p], true);
            const auto final_blocks = helpers::dedupe_full_overlaps(visual, &pages[p]);
            std::set<uint64_t> visual_ids, final_ids;
            for (const auto & b : visual) visual_ids.insert(b.uid);
            for (const auto & b : final_blocks) final_ids.insert(b.uid);
            std::vector<block> recheck;
            for (const auto & b : blocks)
                if (!visual_ids.count(b.uid) || final_ids.count(b.uid)) recheck.push_back(b);
            blocks = helpers::dedupe_visual_parents(raw_pages[p], recheck, pages[p], true);
        } else {
            blocks = helpers::dedupe_visual_parents(raw_pages[p], blocks, pages[p], false);
            blocks = helpers::dedupe_complete_text(blocks);
            blocks = helpers::dedupe_exact_textual(blocks);
        }
    }

    out.pages = pages_blocks;
    out.page_sizes = sizes;
    std::string md;
    for (size_t p = 0; p < pages_blocks.size(); ++p) {
        if (p) md += "\n\n";
        md += helpers::blocks_to_markdown(pages_blocks[p]);
    }
    out.markdown = helpers::space_markdown_math(md);

    for (const auto & kv : run.outcomes)
        if (kv.second.capped || !kv.second.eos) out.unresolved += 1;
    if (out.unresolved && !opts.allow_truncated_content)
        throw std::runtime_error("OXR has " + std::to_string(out.unresolved) +
                                 " request(s) without EOS after upstream recovery; inspect last_trace, or explicitly "
                                 "allow_truncated_content");

    std::string outputs;
    if (opts.want_json) {
        outputs += "\"json\":{\"pages\":[";
        for (size_t p = 0; p < pages_blocks.size(); ++p) {
            if (p) outputs += ",";
            outputs += helpers::serialize_page_json(pages_blocks[p], int(p), sizes[p].first, sizes[p].second);
        }
        outputs += "]}";
    }
    if (opts.want_markdown) outputs += std::string(outputs.empty() ? "" : ",") + "\"markdown\":" + quote(out.markdown);
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    out.json = "{\"file_name\":" + quote(opts.file_name) + ",\"outputs\":{" + outputs +
               "},\"metadata\":{\"page_num\":" + std::to_string(pages.size()) +
               ",\"file_type\":" + quote(ascii_lower(py_str::splitext(opts.file_name).second)) +
               ",\"processing_time_ms\":" + std::to_string(ms) +
               ",\"crispembed\":{\"engine\":\"oxr-1.0\",\"layout_source\":\"provided\",\"native_requests\":" +
               std::to_string(out.native_requests) +
               ",\"unresolved_capped_requests\":" + std::to_string(out.unresolved) + ",\"blueprint_revision\":\"" +
               k_blueprint_revision + "\",\"document_route\":\"native\"}}}";
    return out;
}

// ------------------------------------------------------- shared document model
crisp_marker::document to_marker_document(const result & value, const std::string & source_name) {
    crisp_marker::document doc;
    doc.source_name = source_name;
    doc.backend = "oxr-1.0";
    doc.assets = value.assets;
    doc.metadata["oxr_blueprint_revision"] = k_blueprint_revision;
    for (size_t p = 0; p < value.pages.size(); ++p) {
        crisp_marker::page page;
        page.index = int(p);
        page.width = value.page_sizes[p].first;
        page.height = value.page_sizes[p].second;
        page.extraction_method = "oxr-1.0";
        int order = 0;
        for (const auto & b : value.pages[p]) {
            if (is_numbering(b)) continue; // folded into the formula below, as in the upstream JSON
            crisp_marker::block mb;
            mb.source = "oxr-1.0";
            mb.raw_label = type_name(b.type);
            mb.raw = b.content;
            mb.bbox = { b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3] };
            mb.polygon = {
                { b.bbox[0], b.bbox[1] }, { b.bbox[2], b.bbox[1] }, { b.bbox[2], b.bbox[3] }, { b.bbox[0], b.bbox[3] }
            };
            mb.order = order++;
            mb.confidence = b.layout_score ? *b.layout_score : 0.0;
            mb.text = b.content;
            switch (b.type) {
            case block_type::title:
                mb.type = "SectionHeader";
                mb.heading_level = 1;
                break;
            case block_type::formula:
                mb.type = "Equation";
                break;
            case block_type::table:
                mb.type = "Table";
                mb.html = b.content;
                break;
            case block_type::code_block: {
                mb.type = "Code";
                const std::string fence = "```\n";
                if (b.content.size() >= 8 && b.content.compare(0, 4, fence) == 0 &&
                    b.content.compare(b.content.size() - 4, 4, "\n```") == 0)
                    mb.text = b.content.substr(4, b.content.size() - 8);
                break;
            }
            case block_type::chemical_structure: {
                mb.type = "ChemicalBlock";
                const std::string open = "$\\smiles{";
                if (b.content.size() > open.size() + 2 && b.content.compare(0, open.size(), open) == 0)
                    mb.text = b.content.substr(open.size(), b.content.size() - open.size() - 2);
                break;
            }
            case block_type::picture:
                mb.type = "Picture";
                mb.text.clear();
                if (value.picture_assets.count(b.uid)) mb.image_name = value.picture_assets.at(b.uid);
                break;
            case block_type::caption:
                mb.type = "Caption";
                break;
            case block_type::footnote:
                mb.type = "Footnote";
                break;
            case block_type::page_header:
                mb.type = "PageHeader";
                break;
            case block_type::page_footer:
                mb.type = "PageFooter";
                break;
            default:
                mb.type = "Text";
                break;
            }
            mb.id = "/page/" + std::to_string(p) + "/" + mb.type + "/" + std::to_string(b.idx);
            page.blocks.push_back(std::move(mb));
        }
        doc.pages.push_back(std::move(page));
    }
    return doc;
}

} // namespace oxr_document
