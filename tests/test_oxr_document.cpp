// test_oxr_document.cpp — native OXR document route: self-checks plus a
// JSON-lines differential driver used by tools/oxr_doc_parity.py.
//
//   test-oxr-document                 run built-in synthetic checks
//   test-oxr-document --helpers       read {"fn":...} requests on stdin, one per line
//   test-oxr-document --replay case.json
//        run the full native document route with recognition replayed from a
//        recorded trace (prompt, RGB SHA-256, budget, penalty -> native result)
#include "core/clean_exit.h"
#include "core/json.h"
#include "core/json_value.h"
#include "core/py_str.h"
#include "cv_compat.h"
#include "layout_detect.h"
#include "oxr_document.h"
#include "xycut_plus_plus.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using core_json_value::value;
using namespace oxr_document;

static int checks = 0, failures = 0;
static void check(bool ok, const std::string & label) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
    }
}
static std::string q(const std::string & s) {
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
static std::vector<uint8_t> hex_bytes(const std::string & hex) {
    std::vector<uint8_t> out(hex.size() / 2);
    auto nib = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
    for (size_t i = 0; i < out.size(); ++i) out[i] = uint8_t(nib(hex[2 * i]) * 16 + nib(hex[2 * i + 1]));
    return out;
}
static std::string to_hex(const std::vector<uint8_t> & b) {
    static const char * h = "0123456789abcdef";
    std::string s;
    for (uint8_t c : b) {
        s += h[c >> 4];
        s += h[c & 15];
    }
    return s;
}
static const value & need(const value & v, const char * key) {
    const value * x = v.get(key);
    if (!x) throw std::invalid_argument(std::string("missing ") + key);
    return *x;
}
static block block_from(const value & v) {
    block b;
    if (!type_from_name(need(v, "type").s, b.type)) throw std::invalid_argument("bad type " + need(v, "type").s);
    const value & bb = need(v, "bbox");
    for (int k = 0; k < 4; ++k) b.bbox[size_t(k)] = bb.a[size_t(k)].number();
    if (const value * x = v.get("idx")) b.idx = int(x->i);
    if (const value * x = v.get("content")) b.content = x->s;
    if (const value * x = v.get("format")) {
        const std::string f = x->s;
        b.format = f == "html"       ? content_format::html
                   : f == "latex"    ? content_format::latex
                   : f == "markdown" ? content_format::markdown
                                     : content_format::text;
    }
    if (const value * x = v.get("text_before")) b.text_before = x->s;
    if (const value * x = v.get("text_after")) b.text_after = x->s;
    if (const value * x = v.get("generated_role")) b.generated_role = x->s;
    if (const value * x = v.get("layout_score"))
        if (x->is_number()) b.layout_score = x->number();
    static uint64_t uid = 1000000;
    b.uid = ++uid;
    return b;
}
static std::vector<block> blocks_from(const value & v) {
    std::vector<block> out;
    for (const auto & x : v.a) out.push_back(block_from(x));
    return out;
}
static std::string order_json(const std::vector<int> & o) {
    std::string s = "[";
    for (size_t i = 0; i < o.size(); ++i) s += (i ? "," : "") + std::to_string(o[i]);
    return s + "]";
}

static std::string helper(const value & req) {
    const std::string fn = need(req, "fn").s;
    if (fn == "xycut")
        return "{\"order\":" + order_json(helpers::xycut_plus_order(blocks_from(need(req, "blocks")))) + "}";
    if (fn == "postprocess") {
        block b;
        type_from_name(need(req, "type").s, b.type);
        helpers::postprocess(b, need(req, "raw").s);
        return "{\"content\":" + q(b.content) + ",\"format\":" + q(format_name(b.format)) +
               ",\"text_before\":" + q(b.text_before) + ",\"text_after\":" + q(b.text_after) + "}";
    }
    if (fn == "space_markdown_math") return "{\"text\":" + q(helpers::space_markdown_math(need(req, "text").s)) + "}";
    if (fn == "clean_formula")
        return "{\"text\":" + q(helpers::clean_formula_markdown_delimiters(need(req, "text").s)) + "}";
    if (fn == "move_numbering")
        return "{\"text\":" + q(helpers::move_formula_numbering_outside_math(need(req, "text").s)) + "}";
    if (fn == "ro_prompt")
        return "{\"prompt\":" +
               q(helpers::reading_order_prompt(blocks_from(need(req, "blocks")), int(need(req, "width").i),
                                               int(need(req, "height").i))) +
               "}";
    if (fn == "ro_parse") {
        std::vector<int> order;
        const bool ok = helpers::parse_reading_order(need(req, "text").s, int(need(req, "count").i), order);
        return std::string("{\"order\":") + (ok ? order_json(order) : "null") + "}";
    }
    if (fn == "ro_repair" || fn == "ro_supplement") {
        const auto blocks = blocks_from(need(req, "blocks"));
        const auto r = fn == "ro_repair" ? helpers::repair_incomplete_reading_order(need(req, "text").s, blocks)
                                         : helpers::supplement_incomplete_reading_order(need(req, "text").s, blocks);
        return std::string("{\"order\":") + (r ? order_json(*r) : "null") + "}";
    }
    if (fn == "resize_area") {
        const auto px = hex_bytes(need(req, "pixels").s);
        const auto out =
            cv_compat::resize_area_u8(px.data(), int(need(req, "w").i), int(need(req, "h").i), int(need(req, "cn").i),
                                      int(need(req, "dw").i), int(need(req, "dh").i));
        return "{\"pixels\":" + q(to_hex(out)) + "}";
    }
    if (fn == "gray") {
        const auto px = hex_bytes(need(req, "pixels").s);
        return "{\"pixels\":" +
               q(to_hex(cv_compat::rgb_to_gray_u8(px.data(), int(need(req, "w").i), int(need(req, "h").i)))) + "}";
    }
    if (fn == "float_repr") {
        std::string out = "[";
        const value & xs = need(req, "values");
        for (size_t i = 0; i < xs.a.size(); ++i) out += (i ? "," : "") + q(py_str::float_repr(xs.a[i].number()));
        return "{\"repr\":" + out + "]}";
    }
    if (fn == "parse_layout") {
        std::vector<std::pair<int, int>> sizes;
        for (const auto & s : need(req, "sizes").a) sizes.push_back({ int(s.a[0].i), int(s.a[1].i) });
        std::optional<double> nms;
        if (need(req, "nms").is_number()) nms = need(req, "nms").number();
        const auto pages =
            helpers::parse_layout_batch(need(req, "payload"), sizes, need(req, "threshold").number(), nms);
        std::string out = "[";
        for (size_t p = 0; p < pages.size(); ++p) {
            out += p ? ",[" : "[";
            for (size_t i = 0; i < pages[p].size(); ++i) {
                const auto & b = pages[p][i];
                out += (i ? "," : "") + std::string("{\"idx\":") + std::to_string(b.idx) +
                       ",\"type\":" + q(type_name(b.type)) + ",\"bbox\":[" + py_str::float_repr(b.bbox[0]) + "," +
                       py_str::float_repr(b.bbox[1]) + "," + py_str::float_repr(b.bbox[2]) + "," +
                       py_str::float_repr(b.bbox[3]) + "],\"layout_score\":" + py_str::float_repr(*b.layout_score) +
                       "}";
            }
            out += "]";
        }
        return "{\"pages\":" + out + "]}";
    }
    if (fn == "page_cleanup") {
        // Final-page cleanup and assembly on already recognized, ordered blocks.
        const auto raw = blocks_from(need(req, "raw"));
        auto blocks = blocks_from(need(req, "blocks"));
        const auto px = hex_bytes(need(req, "pixels").s);
        rgb_page image{ int(need(req, "w").i), int(need(req, "h").i), px.data() };
        const bool full = need(req, "full_overlap").b;
        if (!full) blocks = helpers::dedupe_systematic_cross_category(blocks);
        blocks = helpers::expand_formula_numbering(blocks);
        if (full) {
            const auto visual = helpers::dedupe_visual_parents(raw, blocks, image, true);
            const auto fin = helpers::dedupe_full_overlaps(visual, &image);
            std::vector<block> recheck;
            for (const auto & b : blocks) {
                bool in_visual = false, in_final = false;
                for (const auto & v : visual) in_visual = in_visual || v.uid == b.uid;
                for (const auto & f : fin) in_final = in_final || f.uid == b.uid;
                if (!in_visual || in_final) recheck.push_back(b);
            }
            blocks = helpers::dedupe_visual_parents(raw, recheck, image, true);
        } else {
            blocks = helpers::dedupe_visual_parents(raw, blocks, image, false);
            blocks = helpers::dedupe_complete_text(blocks);
            blocks = helpers::dedupe_exact_textual(blocks);
        }
        return "{\"page\":" + helpers::serialize_page_json(blocks, 0, image.width, image.height) +
               ",\"markdown\":" + q(helpers::space_markdown_math(helpers::blocks_to_markdown(blocks))) + "}";
    }
    throw std::invalid_argument("unknown fn " + fn);
}

// Compact JSON of a layout response value (numbers via Python repr).
static std::string layout_json_text(const value & v) {
    switch (v.kind) {
    case value::null_t:
        return "null";
    case value::bool_t:
        return v.b ? "true" : "false";
    case value::int_t:
        return std::to_string(v.i);
    case value::float_t:
        return py_str::float_repr(v.f);
    case value::string_t:
        return q(v.s);
    case value::array_t: {
        std::string out = "[";
        for (size_t i = 0; i < v.a.size(); ++i) out += (i ? "," : "") + layout_json_text(v.a[i]);
        return out + "]";
    }
    default: {
        std::string out = "{";
        for (size_t i = 0; i < v.o.size(); ++i)
            out += (i ? "," : "") + q(v.o[i].first) + ":" + layout_json_text(v.o[i].second);
        return out + "}";
    }
    }
}

static int run_helpers() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string out;
        try {
            out = helper(core_json_value::parse(line));
        } catch (const std::exception & e) {
            out = "{\"error\":" + q(e.what()) + "}";
        }
        std::cout << out << "\n" << std::flush;
    }
    return 0;
}

// Replay: {"pages":[{"w","h","rgb_path"}], "layout_response":{...}, "options":{...},
//          "trace":[{"prompt","rgb_sha256","max_tokens","repetition_penalty","result":{...}}]}
static int run_replay(const std::string & path) {
    const value c = core_json_value::parse(read_file(path));
    std::vector<std::vector<uint8_t>> storage;
    std::vector<rgb_page> pages;
    for (const auto & p : need(c, "pages").a) {
        storage.push_back(std::vector<uint8_t>());
        const std::string bytes = read_file(need(p, "rgb_path").s);
        storage.back().assign(bytes.begin(), bytes.end());
    }
    size_t k = 0;
    for (const auto & p : need(c, "pages").a)
        pages.push_back({ int(need(p, "w").i), int(need(p, "h").i), storage[k++].data() });
    std::string options_json = "{}";
    if (const value * o = c.get("options_json")) options_json = o->s;
    const options opts = parse_options(options_json);
    std::multimap<std::string, const value *> table;
    std::map<std::string, size_t> used;
    for (const auto & t : need(c, "trace").a) {
        char pen[64];
        std::snprintf(pen, sizeof pen, "%.9g", float(need(t, "repetition_penalty").number()));
        const std::string key = need(t, "prompt").s + "\x1f" + need(t, "rgb_sha256").s + "\x1f" +
                                std::to_string(need(t, "max_tokens").i) + "\x1f" + pen;
        table.insert({ key, &need(t, "result") });
    }
    recognizer rec = [&](const rec_request & r) {
        char pen[64];
        std::snprintf(pen, sizeof pen, "%.9g", float(r.repetition_penalty));
        const std::string key = r.prompt + "\x1f" + py_str::sha256_hex(r.rgb, size_t(r.width) * r.height * 3) + "\x1f" +
                                std::to_string(r.max_tokens) + "\x1f" + pen;
        auto range = table.equal_range(key);
        size_t n = used[key]++;
        for (auto it = range.first; it != range.second; ++it, n--)
            if (!n) {
                const value & res = *it->second;
                rec_result out;
                if (const value * e = res.get("raise")) throw std::runtime_error(e->s);
                out.text = need(res, "text").s;
                out.eos = res.get("eos") && res.get("eos")->b;
                out.capped = res.get("capped") && res.get("capped")->b;
                if (const value * f = res.get("finish_reason")) out.finish_reason = f->s;
                return out;
            }
        throw std::runtime_error("replay: no recorded native result for request " + r.prompt.substr(0, 40) +
                                 " sha=" + py_str::sha256_hex(r.rgb, size_t(r.width) * r.height * 3));
    };
    try {
        // "layout_gguf": native OXR D-FINE default provider instead of a supplied response.
        value layout;
        std::string native_layout;
        if (const value * g = c.get("layout_gguf")) {
            layout_detect::context * dfine = nullptr;
            if (!layout_detect::load(&dfine, g->s.c_str(), 4)) throw std::runtime_error("cannot load " + g->s);
            try {
                layout = detect_layout_response(dfine, pages);
            } catch (...) {
                layout_detect::free(dfine);
                throw;
            }
            layout_detect::free(dfine);
            native_layout = ",\"native_layout_response\":" + layout_json_text(layout);
        } else {
            layout = need(c, "layout_response");
        }
        const result res = process_pages(pages, layout, opts, rec);
        std::cout << "{\"result\":" << res.json << ",\"native_requests\":" << res.native_requests << native_layout
                  << "}\n";
    } catch (const std::exception & e) {
        std::cout << "{\"error\":" << q(e.what()) << "}\n";
    }
    return 0;
}

static void self_checks() {
    using py_str::float_repr;
    check(float_repr(1.0) == "1.0", "repr 1.0");
    check(float_repr(0.1) == "0.1", "repr 0.1");
    check(float_repr(123.456) == "123.456", "repr 123.456");
    check(float_repr(1e16) == "1e+16", "repr 1e16");
    check(float_repr(1234567890123456.0) == "1234567890123456.0", "repr 1.23e15");
    check(float_repr(0.0001) == "0.0001", "repr 1e-4");
    check(float_repr(0.00001) == "1e-05", "repr 1e-5");
    check(py_str::round_half_even(2.5) == 2 && py_str::round_half_even(3.5) == 4 && py_str::round_half_even(-0.5) == 0,
          "banker's rounding");
    check(py_str::floordiv(5.0, std::numeric_limits<double>::infinity()) == 0.0, "5 // inf");
    check(py_str::sha256_hex((const uint8_t *)"abc", 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256 abc");
    check(py_str::splitext("a/b.tar.gz").second == ".gz" && py_str::splitext(".bashrc").second.empty(), "splitext");
    check(py_str::join("outputs", "x") == "outputs/x" && py_str::join("", "x") == "x", "posix join");
    const auto idx = xycut_plus_plus::numpy_argsort({ 3, 1, 2, 1 });
    check(idx == std::vector<long long>({ 1, 3, 2, 0 }), "numpy argsort small");
    block b;
    b.type = block_type::formula;
    helpers::postprocess(b, "$$x^2 \\tag{1}$$");
    check(b.content == "x^2" && b.text_after == "(1)", "formula numbering");
    b = block();
    b.type = block_type::code_block;
    helpers::postprocess(b, "```python\nprint(1)\n```");
    check(b.content == "```\nprint(1)\n```", "code fence");
    b = block();
    b.type = block_type::chemical_structure;
    helpers::postprocess(b, "$\\smiles{CCO}$");
    check(b.content == "$\\smiles{CCO}$", "smiles wrapper");
    std::vector<int> order;
    check(helpers::parse_reading_order(" 1 0\n", 2, order) && order == std::vector<int>({ 1, 0 }), "RO parse");
    check(!helpers::parse_reading_order("1, 0", 2, order), "RO parse rejects comma");
}

static int crispembed_test_main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--helpers") return run_helpers();
        if (argc > 2 && std::string(argv[1]) == "--replay") return run_replay(argv[2]);
        self_checks();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("test-oxr-document: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
