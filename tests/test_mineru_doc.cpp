// Weight-free MinerU document-route test driver.
//
//   test-mineru-doc re    < cases   regex engine vs CPython re (fuzz_regex.py)
//   test-mineru-doc fn    < cases   one upstream function per line (fuzz_postprocess.py)
//   test-mineru-doc page  FILE      replay recorded recognition through the full route
//   test-mineru-doc self            built-in unit checks (no inputs)
//
// Line protocol (re/fn): tab-separated hex-encoded UTF-8 fields; output one
// hex-encoded result per line, or "ERR <message>".
#include "core/clean_exit.h"
#include "mineru_document.h"
#include "mineru_postprocess.h"
#include "otsl_table.h"
#include "pyre.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string unhex(const std::string & h) {
    std::string o;
    for (size_t i = 0; i + 1 < h.size(); i += 2) o.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    return o;
}
std::string hex(const std::string & s) {
    static const char * d = "0123456789abcdef";
    std::string o;
    for (unsigned char c : s) {
        o += d[c >> 4];
        o += d[c & 15];
    }
    return o;
}
std::vector<std::string> fields(const std::string & line) {
    std::vector<std::string> out;
    std::istringstream in(line);
    std::string f;
    while (std::getline(in, f, '\t')) out.push_back(f);
    if (!line.empty() && line.back() == '\t') out.push_back(std::string());
    return out;
}

int mode_re() {
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto f = fields(line);
        try {
            if (f.size() < 4) throw std::invalid_argument("bad line");
            pyre::regex r(unhex(f[1]).c_str(), static_cast<unsigned>(std::stoi(f[2])));
            const auto s = pyre::from_utf8(unhex(f[3]));
            std::string out;
            if (f[0] == "spans") {
                for (auto & m : r.finditer(s)) {
                    for (int g = 0; g <= r.groups(); ++g)
                        out += std::to_string(m.start(g)) + "," + std::to_string(m.end(g)) + ";";
                    out += "|";
                }
            } else if (f[0] == "split") {
                for (auto & t : r.split(s)) out += hex(pyre::to_utf8(t)) + ",";
            } else if (f[0] == "full") {
                out = r.fullmatch(s) ? "1" : "0";
            }
            std::cout << hex(out) << "\n";
        } catch (const std::exception & e) {
            std::cout << "ERR " << e.what() << "\n";
        }
    }
    return 0;
}

std::string run_fn(const std::string & name, const std::string & arg) {
    using namespace mineru_pp;
    const ustring a = pyre::from_utf8(arg);
    auto s = [](const ustring & x) { return pyre::to_utf8(x); };
    if (name == "otsl") return s(convert_otsl_to_html(a));
    if (name == "otsl_basic") return otsl_table::to_html(arg, otsl_table::profile::mineru_basic_v1);
    if (name == "equation") return s(process_equation(a));
    if (name == "brackets") return s(add_equation_brackets(a));
    if (name == "left_right") return s(try_match_equation_left_right(a));
    if (name == "big") return s(try_fix_equation_big(a));
    if (name == "braces") return s(try_fix_unbalanced_braces(a));
    if (name == "double_subscript") return s(try_fix_equation_double_subscript(a));
    if (name == "text") {
        ustring t = try_convert_display_to_inline(a);
        t = try_fix_macro_spacing_in_markdown(t);
        return s(try_move_underscores_outside(t));
    }
    if (name == "eq_wrap") return s(replace_table_formula_delimiters(a, true));
    if (name == "md_table") {
        ustring out;
        return convert_markdown_table_to_html(a, out) ? s(out) : std::string("<None>");
    }
    if (name == "malformed") return has_malformed_markdown_table(a) ? "1" : "0";
    if (name == "mermaid_strict") return s(extract_and_validate_mermaid_strict(a));
    if (name == "mermaid_rep") return detect_mermaid_repetition_issues(a) ? "1" : "0";
    if (name == "mermaid_legal") return s(try_fix_mermaid_node_id_legalization(a));
    if (name == "mermaid_edges") return s(try_fix_mermaid_invalid_edge_lines(a));
    if (name == "mermaid_subgraph") return s(try_fix_mermaid_invalid_subgraph_lines(a));
    if (name == "chart_rep") return s(try_fix_chart_invalid_repetition(a));
    if (name == "chart_valid") return validate_content_repetition(a) ? "1" : "0";
    if (name == "image") {
        const auto r = process_image_or_chart(a);
        return s(r.cls) + "\x1f" + s(r.sub_class) + "\x1f" + s(r.caption) + "\x1f" + s(r.content);
    }
    if (name == "pure_table") return s(convert_pure_table_content_to_html(a));
    if (name == "float") return python_float_repr(std::stod(arg));
    if (name == "lower") return s(pyre::lower(a));
    if (name == "layout") return mineru_doc::layout_blocks_json(arg);
    throw std::invalid_argument("unknown function " + name);
}

int mode_fn() {
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto f = fields(line);
        try {
            if (f.size() < 2) throw std::invalid_argument("bad line");
            std::cout << hex(run_fn(f[0], unhex(f[1]))) << "\n";
        } catch (const std::exception & e) {
            std::cout << "ERR " << e.what() << "\n";
        }
    }
    return 0;
}

int mode_page(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try {
        std::cout << mineru_doc::replay_case_json(ss.str()) << "\n";
    } catch (const std::exception & e) {
        std::fprintf(stderr, "replay failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

int failures = 0;
void check(bool ok, const char * what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

int mode_self() {
    using namespace mineru_pp;
    check(python_float_repr(1.0) == "1.0", "repr 1.0");
    check(python_float_repr(0.001) == "0.001", "repr 0.001");
    check(python_float_repr(0.123) == "0.123", "repr 0.123");
    check(python_float_repr(1e-05) == "1e-05", "repr 1e-05");
    check(python_float_repr(1e16) == "1e+16", "repr 1e16");
    check(pyre::to_utf8(convert_otsl_to_html(pyre::U("<fcel>a<lcel><nl><fcel>b<fcel>c<nl>"))) ==
              "<table><tr><td colspan=\"2\">a</td></tr><tr><td>b</td><td>c</td></tr></table>",
          "otsl colspan");
    check(pyre::to_utf8(add_equation_brackets(pyre::U("x"))) == "\\[\nx\n\\]", "equation brackets");
    std::string err;
    check(mineru_doc::self_test(err), err.c_str());
    if (!failures) std::puts("test-mineru-doc self: all checks passed");
    return failures ? 1 : 0;
}
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "self";
    if (mode == "re") return mode_re();
    if (mode == "fn") return mode_fn();
    if (mode == "page" && argc > 2) return mode_page(argv[2]);
    if (mode == "self") return mode_self();
    std::fprintf(stderr, "usage: %s re|fn|page FILE|self\n", argv[0]);
    return 2;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
