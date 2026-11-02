// mineru_postprocess.h - faithful C++ translation of the pinned mineru-vl-utils
// document post-processing (opendatalab/mineru-vl-utils
// c9797cde8f54cc48306196fc6aebe73ea31dd4a3, Apache-2.0; see
// licenses/mineru-vl-utils-NOTICE). Changes from upstream: translated from
// Python to C++, logging removed, exceptions mapped to the same fallbacks.
//
// Strings are Python `str` values (Unicode code points, pyre::ustring) so
// lengths, slicing, regular expressions and case mapping follow CPython.
#pragma once

#include "pyre.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mineru_pp {

using pyre::ustring;

// Python ContentBlock (a dict). Key order is significant for the exact JSON
// profile: type, bbox, angle, content, then `extra_keys` in insertion order.
struct block {
    std::string type;
    std::array<double, 4> bbox = { { 0, 0, 0, 0 } };
    int angle = -1; // None
    bool has_content = false;
    ustring content;
    bool merge_prev = false;
    ustring sub_type;
    std::vector<std::string> extra_keys; // "merge_prev", "sub_type"
    // Internal helper metadata (always removed before output).
    std::vector<std::pair<ustring, ustring>> table_image_tokens; // token -> data URI
    bool has_token_map = false;
    bool absorbed = false;

    bool has_key(const std::string & k) const;
    void add_key(const std::string & k);
    void remove_key(const std::string & k);
    void set_type(const std::string & value); // ContentBlock.type setter
    void set_content(const ustring & value) {
        has_content = true;
        content = value;
    }
    void set_none() {
        has_content = false;
        content.clear();
    }
    bool truthy() const { return has_content && !content.empty(); }
};
using block_ptr = std::shared_ptr<block>;
using block_list = std::vector<block_ptr>;

block_ptr make_block(const std::string & type, const std::array<double, 4> & bbox, int angle, bool merge_prev = false);

struct options {
    bool simple_post_process = false;
    bool handle_equation_block = true;
    bool abandon_list = false;
    bool abandon_paratext = false;
    bool enable_table_formula_eq_wrap = false;
};

// MinerUClientHelper.post_process (includes its exception fallback).
block_list post_process_helper(const block_list & blocks, const options & opts);

// Exact Python-compatible renderers.
std::string python_float_repr(double value);
std::string json_quote(const ustring & value);      // json.dumps(ensure_ascii=False)
std::string blocks_json(const block_list & blocks); // json.dumps(..., separators=(",", ":"))
std::string json2md(const block_list & blocks);

// Individual upstream functions (exposed for differential tests).
ustring convert_otsl_to_html(const ustring & otsl);
ustring html_escape(const ustring & s);
ustring try_fix_equation_delimeters(const ustring & s);
ustring try_match_equation_left_right(const ustring & s);
ustring try_fix_equation_double_subscript(const ustring & s);
ustring try_fix_equation_eqqcolon(const ustring & s);
ustring try_fix_equation_big(const ustring & s);
ustring try_fix_equation_leq(const ustring & s);
ustring try_fix_unbalanced_braces(const ustring & s);
ustring process_equation(const ustring & s);
ustring add_equation_brackets(const ustring & s);
ustring try_convert_display_to_inline(const ustring & s);
ustring try_fix_macro_spacing_in_markdown(const ustring & s);
ustring try_move_underscores_outside(const ustring & s);
ustring replace_table_image_tokens(const ustring & content, const std::vector<std::pair<ustring, ustring>> & map);
ustring replace_table_formula_delimiters(const ustring & content, bool enabled);
bool convert_markdown_table_to_html(const ustring & content, ustring & out);
bool has_malformed_markdown_table(const ustring & content);
ustring extract_and_validate_mermaid_strict(const ustring & content);
bool detect_mermaid_repetition_issues(const ustring & content);
ustring try_fix_mermaid_node_id_legalization(const ustring & content);
ustring try_fix_mermaid_invalid_edge_lines(const ustring & content);
ustring try_fix_mermaid_invalid_subgraph_lines(const ustring & content);
ustring try_fix_chart_invalid_repetition(const ustring & content);
bool validate_content_repetition(const ustring & content);
ustring extract_largest_legal_markdown_table(const ustring & content);
struct image_analysis {
    ustring cls, sub_class, caption, content;
};
image_analysis process_image_or_chart(const ustring & content);
ustring convert_pure_table_content_to_html(const ustring & content);
ustring combined_equations(std::vector<ustring> contents);

} // namespace mineru_pp
