// SPDX-License-Identifier: Apache-2.0
// Owned deterministic document fixtures; no model, network, Python or PDFium.
#include "marker_document.h"
#include "core/clean_exit.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crisp_marker;

namespace {
int checks = 0, failures = 0;
void check(bool condition, const char * name) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", name);
    }
}
bool contains(const std::string & text, const std::string & part) {
    return text.find(part) != std::string::npos;
}
void throws(const std::function<void()> & fn, const char * name) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, name);
}
line text_line(const std::string & text, double x, double y, double width, double size = 12, bool bold = false,
               bool mono = false) {
    line value;
    value.bbox = { x, y, x + width, y + size };
    span item;
    item.text = text;
    item.bbox = value.bbox;
    item.font_size = size;
    item.bold = bold;
    item.monospace = mono;
    value.spans.push_back(item);
    return value;
}
block text_block(const std::string & text, const std::string & type, box bounds) {
    block value;
    value.text = text;
    value.type = type;
    value.bbox = bounds;
    value.source = "owned-fixture";
    return value;
}
const block * find_type(const std::vector<block> & values, const std::string & type) {
    for (const auto & value : values) {
        if (value.type == type) return &value;
        if (const auto * nested = find_type(value.children, type)) return nested;
    }
    return nullptr;
}
document structured_document() {
    document d;
    d.source_name = "owned<&>.pdf";
    d.backend = "teleocr-ggml";
    page p;
    p.index = 0;
    p.width = 600;
    p.height = 800;
    p.extraction_method = "owned-fixture";
    auto title = text_block("A < B & C", "SectionHeader", { 40, 80, 420, 104 });
    title.lines = { text_line(title.text, 40, 80, 380, 24, true) };
    p.blocks.push_back(title);
    auto body = text_block("Hello <script>alert(1)</script> & world.", "Text", { 40, 130, 540, 144 });
    body.raw = body.text;
    body.raw_label = "text";
    body.generated_tokens = 7;
    body.confidence = 0.75;
    p.blocks.push_back(body);
    auto heading = text_block("Details", "SectionHeader", { 40, 180, 300, 196 });
    heading.lines = { text_line("Details", 40, 180, 260, 16, true) };
    p.blocks.push_back(heading);
    p.blocks.push_back(text_block("- First", "ListItem", { 40, 220, 300, 232 }));
    p.blocks.push_back(text_block("- Nested", "ListItem", { 60, 240, 300, 252 }));
    p.blocks.push_back(text_block("- Second", "ListItem", { 40, 260, 300, 272 }));
    p.blocks.push_back(text_block("if (a < b) {\n  return 1;\n}", "Code", { 40, 310, 500, 350 }));
    p.blocks.push_back(text_block("\\[x^2 + y^2 = z^2\\]", "Equation", { 100, 380, 500, 410 }));
    auto table = text_block("<fcel>Name<lcel><fcel>Count<nl><ucel><xcel><fcel>2<nl>", "Table", { 40, 440, 540, 480 });
    table.raw = table.text;
    p.blocks.push_back(table);
    auto picture = text_block("Figure <one>", "Picture", { 40, 510, 300, 620 });
    picture.image_name = "page-0-figure.png";
    p.blocks.push_back(picture);
    p.blocks.push_back(text_block("Figure 1: Safe caption", "Caption", { 40, 630, 400, 644 }));
    p.blocks.push_back(text_block("1 A retained footnote.", "Footnote", { 40, 720, 400, 730 }));
    d.pages.push_back(p);
    return d;
}

void test_layout() {
    std::vector<std::string> warnings;
    const std::string layout = "<box:100 200 900 800><label:title><up>\n"
                               "<box:50 50 500 50 500 400 50 400><label:table><rotate right>\n"
                               "<box:10 10 20 20><label:seal><left>\n"
                               "<box:0 0 1000 1000><label:unknown><mystery>\n";
    auto blocks = parse_teleocr_layout(layout, 600, 800, 8, warnings);
    check(blocks.size() == 4, "layout accepts rectangles and polygons");
    check(blocks[0].type == "SectionHeader" && blocks[0].raw_label == "title", "canonical/source labels preserved");
    check(blocks[0].bbox.x0 == 60 && blocks[0].bbox.y0 == 160 && blocks[0].bbox.x1 == 540 && blocks[0].bbox.y1 == 640 &&
              blocks[0].polygon.size() == 4,
          "layout scales exact normalized geometry");
    check(blocks[1].angle == 90 && blocks[2].angle == 270 && blocks[2].type == "Text",
          "layout orientation and seal dispatch");
    check(blocks[0].order == 0 && blocks[3].order == 3, "model reading order retained");
    check(blocks[0].raw == "<box:100 200 900 800><label:title><up>", "raw layout preserved");
    check(blocks[3].warnings.size() == 2, "unknown label and orientation diagnosed");
    const std::string bad = "<box:-1 0 30 30><label:text><up>\n"
                            "<box:0 0 1001 1000><label:text><up>\n"
                            "<box:20 10 10 20><label:text><up>\n"
                            "<box:0 0 500 500 0 500 500 0><label:text><up>\n"
                            "<box:0 0 50 0 100 0><label:text><up>\n"
                            "<box:0 0 100 100><label:not-a-label><up>\n"
                            "<box:0 0 100 100><label:text><up><extra>\n"
                            "<box:0 0 100 100><label:text>\n";
    warnings.clear();
    check(parse_teleocr_layout(bad, 600, 800, 8, warnings).empty() && warnings.size() == 8,
          "malformed geometry and syntax rejected");
    warnings.clear();
    check(parse_teleocr_layout(layout, 600, 800, 2, warnings).size() == 2 && !warnings.empty(),
          "layout block cap visible");
    std::string many;
    for (int i = 0; i < 500; ++i) many += "bad\n";
    warnings.clear();
    parse_teleocr_layout(many, 600, 800, 2, warnings);
    check(warnings.size() == 129, "layout diagnostics bounded");
    throws([&] { parse_teleocr_layout("", 0, 800, 8, warnings); }, "zero dimensions rejected");
    throws([&] { parse_teleocr_layout("", std::numeric_limits<double>::infinity(), 800, 8, warnings); },
           "infinite dimensions rejected");
    throws([&] { parse_teleocr_layout("", 600, 800, 0, warnings); }, "zero block limit rejected");
    warnings.clear();
    const auto chart = parse_teleocr_layout("<box:0 0 1000 1000><label:char><up>", 600, 800, 8, warnings);
    check(chart.size() == 1 && chart[0].type == "Table" && chart[0].raw_label == "char" && !chart[0].warnings.empty(),
          "scientific figure implied-table task keeps honest source label");
    std::string bounded(200000, '\n');
    check(parse_teleocr_layout(bounded, 600, 800, 8, warnings).empty() && contains(warnings.back(), "65536-line"),
          "pathological empty-layout parsing bounded");
}

void test_tables() {
    std::vector<std::string> warnings;
    const std::string table =
        otsl_to_html("<fcel>A & <script>x</script><lcel><fcel>B<nl><ucel><xcel><fcel>C<nl>", warnings);
    check(table == "<table><tr><td rowspan=\"2\" colspan=\"2\">A &amp; "
                   "&lt;script&gt;x&lt;/script&gt;</td><td>B</td></tr><tr><td>C</td></tr></table>",
          "OTSL exact row/column span HTML with escaped model text");
    check(warnings.empty(), "valid OTSL has no warnings");
    check(otsl_to_html("<fcel>A<ecel><nl><fcel>B<nl>", warnings) ==
              "<table><tr><td>A</td><td></td></tr><tr><td>B</td><td></td></tr></table>",
          "ragged OTSL rows padded");
    check(!warnings.empty(), "ragged rows diagnosed");
    warnings.clear();
    check(otsl_to_html("<lcel><fcel>A<nl>", warnings).empty() && !warnings.empty(), "orphan continuation rejected");
    warnings.clear();
    check(otsl_to_html("<fcel>A<lcel><nl><ucel><ecel><nl>", warnings).empty(),
          "inconsistent spanning rectangle rejected");
    check(otsl_to_html("<table><tr><td>A &amp; B</td></tr></table>", warnings) ==
              "<table><tr><td>A &amp; B</td></tr></table>",
          "safe HTML table canonicalized without double escaping");
    check(otsl_to_html("<table><tr><td onclick=\"bad()\">x</td></tr></table>", warnings).empty(),
          "table event attributes rejected");
    check(otsl_to_html("<table><tr><td><script>x</script></td></tr></table>", warnings).empty(),
          "active table tags rejected");
    check(otsl_to_html("<table><tr><td colspan=\"99999999\">x</td></tr></table>", warnings).empty(),
          "oversized HTML spans rejected");
    check(otsl_to_html("<table><tr><td>x</tr></td></table>", warnings).empty(), "malformed HTML nesting rejected");
    check(otsl_to_html("<fcel>" + std::string(10000, '<') + "<nl>", warnings).size() > 10000,
          "many literal less-than signs in OTSL cell remain linear and escaped");
    std::vector<line> lines;
    for (int r = 0; r < 4; ++r) {
        line row;
        row.bbox = { 40, 100.0 + r * 20, 480, 112.0 + r * 20 };
        for (int c = 0; c < 3; ++c) {
            auto l = text_line(r == 0 ? std::string("Heading") + std::to_string(c) : std::to_string(r * 10 + c),
                               40 + c * 170, 100 + r * 20, 100, 12, r == 0);
            row.spans.push_back(l.spans.front());
        }
        lines.push_back(row);
    }
    warnings.clear();
    const std::string digital = pdf_table_html(lines, warnings);
    check(digital ==
              "<table><tr><th>Heading0</th><th>Heading1</th><th>Heading2</th></tr><tr><td>10</td><td>11</td><td>12</"
              "td></tr><tr><td>20</td><td>21</td><td>22</td></tr><tr><td>30</td><td>31</td><td>32</td></tr></table>",
          "PDF grid exact header and cell matrix");
    check(warnings.empty(), "reliable PDF grid does not warn");
    check(pdf_table_html({ lines[0], lines[1] }, warnings).empty() && !warnings.empty(),
          "two-row ambiguous table falls back");
}

void test_text_quality() {
    check(text_is_usable("Crème brûlée — Ελληνικά 日本語 中文 مرحبا"), "Unicode language text accepted");
    check(text_is_usable("A"), "single meaningful letter accepted");
    check(!text_is_usable("") && !text_is_usable(" \t\n"), "empty text rejected");
    check(!text_is_usable("������") && !text_is_usable("abc\x01\x02"), "replacement and control glyphs rejected");
    check(!text_is_usable(std::string("x") + char(0xFF)), "invalid UTF-8 rejected");
    check(!text_is_usable(std::string(80, 'a')), "long repetition rejected");
    check(!text_is_usable("......"), "symbol-only text rejected");
    check(escape_html("<>&\"'\x01") == "&lt;&gt;&amp;&quot;&#39;&#xfffd;",
          "HTML escaping includes unsafe control characters");
}

void test_layout_heuristics() {
    page p;
    p.index = 0;
    p.width = 600;
    p.height = 800;
    p.text_lines = { text_line("Document heading", 40, 70, 500, 24, true),
                     text_line("This is a substantial line of body text to establish font size.", 40, 120, 500),
                     text_line("A second ordinary paragraph line continues here.", 40, 136, 500),
                     text_line("A third ordinary paragraph line follows.", 40, 152, 500),
                     text_line("Details", 40, 200, 250, 16, true),
                     text_line("- First bullet", 40, 240, 300),
                     text_line("- Second bullet", 40, 258, 300),
                     text_line("return a < b;", 40, 310, 300, 12, false, true),
                     text_line("1 A small footnote.", 40, 730, 350, 9) };
    build_text_blocks(p);
    check(p.blocks.front().type == "SectionHeader", "font/size heading detection");
    const block * body = find_type(p.blocks, "Text");
    check(body && body->lines.size() == 3 && contains(body->text, "size. A second"), "body lines merge into paragraph");
    check(find_type(p.blocks, "ListItem") != nullptr && find_type(p.blocks, "Code") != nullptr &&
              find_type(p.blocks, "Footnote") != nullptr,
          "list/code/footnote heuristic classification");
    document d;
    d.backend = "pdfium-heuristic";
    d.pages.push_back(p);
    options opts;
    opts.mode = 1;
    process_document(d, opts);
    const block * group = find_type(d.pages[0].blocks, "ListGroup");
    check(group && group->children.size() == 2, "adjacent list items grouped");
    std::vector<const block *> headings;
    for (const auto & b : d.pages[0].blocks)
        if (b.type == "SectionHeader") headings.push_back(&b);
    check(headings.size() == 2 && headings[0]->heading_level == 1 && headings[1]->heading_level == 2,
          "relative heading levels assigned");
    const block * code = find_type(d.pages[0].blocks, "Code");
    check(code && code->section_hierarchy.size() == 2, "section hierarchy inherited by content");
    page columns;
    columns.width = 600;
    columns.height = 800;
    columns.text_lines = {
        text_line("A left column line.", 40, 100, 220),    text_line("A right column line.", 340, 100, 220),
        text_line("Second left paragraph.", 40, 140, 220), text_line("Second right paragraph.", 340, 140, 220),
        text_line("Third left paragraph.", 40, 180, 220),  text_line("Third right paragraph.", 340, 180, 220)
    };
    build_text_blocks(columns);
    check(columns.blocks.size() == 6 && contains(columns.blocks[1].text, "Second left") &&
              contains(columns.blocks[3].text, "right"),
          "two-column reading order is column-major");
    check(!columns.warnings.empty(), "column heuristic provenance visible");
    page headed_columns;
    headed_columns.width = 600;
    headed_columns.height = 800;
    headed_columns.text_lines = { text_line("Two-column reading order", 40, 60, 500, 22, true),
                                  text_line("Left column", 40, 110, 150, 15, true),
                                  text_line("Right column", 340, 110, 150, 15, true),
                                  text_line("Left one stays together.", 40, 145, 220, 11),
                                  text_line("Right one begins here.", 340, 145, 220, 11),
                                  text_line("Left two follows one.", 40, 173, 220, 11),
                                  text_line("Right two follows one.", 340, 173, 220, 11),
                                  text_line("Left three ends this column.", 40, 201, 220, 11),
                                  text_line("Right three ends the page.", 340, 201, 220, 11) };
    build_text_blocks(headed_columns);
    check(find_type(headed_columns.blocks, "Table") == nullptr,
          "bold headings do not turn two prose columns into a table");
    std::string headed_order;
    for (const auto & b : headed_columns.blocks) headed_order += b.text + "\n";
    check(headed_order.find("Left three") < headed_order.find("Right column") &&
              headed_order.find("Right column") < headed_order.find("Right one"),
          "bold-column headers retain column-major body order");
    page rotated;
    rotated.width = 612;
    rotated.height = 792;
    line vertical_title;
    vertical_title.bbox = { 295.27, 82.024, 316.72, 246.364 };
    auto r1 = text_line("Rotated ", 299.714, 82.024, 17.006, 22, true).spans.front();
    r1.bbox = { 299.714, 82.024, 316.72, 176.734 };
    auto r2 = text_line("page", 295.27, 188.064, 17.05, 22, true).spans.front();
    r2.bbox = { 295.27, 188.064, 312.32, 246.364 };
    vertical_title.spans = { r1, r2 };
    line vertical_body;
    vertical_body.bbox = { 257.516, 81.176, 269.12, 280.508 };
    auto v1 = text_line("Rotation ", 259.844, 81.176, 9.276).spans.front();
    v1.bbox = { 259.844, 81.176, 269.12, 129.656 };
    auto v2 = text_line("preserves this sentence.", 257.516, 135.56, 11.604).spans.front();
    v2.bbox = { 257.516, 135.56, 269.12, 280.508 };
    vertical_body.spans = { v1, v2 };
    rotated.text_lines = { vertical_body, vertical_title };
    build_text_blocks(rotated);
    check(rotated.blocks.size() == 2 && rotated.blocks[0].text == "Rotated page" &&
              rotated.blocks[1].text == "Rotation preserves this sentence.",
          "90-degree vertical word flow restores geometric order");
    check(!rotated.warnings.empty() && contains(rotated.warnings[0], "90-degree"),
          "vertical reading-order policy disclosed");
    page counterclockwise;
    counterclockwise.width = 600;
    counterclockwise.height = 800;
    line first_ccw, second_ccw;
    first_ccw.bbox = { 90, 100, 102, 260 };
    auto c1 = text_line("First ", 90, 210, 12).spans.front();
    c1.bbox = { 90, 210, 102, 260 };
    auto c2 = text_line("line", 90, 100, 12).spans.front();
    c2.bbox = { 90, 100, 102, 180 };
    first_ccw.spans = { c1, c2 };
    second_ccw = first_ccw;
    second_ccw.bbox = { 150, 100, 162, 260 };
    second_ccw.spans[0].text = "Second ";
    for (auto & s : second_ccw.spans) {
        s.bbox.x0 += 60;
        s.bbox.x1 += 60;
    }
    counterclockwise.text_lines = { second_ccw, first_ccw };
    build_text_blocks(counterclockwise);
    check(counterclockwise.blocks.size() == 2 && counterclockwise.blocks[0].text == "First line" &&
              counterclockwise.blocks[1].text == "Second line" && contains(counterclockwise.warnings[0], "270-degree"),
          "270-degree upward word flow restores geometric order");
    page ambiguous;
    ambiguous.width = 600;
    ambiguous.height = 800;
    line one_span = text_line("Provider first", 210, 100, 12);
    one_span.bbox = { 210, 100, 222, 300 };
    one_span.spans[0].bbox = one_span.bbox;
    line other_span = one_span;
    other_span.spans[0].text = "Provider second";
    other_span.bbox = { 100, 100, 112, 300 };
    other_span.spans[0].bbox = other_span.bbox;
    ambiguous.text_lines = { one_span, other_span };
    build_text_blocks(ambiguous);
    check(ambiguous.blocks.size() == 2 && ambiguous.blocks[0].text == "Provider first" &&
              ambiguous.blocks[1].text == "Provider second" && contains(ambiguous.warnings[0], "ambiguous"),
          "ambiguous vertical orientation preserves provider order with warning");
    page code_page;
    code_page.width = 600;
    code_page.height = 800;
    code_page.text_lines = { text_line("if (left < right) {", 40, 100, 250, 11, false, true),
                             text_line("    print(\"safe & literal\");", 66.4, 118, 280, 11, false, true),
                             text_line("}", 40, 136, 8, 11, false, true) };
    build_text_blocks(code_page);
    check(code_page.blocks.size() == 1 && code_page.blocks[0].type == "Code" &&
              code_page.blocks[0].text == "if (left < right) {\n    print(\"safe & literal\");\n}",
          "code gold preserves exact leading indentation and line breaks");
}

void test_processors_and_renderers() {
    document d = structured_document();
    options opts;
    process_document(d, opts);
    const block * list = find_type(d.pages[0].blocks, "ListGroup");
    check(list && list->children.size() == 2 && list->children.front().children.size() == 1 &&
              list->children.front().children.front().children.size() == 1,
          "nested list indentation produces recursive structure");
    const block * figure = find_type(d.pages[0].blocks, "PictureGroup");
    check(figure && figure->children.size() == 2 && figure->children[1].type == "Caption",
          "caption grouped with adjacent image");
    check(d.pages[0].blocks.back().type == "Footnote", "footnote moved to end");
    const std::string md = render_document(d, "markdown");
    check(contains(md, "# A &lt; B &amp; C\n") && contains(md, "## Details\n"), "Markdown heading levels and escaping");
    check(contains(md, "- First\n  - Nested\n- Second"), "nested list Markdown exact structure");
    check(contains(md, "$$\nx^2 + y^2 = z^2\n$$"), "equation delimiters normalized");
    check(contains(md, "<td rowspan=\"2\" colspan=\"2\">Name</td>"), "spanning table preserved in Markdown");
    check(contains(md, "![Figure &lt;one&gt;](page-0-figure.png)"), "figure asset referenced");
    check(!contains(md, "<script>") && contains(md, "&lt;script&gt;"), "raw OCR cannot inject Markdown HTML");
    const std::string html = render_document(d, "html");
    check(contains(html, "<h1>") && contains(html, "<h2>") &&
              contains(html, "<ul><li>First<ul><li>Nested</li></ul></li><li>Second</li></ul>"),
          "HTML nested structure");
    check(contains(html, "if (a &lt; b)") && !contains(html, "<script>"), "HTML code/text escaping");
    const std::string json = render_document(d, "json");
    check(contains(json, "\"schema\":\"crispembed.marker/1\"") && contains(json, "\"backend\":\"teleocr-ggml\"") &&
              contains(json, "\"surya_parity\":\"not claimed;"),
          "JSON honest schema and backend provenance");
    check(contains(json, "\"raw_label\":\"text\"") && contains(json, "\"confidence\":0.75") &&
              contains(json, "\"generated_tokens\":7") && contains(json, "\"source\":\"owned-fixture\""),
          "JSON source and inference details retained");
    check(contains(json, "<content-ref src=\\\"/page/0/") && contains(json, "\"children\":["),
          "Marker-like recursive content references");
    const std::string chunks = render_document(d, "chunks");
    check(contains(chunks, "\"page_info\":{\"0\":") && contains(chunks, "\"page\":0") &&
              contains(chunks, "\"section_hierarchy\":{\"1\":"),
          "chunks include page and section hierarchy");
    throws([&] { render_document(d, "exe"); }, "unknown renderer rejected");

    document unsafe;
    page p;
    p.width = 600;
    p.height = 800;
    auto b = text_block("", "Text", { 0, 0, 100, 20 });
    b.html = "<img src=x onerror=alert(1)>";
    p.blocks.push_back(b);
    auto img = text_block("bad", "Picture", { 0, 100, 100, 200 });
    img.image_name = "javascript:alert(1)";
    p.blocks.push_back(img);
    unsafe.pages.push_back(p);
    process_document(unsafe, opts);
    const std::string escaped = render_document(unsafe, "html");
    check(!contains(escaped, "<img") && contains(escaped, "&lt;img"),
          "supplied unsafe HTML and image URLs cannot execute");
    check(!unsafe.pages[0].blocks[0].warnings.empty(), "discarded HTML diagnosed");
    unsafe.pages[0].blocks.push_back(
        text_block("x$$\n<script>alert(1)</script>\n$$y", "Equation", { 0, 300, 400, 320 }));
    const std::string math_md = render_document(unsafe, "markdown");
    check(!contains(math_md, "<script>") && contains(math_md, "\\$\\$") && contains(math_md, "\\lt{}script\\gt{}"),
          "malformed formula cannot close math and inject active HTML");
    page linked;
    linked.width = 600;
    linked.height = 800;
    auto safe = text_block("example", "Text", { 40, 40, 100, 52 });
    safe.lines = { text_line("example", 40, 40, 60) };
    safe.lines[0].spans[0].url = "https://example.com/?a=1&b=2";
    linked.blocks.push_back(safe);
    auto evil = safe;
    evil.lines[0].spans[0].url = "javascript:alert(1)";
    linked.blocks.push_back(evil);
    document links;
    links.pages.push_back(linked);
    const std::string linked_html = render_document(links, "html");
    check(contains(linked_html, "href=\"https://example.com/?a=1&amp;b=2\"") && !contains(linked_html, "javascript:"),
          "safe PDF links preserved and active URL schemes excluded");
    document broken;
    page bp;
    bp.width = 600;
    bp.height = 800;
    auto invalid = text_block(std::string("a") + char(0xFF), "Text", { 40, 100, 100, 112 });
    invalid.raw = invalid.text;
    invalid.token_cap_reached = true;
    bp.blocks.push_back(invalid);
    bp.raw_layout = std::string("x") + char(0xC2);
    bp.layout_token_cap_reached = true;
    broken.pages.push_back(bp);
    process_document(broken, opts);
    const std::string repaired = render_document(broken, "json");
    check(contains(repaired, "\"raw_bytes_hex\":\"61ff\"") && contains(repaired, "\"text_bytes_hex\":\"61ff\"") &&
              contains(repaired, "\"raw_layout_bytes_hex\":\"78c2\"") && contains(repaired, "a\xEF\xBF\xBD"),
          "malformed UTF-8 replaced for rendering with exact raw/text/layout bytes preserved");
    check(render_document(broken, "markdown").find(char(0xFF)) == std::string::npos &&
              render_document(broken, "html").find(char(0xFF)) == std::string::npos,
          "all public text renderers emit valid UTF-8 at truncated byte-token boundaries");
    document controls;
    page control_page;
    control_page.width = 600;
    control_page.height = 800;
    auto code_controls =
        text_block(std::string("code") + '\0' + "tail\n\tindent" + '\x01' + '\x7f', "Code", { 40, 100, 300, 140 });
    code_controls.raw = code_controls.text;
    auto math_controls = text_block(std::string("x") + '\0' + "<y" + '\x07', "Equation", { 40, 180, 300, 200 });
    math_controls.raw = math_controls.text;
    control_page.blocks = { code_controls, math_controls };
    controls.pages.push_back(control_page);
    process_document(controls, opts);
    const std::string control_md = render_document(controls, "markdown");
    check(control_md.size() == std::string(control_md.c_str()).size() &&
              control_md.find_first_of(std::string("\0\x01\x07\x7f", 4)) == std::string::npos,
          "Code and Equation Markdown remain complete and safe through NUL-terminated C-string consumers");
    check(contains(control_md, "code\xEF\xBF\xBDtail\n\tindent\xEF\xBF\xBD\xEF\xBF\xBD") &&
              contains(control_md, "x\xEF\xBF\xBD\\lt{}y\xEF\xBF\xBD"),
          "Code and Equation controls replaced without losing safe text, whitespace or math");
    const std::string control_json = render_document(controls, "json");
    check(contains(control_json, "\"raw\":\"code\\u0000tail\\n\\tindent\\u0001") &&
              contains(control_json, "\"raw\":\"x\\u0000<y\\u0007\"") &&
              contains(control_json, "\"text\":\"x\\u0000<y\\u0007\""),
          "JSON preserves exact original control characters via escaped raw and text fields");
    check(!controls.pages[0].blocks[0].warnings.empty() && !controls.pages[0].blocks[1].warnings.empty(),
          "Code and Equation control replacements disclosed in block diagnostics");
}

void test_margins_and_continuation() {
    document d;
    d.backend = "pdfium-heuristic";
    for (int i = 0; i < 2; ++i) {
        page p;
        p.index = i;
        p.width = 600;
        p.height = 800;
        p.blocks.push_back(text_block("Running report", "Text", { 40, 20, 250, 32 }));
        auto body = text_block(i ? "continued on the next page." : "A paragraph that is incom-", "Text",
                               i ? box{ 40, 100, 500, 130 } : box{ 40, 650, 500, 680 });
        body.lines = { text_line(i ? "continued on the" : "A paragraph that is", 40, i ? 100 : 650, 460),
                       text_line(i ? "next page." : "incom-", 40, i ? 118 : 668, 460) };
        p.blocks.push_back(body);
        p.blocks.push_back(text_block(std::to_string(i + 1), "Text", { 290, 770, 310, 782 }));
        d.pages.push_back(p);
    }
    options opts;
    opts.mode = 1;
    process_document(d, opts);
    check(d.pages[0].blocks.front().type == "PageHeader" && d.pages[0].blocks.front().ignored,
          "repeated header identified and suppressed");
    check(d.pages[0].blocks.back().type == "PageFooter" && d.pages[0].blocks.back().ignored,
          "page footer number suppressed");
    check(d.pages[0].blocks[1].continuation, "justified cross-page paragraph continuation");
    const std::string md = render_document(d, "markdown");
    check(!contains(md, "Running report") && contains(md, "incomcontinued"),
          "continued paragraphs join without page headers");
    const std::string json = render_document(d, "json");
    check(contains(json, "\"text\":\"Running report\"") && contains(json, "\"ignored\":true"),
          "suppressed marginalia retained in JSON");
    opts.include_page_headers = true;
    process_document(d, opts);
    check(contains(render_document(d, "markdown"), "Running report"), "header retention option honored");
    d.pages[0].blocks[1].token_cap_reached = true;
    d.pages[0].blocks[1].generated_tokens = 23;
    process_document(d, opts);
    check(contains(render_metadata(d), "\"capped_blocks\":1") &&
              contains(render_metadata(d), "\"generated_tokens\":23"),
          "metadata exposes incomplete generation counts");
}

} // namespace

static int crispembed_test_main() {
    try {
        test_layout();
        test_tables();
        test_text_quality();
        test_layout_heuristics();
        test_processors_and_renderers();
        test_margins_and_continuation();
    } catch (const std::exception & error) {
        ++failures;
        std::fprintf(stderr, "UNEXPECTED: %s\n", error.what());
    }
    std::printf("marker document: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
