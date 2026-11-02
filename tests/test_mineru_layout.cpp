// Weight-free MinerU parser/geometry/OTSL/serialization regressions.
// Expected behavior is from the pinned official mineru-vl-utils blueprint.
#include "mineru_layout.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <stdexcept>

static int checks = 0, failures = 0;
static void expect(bool ok, const char * what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

static std::string layout(const char * coords, const char * type, const char * suffix = "<|rotate_up|>") {
    return std::string("<|box_start|>") + coords + "<|box_end|><|ref_start|>" + type + "<|ref_end|>" + suffix;
}

static int crispembed_test_main() {
    using namespace mineru;
    const auto p =
        parse_layout(layout("900 800 100 200", "TEXT", "<|rotate_right|><|txt_contd|>txt_contd_tgt") +
                     layout("0 0 100 100", "unknown") + layout("0 0 100 100", "inline_formula") +
                     layout("1001 0 500 900", "text") + layout("4 0 4 900", "text") +
                     layout("999999999999999999999999 0 5 9", "text") + layout("0 0 10 10", "imaginary") +
                     layout("0 0 100 100", "header", "") + layout("-1 0 10 10", "text") + layout("0 0 10 10 ", "text"));
    expect(p.size() == 3, "invalid/inline blocks skipped, unknown becomes image");
    if (p.size() == 3) {
        expect(p[0].type == "text" && p[0].bbox == std::array<double, 4>{ { .1, .2, .9, .8 } },
               "lowercase type and reversed boxes normalize");
        expect(p[0].angle == 90 && p[0].merge_prev, "rotation and continuation parsed");
        expect(p[1].type == "image" && p[1].angle == 0, "unknown is image");
        expect(p[2].angle == -1, "missing rotation preserves null");
    }
    const auto merged =
        parse_layout(layout("0 0 1000 1000", "text") + layout("1 1 999 999", "text") +
                     layout("0 0 1000 1000", "text", "<|rotate_down|>") + layout("0 0 1000 1000", "title"));
    expect(merged.size() == 3, "IoU dedup groups by type/angle and keeps first order");
    const auto threshold = parse_layout(layout("0 0 1000 1000", "text") + layout("0 0 900 1000", "text"));
    expect(threshold.size() == 2, "IoU equal 0.9 is not suppressed");
    const auto internal = parse_layout(
        layout("0 0 1000 1000", "table") + layout("100 100 900 900", "text") + layout("100 100 200 200", "equation") +
        layout("100 100 900 900", "equation_block") + layout("100 100 900 900", "table_caption"));
    expect(internal.size() == 2 && internal[1].type == "table_caption", "table internal text/equation suppressed");
    auto captions = parse_layout(layout("0 0 1000 1000", "image_block") + layout("100 100 500 500", "image") +
                                 layout("100 100 200 200", "image_caption"));
    filter_internal_captions(captions);
    expect(captions.size() == 2 && !should_extract(captions, 1, true),
           "internal caption removed, contained visual skipped");
    block b;
    b.type = "text";
    b.bbox = { { .1, .3, .5, .9 } };
    const auto crop = crop_geometry(b, 5, 5);
    expect(crop.x0 == 0 && crop.y0 == 2 && crop.x1 == 2 && crop.y1 == 4, "PIL half ties crop to even");
    expect(round_even(-.5) == 0 && round_even(-1.5) == -2 && round_even(2.5) == 2 && round_even(3.5) == 4,
           "rounding independent of ties-away C++ round");
    bool rejected = false;
    try {
        b.bbox = { { 0, 0, .001, .001 } };
        crop_geometry(b, 10, 10);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    expect(rejected, "zero-pixel crops rejected");
    const auto t = normalize_crop_geometry(1000, 2, 90);
    expect(t.rotated_width == 2 && t.rotated_height == 1000 && t.padded_width == 20 && t.padded_height == 1000 &&
               t.paste_x == 9 && t.paste_y == 0 && t.output_width == 28 && t.output_height == 1400,
           "rotation then centered white padding then minimum-edge upscale");
    expect(std::string(prompt_for_type("table")) == "\nTable Recognition:" &&
               std::string(prompt_for_type("equation")) == "\nFormula Recognition:" &&
               std::string(prompt_for_type("[layout]")) == "\nLayout Detection:" &&
               std::string(prompt_for_type("title")) == "\nText Recognition:",
           "exact task prompts");
    b.type = "chart";
    b.bbox = { { 0, 0, .5, .5 } };
    expect(!should_extract(b) && should_extract(b, true), "image analysis opt-in");
    b.bbox = { { 0, 0, .1, .09 } };
    expect(!should_extract(b, true), "small visual omitted by source eligibility threshold");
    expect(otsl_to_html("<fcel>A&\"<fcel>B<nl><fcel>C<ecel><nl>") ==
               "<table><tr><td>A&amp;&quot;</td><td>B</td></tr><tr><td>C</td><td></td></tr></table>",
           "OTSL escapes HTML");
    expect(otsl_to_html("<fcel>A<lcel><nl><ucel><xcel><nl>") ==
               "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td></tr><tr></tr></table>",
           "OTSL two-dimensional span");
    expect(otsl_to_html("<fcel>A<nl><fcel>B<fcel>C<nl>") ==
               "<table><tr><td>A</td><td></td></tr><tr><td>B</td><td>C</td></tr></table>",
           "OTSL ragged rows padded");
    expect(otsl_to_html("<table><tr><td>known</td></tr></table>") == "<table><tr><td>known</td></tr></table>",
           "existing HTML unchanged");
    expect(otsl_to_html("not a table").empty(), "non-OTSL has no inferred table");
    expect(otsl_to_html("<lcel><nl>").empty(), "all span markers have no origin cell");
    expect(otsl_to_html("<ecel>ignore me<nl>") == "<table><tr><td></td></tr></table>", "ecel does not consume text");
    expect(otsl_to_html("<fcel>\xe3\x80\x80"
                        "A\xc2\xa0<nl>") == "<table><tr><td>A</td></tr></table>",
           "OTSL trims Python Unicode whitespace");
    expect(json_quote("\"\\\n\t\x01") == "\"\\\"\\\\\\n\\t\\u0001\"", "JSON escaping");
    b.type = "text";
    b.bbox = { { 0, 0, 1, 1 } };
    b.angle = -1;
    b.has_content = true;
    b.content = "First";
    block title = b;
    title.type = "title";
    title.content = "Heading";
    block continuation = b;
    continuation.merge_prev = true;
    continuation.content = "second";
    block cjk = continuation;
    cjk.content = "中文";
    expect(blocks_markdown({ b, title, continuation, cjk }) == "First second中文\n\nHeading",
           "continuation merges into last text, CJK has no inserted space");
    expect(blocks_json({ b }) ==
               "[{\"type\":\"text\",\"bbox\":[0,0,1,1],\"angle\":null,\"content\":\"First\",\"merge_prev\":false}]",
           "block JSON preserves null angle");
    block placeholder = b;
    placeholder.content = "[Non-Text]";
    block header = placeholder;
    header.type = "header";
    block list = b;
    list.type = "list_item";
    std::vector<block> basic = { placeholder, header, list };
    post_process_basic(basic);
    expect(basic.size() == 2 && basic[0].content.empty() && basic[1].type == "text",
           "basic postprocessing source placeholders/list items");
    std::printf("MinerU layout: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
