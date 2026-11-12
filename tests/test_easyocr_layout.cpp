#include "easyocr_layout.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <utility>

static int crispembed_test_main() {
    std::vector<easyocr_layout::word> words = {
        { "world", 100, 20, 80, 20, .8f, 0, 1, 1 },
        { "hello", 10, 20, 80, 20, .9f, 0, 1, 0 },
        { "next", 10, 60, 80, 20, .7f, 0, 2, 0 },
    };
    words = easyocr_layout::reading_order(std::move(words));
    if (words[0].text != "hello" || words[1].text != "world" || words[2].text != "next") return 1;
    auto boxes = easyocr_layout::normalize_boxes(words, 200, 100);
    if (boxes[0].x0 != 50 || boxes[0].y0 != 200 || boxes[0].x1 != 450 || boxes[0].y1 != 400) return 2;
    if (easyocr_layout::join_lines(words) != "hello world\nnext") return 3;
    const std::vector<easyocr_layout::region> regions = {
        { 100, 20, 30, 12, .8f },
        { 10, 21, 40, 12, .9f },
        { 15, 70, 50, 12, .7f },
    };
    const auto lines = easyocr_layout::group_lines(regions);
    if (lines.size() != 2 || lines[0].x != 10 || lines[0].w != 120 || lines[0].line != 0 || lines[1].line != 1)
        return 4;
    const auto ordered = easyocr_layout::order_words(regions);
    if (ordered.size() != 3 || ordered[0].x != 10 || ordered[1].x != 100 || ordered[2].line != 1) return 5;
    if (easyocr_layout::order_regions(regions, easyocr_layout::ordering_mode::lines).size() != 2) return 6;
    if (easyocr_layout::order_regions(regions, easyocr_layout::ordering_mode::words).size() != 3) return 7;
    if (easyocr_layout::group_dbnet_lines(regions).size() != 2) return 8;
    // Geometry-only reproduction of the retained Android fixture: a spanning
    // heading and four same-baseline pairs of body lines. These are synthetic
    // detector fragments, not a claim to have recovered the device's raw boxes.
    std::vector<easyocr_layout::region> columns = {
        { 300, 60, 260, 72, .9f },
        { 580, 60, 260, 72, .9f },
        { 860, 60, 260, 72, .9f },
        { 1140, 60, 340, 72, .9f },
    };
    for (int row = 0; row < 4; ++row) {
        const float y = 250.0f + 150.0f * row;
        columns.push_back({ 80, y, 140, 50, .9f });
        columns.push_back({ 260, y, 140, 50, .9f });
        columns.push_back({ 440, y, 140, 50, .9f });
        columns.push_back({ 1040, y, 220, 50, .9f });
        columns.push_back({ 1300, y, 220, 50, .9f });
    }
    const auto legacy = easyocr_layout::group_dbnet_lines(columns);
    const auto split = easyocr_layout::group_dbnet_lines(columns, 2.5f);
    if (legacy.size() != 5 || split.size() != 9 || split[0].x != 300 || split[0].w != 1180) return 9;
    for (int row = 0; row < 4; ++row) {
        const auto & left = split[1 + row * 2];
        const auto & right = split[2 + row * 2];
        if (left.x != 80 || left.w != 500 || right.x != 1040 || right.w != 480 || left.y != right.y) return 10;
    }
    // Connected/overlapping fragments and ordinary word spacing stay together;
    // the explicit-off alternative retains exactly the historical body width.
    const std::vector<easyocr_layout::region> connected = {
        { 10, 20, 100, 40, .8f },
        { 90, 20, 70, 40, .8f },
        { 260, 20, 60, 40, .8f },
    };
    const auto joined = easyocr_layout::group_dbnet_lines(connected, 2.5f);
    if (joined.size() != 1 || joined[0].x != 10 || joined[0].w != 310) return 11;
    if (easyocr_layout::group_dbnet_lines(columns, 0.0f)[1].w != legacy[1].w || legacy[1].w != 1440) return 12;
    if (easyocr_layout::group_dbnet_lines({}, 2.5f).size() != 0) return 13;
    std::printf("dbnet-gutters synthetic_fragments=%zu legacy_crops=%zu candidate_crops=%zu legacy_body_width=%.0f "
                "candidate_left_width=%.0f candidate_right_width=%.0f\n",
                columns.size(), legacy.size(), split.size(), legacy[1].w, split[1].w, split[2].w);
    std::printf("easyocr-layout PASS words=%zu normalized=%d,%d,%d,%d\n", words.size(), boxes[0].x0, boxes[0].y0,
                boxes[0].x1, boxes[0].y1);
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
