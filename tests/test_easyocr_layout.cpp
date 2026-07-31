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
    std::printf("easyocr-layout PASS words=%zu normalized=%d,%d,%d,%d\n", words.size(), boxes[0].x0, boxes[0].y0,
                boxes[0].x1, boxes[0].y1);
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
