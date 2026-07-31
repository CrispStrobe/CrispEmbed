#include "easyocr_layout.h"

#include <cstdio>
#include <utility>

int main() {
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
    std::printf("easyocr-layout PASS words=%zu normalized=%d,%d,%d,%d\n", words.size(), boxes[0].x0, boxes[0].y0,
                boxes[0].x1, boxes[0].y1);
    return 0;
}
