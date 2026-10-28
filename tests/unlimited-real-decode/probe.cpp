// Test-only replay of retained IDs through the actual production text decoder.
#include "core/clean_exit.h"
#include "unlimited_ocr2_text.h"
#include "fixture.h"
#include <fstream>
#include <string>
#include <vector>
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    using namespace real_decode_fixture;
    std::vector<std::string> base(128000);
    for (const auto & piece : pieces) {
        if (piece.id < 0 || piece.id >= 128000) return 3;
        base[size_t(piece.id)] = piece.value;
    }
    for (const auto & piece : added)
        if (uocr2_text::added_token(piece.id) != piece.value) return 4;
    for (const auto & row : cases) {
        const auto raw = uocr2_text::decode_raw(base, row.ids, row.count, 1);
        const auto clean = uocr2_text::clean(raw);
        const std::string prefix = std::string(argv[1]) + "/" + row.key;
        std::ofstream(prefix + ".raw.txt", std::ios::binary).write(raw.data(), raw.size());
        std::ofstream(prefix + ".clean.txt", std::ios::binary).write(clean.data(), clean.size());
    }
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
