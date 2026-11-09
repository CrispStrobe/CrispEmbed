#include "lighton4b_head_bf16.h"
#include "core/clean_exit.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LIGHTON4B_HEAD_BF16", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_LIGHTON4B_HEAD_BF16", value, 1);
    else
        unsetenv("CRISPEMBED_LIGHTON4B_HEAD_BF16");
#endif
}
static uint32_t bits(float value) {
    uint32_t b;
    std::memcpy(&b, &value, 4);
    return b;
}
static float value(uint32_t b) {
    float x;
    std::memcpy(&x, &b, 4);
    return x;
}
static int run() {
    using namespace qwen35_ocr;
    const std::vector<float> raw = { 18.55112075805664f, 18.55838394165039f };
    for (const char * flag : { static_cast<const char *>(nullptr), "", "0" }) {
        gate(flag);
        auto x = raw;
        lighton4b_head_bf16(x);
        if (x != raw || lighton4b_greedy_id(x) != 1) return 1;
    }
    gate("1");
    auto x = raw;
    lighton4b_head_bf16(x);
    if (x != std::vector<float>({ 18.5f, 18.5f }) || lighton4b_greedy_id(x) != 0) return 2;
    // Source row33 IDs401/1778 and full-vocabulary first-index tie behavior.
    x.assign(248320, -100.0f);
    x[401] = raw[0];
    x[1778] = raw[1];
    lighton4b_head_bf16(x);
    if (lighton4b_greedy_id(x) != 401) return 3;
    // Explicit IEEE round-to-nearest-even, signed zero, subnormal and overflow.
    for (auto pair : { std::pair<uint32_t, uint32_t>{ 0x3f808000, 0x3f800000 },
                       { 0x3f818000, 0x3f820000 },
                       { 0xbf808000, 0xbf800000 },
                       { 0xbf818000, 0xbf820000 },
                       { 0x00008000, 0x00000000 },
                       { 0x00018000, 0x00020000 },
                       { 0x80000000, 0x80000000 },
                       { 0x7f7fffff, 0x7f800000 },
                       { 0x7f800000, 0x7f800000 },
                       { 0xff800000, 0xff800000 },
                       { 0x7f800001, 0x7fc00000 },
                       { 0xff800001, 0xffc00000 } }) {
        x = { value(pair.first) };
        lighton4b_head_bf16(x);
        if (bits(x[0]) != pair.second) return 4;
    }
    const float inf = std::numeric_limits<float>::infinity();
    if (lighton4b_greedy_id({ -inf, inf, inf }) != 1 || lighton4b_greedy_id({ -inf, -inf }) != 0 ||
        lighton4b_greedy_id({ 0, std::numeric_limits<float>::quiet_NaN() }) != -1 || lighton4b_greedy_id({}) != -1)
        return 5;
    std::puts("HEAD_BF16 passed=1 source_row33_tie=1 rne_specials=12 gate_controls=3 lowest_id=1");
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
