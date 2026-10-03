#pragma once
#include <string>

namespace core_utf8 {
// A single byte-BPE token can end in a partial UTF-8 character. Match the
// tokenizer's replacement decoding before printing it in UTF-8 JSON.
inline std::string token_utf8_lossy(const char * bytes, int size) {
    std::string out;
    for (int i = 0; i < size;) {
        const unsigned char c = (unsigned char)bytes[i];
        if (c < 0x80) {
            out += bytes[i++];
            continue;
        }
        const int n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        int j = 1;
        for (; n && j < n && i + j < size; ++j) {
            const unsigned char b = (unsigned char)bytes[i + j];
            if (b < 0x80 || b > 0xbf ||
                (j == 1 && ((c == 0xe0 && b < 0xa0) || (c == 0xed && b > 0x9f) || (c == 0xf0 && b < 0x90) ||
                            (c == 0xf4 && b > 0x8f))))
                break;
        }
        if (n && j == n) {
            out.append(bytes + i, n);
            i += n;
        } else {
            out += "\xef\xbf\xbd";
            i += j;
        }
    }
    return out;
}

} // namespace core_utf8
