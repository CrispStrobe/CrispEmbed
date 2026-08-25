// core/punct_marks.h — "does this text already end in a sentence mark?"
//
// One implementation, because this predicate has now been written three times
// in two repos and the copies disagreed. CrispASR's crisp_punc carried it as an
// inlined test (issue #300, "never stack punctuation on punctuation");
// CrispEmbed's fireredpunc.cpp never got it and shipped a live defect where any
// already-punctuated input collected a second mark:
//
//   "ask not what your country can do for you."  ->  "...for you.."
//   "是的，我们需要更多时间。"                       ->  "...更多时间。。"
//   "does this already end in a question mark?"   ->  "...question mark??"
//
// For CrispASR that is an edge case (it punctuates its own ASR output, which
// arrives bare). For CrispEmbed it is the COMMON case: `--punct-model` is a
// post-processor over the user's OCR text, which usually IS already punctuated.
//
// ⚠ CROSS-REPO MIRROR. crisp_punc/src/fireredpunc.cpp lives in CrispASR but
// compiles against the CONSUMER's src/core, so this header must exist in both
// trees under the same name. Keep the copies in sync; the guard is
// tests/test_punct_marks.cpp (CrispEmbed) plus the sibling-crispasr CI job.
#pragma once

#include <string>

namespace core_punct {

// True when `s` ends in a sentence-final or clause mark, ASCII or CJK
// full-width. Byte-wise on UTF-8, so no locale or ICU dependency.
//
// Deliberately NOT included: the closing quotes and brackets that can follow a
// mark ("...for you." ), and the ellipsis. A closing quote means the mark is
// inside a quotation and the sentence around it may still want its own; adding
// them here would suppress correct punctuation, which is the worse failure.
inline bool ends_in_mark(const std::string & s) {
    if (s.empty()) return false;
    const char last = s[s.size() - 1];
    if (last == '.' || last == ',' || last == '?' || last == '!' || last == ';' || last == ':') return true;
    if (s.size() >= 3) {
        const unsigned char b0 = (unsigned char)s[s.size() - 3];
        const unsigned char b1 = (unsigned char)s[s.size() - 2];
        const unsigned char b2 = (unsigned char)s[s.size() - 1];
        // ，U+FF0C EF BC 8C · ？U+FF1F EF BC 9F · ！U+FF01 EF BC 81 · 。U+3002 E3 80 82
        if (b0 == 0xEF && b1 == 0xBC && (b2 == 0x8C || b2 == 0x9F || b2 == 0x81)) return true;
        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) return true;
    }
    return false;
}

} // namespace core_punct
