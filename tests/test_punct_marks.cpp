// Unit test for core_punct::ends_in_mark — the "never stack punctuation on
// punctuation" predicate (#300).
//
// Host-only: no model, no ggml. The defect it guards shipped in CrispEmbed's
// fireredpunc.cpp, where already-punctuated input collected a second mark
// ("...for you." -> "...for you.."). CrispASR's crisp_punc had the fix inlined
// and CrispEmbed's copy never got it, so the predicate now lives in one header
// both trees mirror — and the property it must satisfy lives here rather than
// in either caller.
//
// The checks are written as the two directions that can break independently:
// a mark that is NOT recognised re-opens the stacking bug, and a non-mark that
// IS recognised silently suppresses correct punctuation. Both are represented.

#include "core/clean_exit.h"
#include "core/punct_marks.h"

#include <cstdio>
#include <string>

using core_punct::ends_in_mark;

static int g_failures = 0;

static void check(const char * name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_failures++;
}

static int crispembed_test_main() {
    std::printf("test_punct_marks\n");

    // --- ASCII marks: every one the restorer can emit must be recognised, or
    // the stacking bug returns for that mark specifically.
    check("ascii period", ends_in_mark("ask not what your country can do for you."));
    check("ascii comma", ends_in_mark("first clause,"));
    check("ascii question", ends_in_mark("does this already end in a question mark?"));
    check("ascii exclamation", ends_in_mark("hello world!"));
    check("ascii semicolon", ends_in_mark("one thing;"));
    check("ascii colon", ends_in_mark("as follows:"));

    // --- CJK full-width marks. These are three UTF-8 bytes, so a byte-wise
    // check that only looked at the last byte would miss them entirely — and
    // the Chinese vocab this model serves is its PRIMARY input language.
    check("fullwidth ideographic full stop 。", ends_in_mark("是的，我们需要更多时间。"));
    check("fullwidth comma ，", ends_in_mark("是的，"));
    check("fullwidth question ？", ends_in_mark("你好吗？"));
    check("fullwidth exclamation ！", ends_in_mark("太好了！"));

    // --- Negatives. A false positive here suppresses a mark the model
    // correctly predicted, which is the worse of the two failure modes.
    check("bare word is not a mark", !ends_in_mark("hello world"));
    check("empty string is not a mark", !ends_in_mark(""));
    check("bare CJK is not a mark", !ends_in_mark("是的我们需要更多时间"));
    check("digit is not a mark", !ends_in_mark("in 1999"));
    check("hyphen is not a mark", !ends_in_mark("well-known-"));
    check("closing quote after a mark is deliberately NOT a mark", !ends_in_mark("he said \"stop.\""));

    // --- Byte-safety: a multi-byte character whose TRAILING byte happens to
    // equal a full-width mark's trailing byte must not be mistaken for one.
    // 。is E3 80 82; 三 is E4 B8 89 — different lead, must not match.
    check("other CJK char sharing no prefix", !ends_in_mark("三"));
    // A 2-byte sequence is too short for the 3-byte compare to read past the
    // start of the string; this must not crash or false-positive.
    check("two-byte char is safe", !ends_in_mark("é"));
    check("single byte is safe", !ends_in_mark("a"));

    // --- The property the caller actually depends on: appending is a no-op
    // exactly when the text already ends in a mark, so applying the guard
    // twice is idempotent.
    {
        std::string s = "already done.";
        const std::string before = s;
        if (!ends_in_mark(s)) s += ".";
        check("guarded append is a no-op on punctuated text", s == before);
        std::string t = "not yet done";
        if (!ends_in_mark(t)) t += ".";
        check("guarded append still punctuates bare text", t == "not yet done.");
    }

    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
