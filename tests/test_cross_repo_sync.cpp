// test_cross_repo_sync.cpp — the files CrispEmbed mirrors from CrispASR must
// not drift.
//
// CrispASR has tests/test-copies-in-sync.cpp for its own two copies of
// fireredpunc/pcs/lid/truecase. It is the single most effective test in either
// tree: it caught a #308 revert the moment one copy was updated without the
// other. There was no CROSS-repo equivalent, and issue #50 is what that costs.
//
// Four distinct bugs in one investigation, all the same seam:
//   1. std::map vs std::unordered_map in audio_tower (fixed upstream earlier)
//   2. core/ggml_cpu_backend.h existed in CrispASR's core and not CrispEmbed's
//   3. fireredpunc_debug_token_ids existed in one copy only
//   4. crisp_audio linked crispasr_link_ggml_cuda, a target only CrispASR
//      defines — invisible until a CUDA build on a Kaggle P100
// plus a silent behaviour swap: the two fireredpunc copies had diverged so
// badly that merely HAVING a sibling CrispASR changed the punctuation output.
//
// WHAT THIS COMPARES, and why not bytes. Each repo clang-formats to its own
// style (`T * p` here, `T* p` there) and each copy carries repo-specific prose
// — the mirrored headers say different things about which half they are. So
// bytes cannot match and should not. Comments and whitespace are stripped and
// the CODE is compared. That permits formatting and prose to differ while any
// semantic drift fails.
//
// The stripper is literal-aware: "https://..." inside a string is not a
// comment, and a '"' inside a comment does not open a string. Getting that
// backwards would either mangle every URL or silently compare nothing.
//
// SKIPS CLEANLY when no sibling CrispASR is present, so a standalone checkout
// is unaffected. The CI coverage is the sibling-crispasr job.

#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(const std::string & name, bool ok, const std::string & detail = "") {
    g_checks++;
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) {
        g_failures++;
        if (!detail.empty()) std::printf("        %s\n", detail.c_str());
    }
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Strip // and /* */ comments and ALL whitespace, tracking string and char
// literals so a "//" inside a literal survives and a quote inside a comment
// does not open one. Raw string literals are not used in either file; if that
// changes this needs to learn R"(...)".
std::string code_only(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    enum { CODE, SLASH_SLASH, SLASH_STAR, STR, CHR } st = CODE;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        const char n = (i + 1 < s.size()) ? s[i + 1] : '\0';
        switch (st) {
        case CODE:
            if (c == '/' && n == '/') {
                st = SLASH_SLASH;
                i++;
            } else if (c == '/' && n == '*') {
                st = SLASH_STAR;
                i++;
            } else if (c == '"') {
                st = STR;
                out += c;
            } else if (c == '\'') {
                st = CHR;
                out += c;
            } else if (!std::isspace((unsigned char)c)) {
                out += c;
            }
            break;
        case SLASH_SLASH:
            if (c == '\n') st = CODE;
            break;
        case SLASH_STAR:
            if (c == '*' && n == '/') {
                st = CODE;
                i++;
            }
            break;
        case STR:
            out += c;
            if (c == '\\' && i + 1 < s.size()) {
                out += n;
                i++;
            } else if (c == '"') {
                st = CODE;
            }
            break;
        case CHR:
            out += c;
            if (c == '\\' && i + 1 < s.size()) {
                out += n;
                i++;
            } else if (c == '\'') {
                st = CODE;
            }
            break;
        }
    }
    return out;
}

struct Mirror {
    const char * ours;   // relative to CrispEmbed root
    const char * theirs; // relative to CrispASR root
};

// Enumerated, not globbed: a new mirrored file should be a deliberate addition
// here rather than something a glob quietly starts or stops covering.
const Mirror kMirrors[] = {
    { "src/core/ggml_cpu_backend.h", "src/core/ggml_cpu_backend.h" },
    { "src/core/punct_marks.h", "src/core/punct_marks.h" },
    { "src/core/env_gate.h", "src/core/env_gate.h" },
    { "src/core/bert_norm.h", "src/core/bert_norm.h" },
    { "src/core/bert_pretok.h", "src/core/bert_pretok.h" },
    { "src/core/unicode_bert_norm.h", "src/core/unicode_bert_norm.h" },
    { "src/core/unicode_categ.h", "src/core/unicode_categ.h" },
};

// fireredpunc is deliberately NOT in kMirrors. Its two copies have sanctioned
// CODE differences — CrispEmbed installs its imatrix collector, CrispASR routes
// the CPU backend through core_cpu_backend for GGML_BACKEND_DL — so a
// whole-file compare would be a list of exceptions long enough to hide the next
// real drift. Assert the properties instead, the way CrispASR's own test does
// for #308: each one is a fix that was, at some point, present in exactly one
// copy.
struct Invariant {
    const char * needle;
    const char * what;
};

const Invariant kPuncInvariants[] = {
    // ⚠ Needles must be SPECIFIC, not merely present. `if (c >= 'A' && c <= 'Z')`
    // alone also matches the legacy tokenizer's lowercasing loop
    // (`... c = c - 'A' + 'a';`), so this check passed with #308 reverted until
    // the body was included. Both drafts of this test were inert for two
    // different reasons; neither would have been noticed without deliberately
    // breaking the code and watching the guard fail.
    { "if (c >= 'A' && c <= 'Z') { cap_next = false; }", "#308: an uppercase letter disarms cap_next" },
    { "core_punct::ends_in_mark", "#300: never stack punctuation on punctuation" },
    { "core_bert::pretokenize", "HF WordPiece pre-tokenizer" },
    { "core_bert::lower_strip_accents", "HF normalizer" },
    { "have_word_alignment", "SentencePiece models must not fall into the empty-output path" },
    { "append_sep", "the [SEP] gate exists (blueprint appends none for BERT)" },
    // `is_sentencepiece` on its own appears all over this file and proves
    // nothing about the scoping; the disjunction is what does.
    { "force_sep || ctx.tokenizer.is_sentencepiece", "the [SEP] removal stays scoped to the BERT path" },
};

} // namespace

static int crispembed_test_main() {
    std::printf("test_cross_repo_sync\n");

    const char * embed_root = std::getenv("CRISPEMBED_SOURCE_DIR");
    const char * asr_root = std::getenv("CRISPASR_SIBLING_DIR");
    if (!embed_root || !asr_root) {
        std::printf("SKIP: set CRISPEMBED_SOURCE_DIR and CRISPASR_SIBLING_DIR\n");
        return 0;
    }
    std::string probe;
    if (!read_file(std::string(asr_root) + "/src/core/gguf_loader.h", probe)) {
        std::printf("SKIP: no CrispASR checkout at %s\n", asr_root);
        return 0;
    }

    for (const auto & m : kMirrors) {
        const std::string a = std::string(embed_root) + "/" + m.ours;
        const std::string b = std::string(asr_root) + "/" + m.theirs;
        std::string sa, sb;
        const bool ha = read_file(a, sa), hb = read_file(b, sb);
        if (!ha || !hb) {
            // A MISSING mirror is the exact shape of issue #50's second
            // instance, so it is a failure and not a skip.
            check(std::string(m.ours) + " exists on both sides", false,
                  std::string(ha ? "" : "missing: " + a) + (hb ? "" : " missing: " + b));
            continue;
        }
        const std::string ca = code_only(sa), cb = code_only(sb);
        std::string detail;
        if (ca != cb) {
            size_t at = 0;
            while (at < ca.size() && at < cb.size() && ca[at] == cb[at]) at++;
            detail = "first difference at stripped offset " + std::to_string(at) + "\n        ours:   ..." +
                     ca.substr(at > 40 ? at - 40 : 0, 100) + "\n        theirs: ..." +
                     cb.substr(at > 40 ? at - 40 : 0, 100);
        }
        check(std::string(m.ours) + " code matches CrispASR's copy", ca == cb, detail);
    }

    // fireredpunc: both copies must carry every fix, whichever repo it came from.
    const std::string punc_ours = std::string(embed_root) + "/src/fireredpunc.cpp";
    const std::string punc_theirs = std::string(asr_root) + "/crisp_punc/src/fireredpunc.cpp";
    for (const std::string & p : { punc_ours, punc_theirs }) {
        std::string src;
        if (!read_file(p, src)) {
            check("fireredpunc readable: " + p, false);
            continue;
        }
        // Search the STRIPPED source, not the raw file. A marker mentioned in
        // a comment would otherwise satisfy the check — which is not
        // hypothetical: the first draft of this test passed while #308 had been
        // reverted in the code, because a comment near the fix quoted the very
        // line the check looks for. Stripping comments makes prose unable to
        // vouch for code. Needles are stripped too, so they stay readable here.
        const std::string code = code_only(src);
        for (const auto & inv : kPuncInvariants) {
            check(std::string(inv.what) + "  [" + (p == punc_ours ? "CrispEmbed" : "CrispASR") + "]",
                  code.find(code_only(inv.needle)) != std::string::npos, std::string("missing marker: ") + inv.needle);
        }
    }

    std::printf("%s — %d checks, %d failure%s\n", g_failures ? "FAILED" : "OK", g_checks, g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
