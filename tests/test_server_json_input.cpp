// Unit test for the HTTP server's JSON string-input parser (issue #34).
//
// Host-only: no model, no ggml — exercises src/core/json.h directly.
// Verifies that valid escaped JSON payloads (values containing ']', \" and \\,
// newlines, and \uXXXX) parse to the correct input cardinality, rather than the
// old delimiter-scan approach that produced "returned 7 embeddings for 6 inputs".
//
// Returns non-zero exit code on any failure (matches the repo's other test-* runners).

#include "core/clean_exit.h"
#include "core/json.h"

#include <cstdio>
#include <string>
#include <vector>

using core_json::json_decode_string;
using core_json::json_escape_legacy;
using core_json::json_escape_strict;
using core_json::json_extract_strings;
using core_json::json_extract_strings_escaped;
using core_json::json_extract_strings_legacy;
using core_json::json_find_key_value;

static int g_failures = 0;

static void check(const char * name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_failures++;
}

// Build a std::string from a literal without raw-string edge cases.
static std::string S(const char * s) {
    return std::string(s);
}

static int crispembed_test_main() {
    std::printf("test_server_json_input\n");

    // Issue #34 primary reproduction: 6 inputs with ], escaped quotes, backslash.
    {
        std::string body = S("{\"model\":\"m\",\"input\":["
                             "\"normal text\","
                             "\"text with ] bracket inside\","
                             "\"text with escaped quote: \\\"hello\\\"\","
                             "\"text with backslash: \\\\\","
                             "\"another normal text\","
                             "\"final item\""
                             "]}");
        std::vector<std::string> out;
        size_t n = json_extract_strings(body, "input", out);
        check("array count == 6", n == 6 && out.size() == 6);
        check("value[1] keeps ] bracket", out.size() > 1 && out[1] == "text with ] bracket inside");
        check("value[2] unescapes quotes", out.size() > 2 && out[2] == "text with escaped quote: \"hello\"");
        check("value[3] unescapes backslash", out.size() > 3 && out[3] == "text with backslash: \\");
        check("value[5] final item intact", out.size() > 5 && out[5] == "final item");
    }

    // Issue #34 second reproduction: 4 inputs incl. an escaped newline.
    {
        std::string body = S(
            "{\"input\":[\"plain text\",\"text with ] bracket\",\"text with \\\"quoted\\\" part\",\"line\\nbreak\"]}");
        std::vector<std::string> out;
        size_t n = json_extract_strings(body, "input", out);
        check("array count == 4", n == 4);
        check("value[3] decodes real newline", out.size() > 3 && out[3] == "line\nbreak");
    }

    // Single-string form.
    {
        std::string body = S("{\"input\":\"hello \\\"world\\\" ]\"}");
        std::vector<std::string> out;
        size_t n = json_extract_strings(body, "input", out);
        check("single count == 1", n == 1);
        check("single unescaped", out.size() == 1 && out[0] == "hello \"world\" ]");
    }

    // "prompt" key, single string with a backslash.
    {
        std::string body = S("{\"model\":\"m\",\"prompt\":\"a \\\\ b\"}");
        std::vector<std::string> out;
        json_extract_strings(body, "prompt", out);
        check("prompt value", out.size() == 1 && out[0] == "a \\ b");
    }

    // \uXXXX BMP code point (é = U+00E9 -> 0xC3 0xA9).
    {
        std::string body = S("{\"input\":[\"caf\\u00e9\"]}");
        std::vector<std::string> out;
        json_extract_strings(body, "input", out);
        check("unicode BMP", out.size() == 1 && out[0] == "caf\xc3\xa9");
    }

    // \uXXXX surrogate pair (U+1F600 -> F0 9F 98 80).
    {
        std::string body = S("{\"input\":[\"\\ud83d\\ude00\"]}");
        std::vector<std::string> out;
        json_extract_strings(body, "input", out);
        check("unicode surrogate pair", out.size() == 1 && out[0] == "\xf0\x9f\x98\x80");
    }

    // Missing key -> 0.
    {
        std::string body = S("{\"model\":\"m\"}");
        std::vector<std::string> out;
        check("missing key -> 0", json_extract_strings(body, "input", out) == 0);
    }

    // Empty array -> 0.
    {
        std::string body = S("{\"input\":[]}");
        std::vector<std::string> out;
        check("empty array -> 0", json_extract_strings(body, "input", out) == 0);
    }

    // Regression: a ']' inside the FIRST element used to truncate the whole array.
    {
        std::string body = S("{\"input\":[\"a]a\",\"b\",\"c\"]}");
        std::vector<std::string> out;
        check("bracket-in-first still yields 3", json_extract_strings(body, "input", out) == 3);
    }

    // Whitespace between key/colon/value.
    {
        std::string body = S("{ \"input\" : [ \"x\" , \"y\" ] }");
        std::vector<std::string> out;
        check("whitespace tolerated", json_extract_strings(body, "input", out) == 2);
    }

    // -----------------------------------------------------------------
    // A/B: legacy (CRISPEMBED_SERVER_LEGACY_JSON=1) vs escaped parser.
    // Contract: IDENTICAL on payloads with no escapes/brackets (so the gate is
    // output-neutral for normal traffic), and differing ONLY where the legacy
    // scan was wrong. This is what makes the gate a safe bisection switch.
    // -----------------------------------------------------------------
    {
        const char * benign[] = {
            "{\"input\":[\"alpha\",\"beta\",\"gamma\"]}",
            "{\"texts\":[\"one\"]}",
            "{\"documents\":[\"doc a\",\"doc b\"]}",
            "{\"labels\":[\"person\",\"org\"]}",
            "{\"query\":\"find me\"}",
            "{\"prompt\":\"hello\"}",
            "{ \"input\" : [ \"x\" , \"y\" ] }",
            "{\"input\":[]}",
            "{\"model\":\"m\"}",
        };
        const char * keys[] = { "input", "texts", "documents", "labels", "query", "prompt", "input", "input", "input" };
        bool all_same = true;
        for (size_t i = 0; i < sizeof(benign) / sizeof(benign[0]); i++) {
            std::vector<std::string> a, b;
            json_extract_strings_legacy(S(benign[i]), keys[i], a);
            json_extract_strings_escaped(S(benign[i]), keys[i], b);
            if (a != b) {
                all_same = false;
                std::printf("      A/B differs on benign payload: %s\n", benign[i]);
            }
        }
        check("A/B: legacy == escaped on all benign payloads", all_same);
    }
    {
        // The bug cases: legacy is wrong, escaped is right. Documents exactly what
        // flipping CRISPEMBED_SERVER_LEGACY_JSON=1 costs you.
        std::string bracket = S("{\"input\":[\"a]a\",\"b\",\"c\"]}");
        std::vector<std::string> la, ea;
        json_extract_strings_legacy(bracket, "input", la);
        json_extract_strings_escaped(bracket, "input", ea);
        // Legacy takes the ']' *inside* "a]a" as the array end, leaving the
        // fragment "a — which has no closing quote, so it drops every element.
        std::printf("      (legacy yields %zu, escaped yields %zu)\n", la.size(), ea.size());
        check("A/B: legacy is wrong on ]-in-string (drops all)", la.size() == 0);
        check("A/B: escaped keeps all 3", ea.size() == 3);

        std::string esc = S("{\"input\":[\"say \\\"hi\\\"\",\"b\"]}");
        std::vector<std::string> lb, eb;
        json_extract_strings_legacy(esc, "input", lb);
        json_extract_strings_escaped(esc, "input", eb);
        check("A/B: legacy mis-splits on \\\" (!=2)", lb.size() != 2);
        check("A/B: escaped yields 2", eb.size() == 2);
        check("A/B: escaped decodes the quote", eb.size() == 2 && eb[0] == "say \"hi\"");
    }
    {
        // Default (no env var set in this process) must be the escaped parser.
        std::vector<std::string> out;
        json_extract_strings(S("{\"input\":[\"a]a\",\"b\",\"c\"]}"), "input", out);
        check("gate defaults to escaped parser", out.size() == 3);
    }

    // =================================================================
    // B2: structural key location — a decoy "key" string appearing as a
    // VALUE must not be mistaken for the key (previously find() latched onto
    // it and returned another key's values, or values for an absent key).
    // =================================================================
    {
        // Key genuinely absent, but a decoy element equals the name.
        std::vector<std::string> out;
        size_t n = json_extract_strings(S("{\"labels\":[\"input\"],\"other\":[\"x\",\"y\"]}"), "input", out);
        check("decoy value not matched as key -> 0", n == 0 && out.empty());
    }
    {
        // Decoy BEFORE the real key: must still find the real one.
        std::vector<std::string> out;
        json_extract_strings(S("{\"labels\":[\"input\"],\"input\":[\"good\"]}"), "input", out);
        check("real key found past a decoy value", out.size() == 1 && out[0] == "good");
    }
    {
        // Key name appearing inside a nested object value must be ignored.
        std::vector<std::string> out;
        size_t n = json_extract_strings(S("{\"a\":{\"input\":[\"nested\"]},\"input\":[\"top\"]}"), "input", out);
        check("nested same-name key ignored (depth>1)", n == 1 && out[0] == "top");
    }
    {
        // A value string equal to the key name at depth 1 (as a plain value).
        std::vector<std::string> out;
        size_t n = json_extract_strings(S("{\"model\":\"input\",\"input\":[\"v\"]}"), "input", out);
        check("depth-1 value equal to key name ignored", n == 1 && out[0] == "v");
    }
    {
        // json_find_key_value returns npos when the key is truly absent.
        check("find_key_value: absent -> npos",
              json_find_key_value(S("{\"labels\":[\"input\"]}"), "input") == std::string::npos);
        check("find_key_value: present -> value index",
              json_find_key_value(S("{\"input\":\"x\"}"), "input") != std::string::npos);
    }

    // =================================================================
    // B1: output escaping — round-trip property + control-char coverage.
    // =================================================================
    {
        // The property that makes the escaper correct: for ANY byte string,
        // decode(escape(x)) == x. Covers every control char plus quotes/backslash.
        std::string all;
        for (int c = 0; c < 256; c++) all.push_back((char)c);
        // A few structured strings too (OCR-ish: tabs + newlines).
        const char * cases[] = {
            "plain", "a\tb\tc", "line1\r\nline2", "quote:\" back:\\ ", "\x01\x02\x1f control", "café \xf0\x9f\x98\x80",
            ""
        };
        bool roundtrip_ok = true;
        auto rt = [&](const std::string & x) {
            std::string wire = std::string("\"") + json_escape_strict(x) + "\"";
            std::string dec;
            size_t end = 0;
            if (!json_decode_string(wire, 0, dec, end) || dec != x) {
                roundtrip_ok = false;
                std::printf("      round-trip FAILED for %zu-byte input\n", x.size());
            }
        };
        rt(all);
        for (const char * c : cases) rt(S(c));
        check("B1 round-trip: decode(escape(x))==x for all 256 bytes + cases", roundtrip_ok);

        // Explicit control-char escapes (the actual bug: raw \t/\r were emitted).
        check("escape tab -> \\t", json_escape_strict(S("a\tb")) == "a\\tb");
        check("escape CR -> \\r", json_escape_strict(S("a\rb")) == "a\\rb");
        check("escape US(0x1f) -> \\u001f", json_escape_strict(S("a\x1f")) == "a\\u001f");
        check("escape quote+backslash", json_escape_strict(S("\"\\")) == "\\\"\\\\");
        check("high bytes pass through", json_escape_strict(S("\xc3\xa9")) == "\xc3\xa9");

        // A/B: the legacy escaper left \t and \r RAW (invalid JSON) — proves the
        // gate reproduces the old, broken behaviour.
        check("A/B legacy leaves tab raw", json_escape_legacy(S("a\tb")) == "a\tb");
        check("A/B legacy leaves CR raw", json_escape_legacy(S("a\rb")) == "a\rb");
        check("A/B legacy == strict on tab-free text",
              json_escape_legacy(S("plain \"q\"")) == json_escape_strict(S("plain \"q\"")));
    }

    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

// Route through core_util::clean_exit per the tools/check_test_clean_exit.sh guard
// (host-only test, but the guard is blanket over tests/*.cpp mains).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
