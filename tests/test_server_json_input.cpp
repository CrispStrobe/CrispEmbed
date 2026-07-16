// Unit test for the HTTP server's JSON string-input parser (issue #34).
//
// Host-only: no model, no ggml — exercises examples/server/json_input.h directly.
// Verifies that valid escaped JSON payloads (values containing ']', \" and \\,
// newlines, and \uXXXX) parse to the correct input cardinality, rather than the
// old delimiter-scan approach that produced "returned 7 embeddings for 6 inputs".
//
// Returns non-zero exit code on any failure (matches the repo's other test-* runners).

#include "json_input.h"

#include <cstdio>
#include <string>
#include <vector>

using crispembed_server::json_decode_string;
using crispembed_server::json_extract_strings;

static int g_failures = 0;

static void check(const char * name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_failures++;
}

// Build a std::string from a literal without raw-string edge cases.
static std::string S(const char * s) {
    return std::string(s);
}

int main() {
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

    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
