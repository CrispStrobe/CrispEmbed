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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using core_json::json_decode_string;
using core_json::json_escape_legacy;
using core_json::json_escape_strict;
using core_json::json_extract_bool;
using core_json::json_extract_documents;
using core_json::json_extract_number;
using core_json::json_extract_number_legacy;
using core_json::json_extract_number_structural;
using core_json::json_extract_strings;
using core_json::json_extract_strings_escaped;
using core_json::json_extract_strings_legacy;
using core_json::json_find_key_value;
using core_json::json_skip_value;

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

    // =================================================================
    // Scalar number extraction (json_extract_number) — same depth-1 finder as
    // the array parser, so it shares the decoy-immunity and honours defaults.
    // =================================================================
    {
        auto approx = [](double a, double b) { return (a - b) < 1e-9 && (b - a) < 1e-9; };
        // Basic int / float / negative / exponent.
        check("number: int", json_extract_number(S("{\"max_tokens\":128}"), "max_tokens", -1) == 128);
        check("number: float", approx(json_extract_number(S("{\"conf\":0.35}"), "conf", -1), 0.35));
        check("number: negative", approx(json_extract_number(S("{\"t\":-2.5}"), "t", 0), -2.5));
        check("number: whitespace", json_extract_number(S("{ \"n\" : 42 }"), "n", -1) == 42);
        // Absent / wrong-type -> default (a string/bool value is NOT a number).
        check("number: absent -> default", json_extract_number(S("{\"a\":1}"), "conf", 7) == 7);
        check("number: string value -> default", json_extract_number(S("{\"conf\":\"0.5\"}"), "conf", 7) == 7);
        check("number: bool value -> default", json_extract_number(S("{\"conf\":true}"), "conf", 7) == 7);
        // Decoy: a string VALUE equal to the key name must not be matched (B2).
        check("number: decoy value ignored",
              json_extract_number(S("{\"label\":\"threshold\",\"threshold\":0.9}"), "threshold", -1) > 0.89 &&
                  json_extract_number(S("{\"label\":\"threshold\",\"threshold\":0.9}"), "threshold", -1) < 0.91);
        check("number: decoy-only -> default (key truly absent)",
              json_extract_number(S("{\"labels\":[\"conf\"]}"), "conf", 5) == 5);
        // A/B: legacy naive scan latches onto the decoy; structural does not.
        double leg = json_extract_number_legacy(S("{\"labels\":[\"conf\"],\"x\":0.1}"), "conf", -1);
        double str = json_extract_number_structural(S("{\"labels\":[\"conf\"],\"x\":0.1}"), "conf", -1);
        std::printf("      (legacy=%.3f structural=%.3f)\n", leg, str);
        check("A/B number: structural returns default on decoy-only", str == -1);
        check("A/B number: legacy is fooled by the decoy (!=default)", leg != -1);
        // A/B: agree on a normal payload.
        check("A/B number: agree on well-formed", json_extract_number_legacy(S("{\"conf\":0.75}"), "conf", 0) ==
                                                      json_extract_number_structural(S("{\"conf\":0.75}"), "conf", 0));
    }

    // T8: the server's remaining non-path fields (text/format/results/
    // autorotate/images) moved off bare body.find() onto these helpers. One
    // nested-decoy case per field: the decoy sits inside a nested object
    // BEFORE the real top-level key, which is exactly where the old textual
    // scan latched on. Each case also shows the naive scan being fooled.
    {
        // "text": decoy inside meta, real value top-level.
        std::string body = S("{\"meta\":{\"text\":\"DECOY\"},\"text\":\"real input\"}");
        std::vector<std::string> out;
        check("T8 text: top-level wins over nested decoy",
              json_extract_strings(body, "text", out) == 1 && out[0] == "real input");
        check("T8 text: naive scan would take the decoy first",
              body.find("\"text\"") < body.find("\"text\"", body.find("\"meta\"") + 20));
    }
    {
        // "format": decoy nested, real top-level.
        std::string body = S("{\"opts\":{\"format\":\"hocr\"},\"format\":\"pdf\"}");
        std::vector<std::string> out;
        check("T8 format: top-level wins over nested decoy",
              json_extract_strings(body, "format", out) == 1 && out[0] == "pdf");
        check("T8 format: decoy-only body yields nothing",
              json_extract_strings(S("{\"opts\":{\"format\":\"hocr\"}}"), "format", out = {}) == 0);
    }
    {
        // "results": value locator must land on the TOP-LEVEL array, not a
        // nested key of the same name.
        std::string body = S("{\"meta\":{\"results\":[{\"text\":\"DECOY\"}]},\"results\":[{\"text\":\"ok\"}]}");
        size_t p = json_find_key_value(body, "results");
        check("T8 results: finder lands on the top-level array",
              p != std::string::npos && body[p] == '[' && body.find("\"ok\"") > p && body.find("\"DECOY\"") < p);
    }
    {
        // "autorotate": nested decoy true, top-level false.
        std::string body = S("{\"opts\":{\"autorotate\":true},\"autorotate\":false}");
        size_t p = json_find_key_value(body, "autorotate");
        check("T8 autorotate: finder lands on the top-level literal",
              p != std::string::npos && body.compare(p, 5, "false") == 0);
        check("T8 autorotate: naive scan finds the nested decoy first", body.find("\"autorotate\"") < p);
    }
    {
        // "images": nested decoy array must not leak paths into the result.
        std::string body = S("{\"meta\":{\"images\":[\"/etc/DECOY\"]},\"images\":[\"/srv/a.png\",\"/srv/b.png\"]}");
        std::vector<std::string> out;
        size_t n = json_extract_strings(body, "images", out);
        check("T8 images: exactly the two top-level paths", n == 2 && out[0] == "/srv/a.png" && out[1] == "/srv/b.png");
        check("T8 images: decoy path did not leak", std::find(out.begin(), out.end(), "/etc/DECOY") == out.end());
    }

    // -----------------------------------------------------------------------
    // Issue #51 — the Cohere / Jina rerank request shape.
    //
    // The defect these guard is specific: json_extract_strings flattens every
    // string inside the array, so the object form
    //     "documents": [{"text":"a"},{"text":"b"}]
    // parses as FOUR documents ["text","a","text","b"] — the wrong cardinality
    // AND wrong content, which the server would then happily score and return
    // with indices that mean nothing to the caller. Each check below is written
    // so the pre-helper behaviour fails it.
    // -----------------------------------------------------------------------
    {
        std::string body =
            S("{\"model\":\"m\",\"query\":\"q\",\"documents\":[{\"text\":\"alpha\"},{\"text\":\"beta\"}]}");
        std::vector<std::string> out;
        size_t n = json_extract_documents(body, "documents", out);
        check("R1 object documents: cardinality is 2, not 4",
              n == 2 && out.size() == 2 && out[0] == "alpha" && out[1] == "beta");
        // The bug this replaces, stated as a property rather than a value.
        std::vector<std::string> flat;
        json_extract_strings(body, "documents", flat);
        check("R1 control: the flat parser really does mis-split this shape", flat.size() == 4);
    }
    {
        // The plain-string form must be untouched — /rerank's existing contract.
        std::string body = S("{\"query\":\"q\",\"documents\":[\"a\",\"b\",\"c\"]}");
        std::vector<std::string> out;
        check("R2 string documents still parse as 3",
              json_extract_documents(body, "documents", out) == 3 && out[2] == "c");
    }
    {
        // Mixed forms, and escaping inside an object element (']' and \" are the
        // exact characters issue #34 mis-split).
        std::string body = S("{\"documents\":[\"plain ] one\",{\"text\":\"obj \\\"two\\\"\"},\"three\"]}");
        std::vector<std::string> out;
        size_t n = json_extract_documents(body, "documents", out);
        check("R3 mixed string/object elements with escapes",
              n == 3 && out[0] == "plain ] one" && out[1] == "obj \"two\"" && out[2] == "three");
    }
    {
        // An object element with no "text" keeps its SLOT. Dropping it would
        // shift every later index, and `index` is the whole point of the
        // response — the caller maps it back onto the array it sent.
        std::string body = S("{\"documents\":[{\"text\":\"a\"},{\"id\":7},{\"text\":\"c\"}]}");
        std::vector<std::string> out;
        size_t n = json_extract_documents(body, "documents", out);
        check("R4 text-less object holds its index slot", n == 3 && out[0] == "a" && out[1].empty() && out[2] == "c");
    }
    {
        // Nested objects inside an element must not confuse the element walk.
        std::string body =
            S("{\"documents\":[{\"meta\":{\"text\":\"DECOY\"},\"text\":\"real\"},{\"text\":\"second\"}]}");
        std::vector<std::string> out;
        size_t n = json_extract_documents(body, "documents", out);
        check("R5 nested decoy inside an element does not win", n == 2 && out[0] == "real" && out[1] == "second");
    }
    {
        // Degenerate single-value forms.
        std::vector<std::string> a, b;
        check("R6 single string value",
              json_extract_documents(S("{\"documents\":\"solo\"}"), "documents", a) == 1 && a[0] == "solo");
        check("R6 single object value",
              json_extract_documents(S("{\"documents\":{\"text\":\"solo\"}}"), "documents", b) == 1 && b[0] == "solo");
        std::vector<std::string> c;
        check("R6 missing key yields nothing", json_extract_documents(S("{\"query\":\"q\"}"), "documents", c) == 0);
    }
    {
        // A '{' inside a STRING element must not be read as an object element.
        std::string body = S("{\"documents\":[\"a { brace\",\"b\"]}");
        std::vector<std::string> out;
        size_t n = json_extract_documents(body, "documents", out);
        check("R7 brace inside a string element", n == 2 && out[0] == "a { brace" && out[1] == "b");
    }
    {
        // json_skip_value: each container must end at ITS OWN closer, and a
        // ']' / '}' inside a string must not close anything.
        std::string obj = S("{\"a\":[1,2],\"b\":\"] } \"}TAIL");
        check("R8 skip_value: object ends before TAIL", json_skip_value(obj, 0) == obj.find("TAIL"));
        std::string arr = S("[\"x\",{\"y\":[3]}]REST");
        check("R8 skip_value: array ends before REST", json_skip_value(arr, 0) == arr.find("REST"));
        std::string str = S("\"esc \\\" quote\", next");
        check("R8 skip_value: string honours the escaped quote", json_skip_value(str, 0) == str.find(","));
        check("R8 skip_value: number stops at the delimiter", json_skip_value(S("42,"), 0) == 2);
    }
    {
        // return_documents — structural, so a nested decoy cannot flip it, and a
        // string value equal to the key name cannot be mistaken for the key.
        check("R9 bool true", json_extract_bool(S("{\"return_documents\":true}"), "return_documents", false));
        check("R9 bool false", !json_extract_bool(S("{\"return_documents\":false}"), "return_documents", true));
        check("R9 bool default when absent", json_extract_bool(S("{\"query\":\"q\"}"), "return_documents", true));
        check("R9 bool numeric 1/0 accepted",
              json_extract_bool(S("{\"return_documents\":1}"), "return_documents", false) &&
                  !json_extract_bool(S("{\"return_documents\":0}"), "return_documents", true));
        check("R9 bool nested decoy does not win",
              !json_extract_bool(S("{\"opts\":{\"return_documents\":true},\"return_documents\":false}"),
                                 "return_documents", true));
        check("R9 bool ignores a quoted value",
              json_extract_bool(S("{\"return_documents\":\"true\"}"), "return_documents", false) == false);
    }
    {
        // The sigmoid the /v1/rerank response applies to the raw logit. It is
        // only worth a test for the property the endpoint depends on: it is
        // strictly monotonic, so it cannot reorder results, and it lands in
        // (0,1) for the logit range a cross-encoder head actually produces.
        auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        const double logits[] = { -12.0, -3.5, -1.0, 0.0, 0.25, 2.0, 11.0 };
        bool monotonic = true, in_range = true;
        for (size_t i = 0; i < sizeof(logits) / sizeof(logits[0]); i++) {
            const double v = sigmoid(logits[i]);
            if (!(v > 0.0 && v < 1.0)) in_range = false;
            if (i > 0 && !(v > sigmoid(logits[i - 1]))) monotonic = false;
        }
        check("R10 sigmoid is strictly monotonic (ranking preserved)", monotonic);
        check("R10 sigmoid stays inside (0,1)", in_range);
        check("R10 sigmoid(0) == 0.5", std::fabs(sigmoid(0.0) - 0.5) < 1e-12);
    }

    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

// Route through core_util::clean_exit per the tools/check_test_clean_exit.sh guard
// (host-only test, but the guard is blanket over tests/*.cpp mains).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
