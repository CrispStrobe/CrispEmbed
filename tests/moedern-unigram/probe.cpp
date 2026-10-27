#include "moedern_text_frontend.h"
#include "core/clean_exit.h"
#include "core/json.h"
#include "core/unicode_categ.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
namespace mc = moedern_colbert;
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MOEDERN_TEXT_FRONTEND", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_MOEDERN_TEXT_FRONTEND", value, 1);
    else
        unsetenv("CRISPEMBED_MOEDERN_TEXT_FRONTEND");
#endif
}
std::string quote(const std::string & s) {
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
std::string unhex(const std::string & s) {
    require(s.size() % 2 == 0, "bad hex length");
    std::string out;
    const std::string digits = "0123456789abcdef";
    for (size_t i = 0; i < s.size(); i += 2) {
        auto a = digits.find(s[i]), b = digits.find(s[i + 1]);
        require(a < 16 && b < 16, "bad hex digit");
        out += char(a * 16 + b);
    }
    return out;
}
struct row {
    std::string id, role, text;
};
row parse(const std::string & line) {
    auto a = line.find('\t'), b = line.find('\t', a == std::string::npos ? 0 : a + 1);
    require(a != std::string::npos && b != std::string::npos, "bad input row");
    return { line.substr(0, a), line.substr(a + 1, b - a - 1), unhex(line.substr(b + 1)) };
}
mc::role role(const row & r) {
    return r.role == "query" ? mc::role::query : r.role == "document" ? mc::role::document : static_cast<mc::role>(99);
}
template <class T> void numbers(std::ostream & out, const std::vector<T> & v) {
    out << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out << ',';
        out << +v[i];
    }
    out << ']';
}
void result(std::ostream & out, const mc::text_tokens & r) {
    out << "{\"ids\":";
    numbers(out, r.ids);
    out << ",\"supplied_mask\":";
    numbers(out, r.supplied_mask);
    out << ",\"effective_mask\":";
    numbers(out, r.effective_mask);
    out << ",\"scoring_mask\":";
    numbers(out, r.scoring_mask);
    out << ",\"expansion_positions\":";
    numbers(out, r.expansion_positions);
    out << '}';
}
int run(int argc, char ** argv) {
    require(argc >= 4, "usage: probe MODEL INPUT OUTPUT | --normalize MODEL INPUT OUTPUT | --gates MODEL INPUT");
    const std::string mode = argv[1];
    const bool segment_mode = mode == "--segments", diagnostic = mode == "--normalize" || segment_mode,
               gates = mode == "--gates";
    const char * model = diagnostic || gates ? argv[2] : argv[1];
    std::ifstream input(diagnostic || gates ? argv[3] : argv[2]);
    require(bool(input), "input open failed");
    std::string line, error;
    if (gates) {
        require(std::getline(input, line).good() || !line.empty(), "first gate input missing");
        const auto r = parse(line);
        for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
            gate(value);
            mc::text_frontend disabled;
            require(!disabled.load(model, error) && error == "MoEdern text frontend gate is OFF", "off gate failed");
            mc::text_tokens out;
            out.ids = { 123 };
            require(!disabled.encode(r.text, role(r), out, error) && out.ids.empty(), "off output stale");
        }
        gate("1");
        mc::text_frontend enabled;
        bool loaded = enabled.load(model, error);
        require(loaded, error.c_str());
        {
            mc::text_frontend other;
            const bool second_loaded = other.load(model, error);
            require(second_loaded, error.c_str());
            mc::text_tokens first_value, second_value;
            require(enabled.encode(r.text, role(r), first_value, error) &&
                        other.encode(r.text, role(r), second_value, error) && first_value.ids == second_value.ids &&
                        first_value.scoring_mask == second_value.scoring_mask,
                    "two-owner token mismatch");
        }
        // Actual production encode must retain both historical normalized role
        // matchers and the subsequently registered raw special matcher.
        mc::text_tokens overlap;
        require(enabled.encode("[q] [d] [Q] ", mc::role::document, overlap, error) &&
                    overlap.ids == std::vector<int32_t>({ 2, 30001, 30000, 30001, 30000, 3 }) &&
                    overlap.supplied_mask == std::vector<uint8_t>(6, 1) &&
                    overlap.effective_mask == std::vector<uint8_t>(6, 1) &&
                    overlap.scoring_mask == std::vector<uint8_t>(6, 1) && overlap.expansion_positions.empty(),
                "raw/normalized added-token matcher history failed");
        gate("0");
        mc::text_tokens out;
        require(enabled.encode(r.text, role(r), out, error) && !out.ids.empty(), "context gate snapshot changed");
        require(!enabled.load(model, error) && !enabled.enabled(), "failed reload retained context");
        std::cout << "PASS unset/empty/0/context gate/output/reload/matcher-history controls\n";
        return 0;
    }
    require(argc == (diagnostic ? 5 : 4), "bad arguments");
    std::ofstream output(diagnostic ? argv[4] : argv[3]);
    require(bool(output), "output open failed");
    mc::text_frontend frontend;
    bool loaded = frontend.load(model, error);
    require(loaded, error.c_str());
    while (std::getline(input, line)) {
        const auto r = parse(line);
        output << "{\"id\":" << quote(r.id);
        if (diagnostic) {
#ifdef CRISPEMBED_MOEDERN_TEXT_TESTING
            if (segment_mode) {
                output << ",\"segment_ids\":";
                numbers(output, frontend.test_unigram(r.text));
            } else {
                const auto normalized = frontend.test_normalize(r.text);
                const auto segments = frontend.test_pretokens(normalized);
                output << ",\"normalized\":" << quote(normalized) << ",\"segments\":[";
                for (size_t i = 0; i < segments.size(); ++i) {
                    if (i) output << ',';
                    output << quote(segments[i]);
                }
                output << ']';
                output << ",\"segment_utf8_boundaries\":[";
                for (size_t i = 0; i < segments.size(); ++i) {
                    if (i) output << ',';
                    std::vector<int> boundaries{ 0 };
                    for (size_t at = 0; at < segments[i].size();) {
                        core_unicode::utf8_next(segments[i].data(), segments[i].size(), at);
                        boundaries.push_back(int(at));
                    }
                    numbers(output, boundaries);
                }
                output << "]";
            }
#else
            throw std::runtime_error("diagnostic absent from ordinary build");
#endif
        } else {
            mc::text_tokens out;
            out.ids = { 123 };
            bool ok = frontend.encode(r.text, role(r), out, error);
            output << ",\"accepted\":" << (ok ? "true" : "false");
            if (ok) {
                output << ",\"encoded\":";
                result(output, out);
            } else {
                require(out.ids.empty() && out.supplied_mask.empty() && out.effective_mask.empty() &&
                            out.scoring_mask.empty() && out.expansion_positions.empty(),
                        "rejected output stale");
                output << ",\"error\":" << quote(error) << ",\"output_empty\":true";
            }
        }
        output << "}\n";
    }
    require(output.good(), "write failure");
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    int rc = 0;
    try {
        rc = run(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        rc = 2;
    }
    core_util::clean_exit(rc);
}
