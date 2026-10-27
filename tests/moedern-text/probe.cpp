#include "moedern_text.h"
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
#include "moedern_text_testing.h"
#endif
#include "core/clean_exit.h"
#include "core/json.h"
#include <cstring>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
namespace mt = moedern_text;
static_assert(
    std::is_same_v<decltype(&mt::encode), bool (*)(mt::context *, const std::string &, mt::role, mt::result &)>,
    "ordinary API has no reference callback argument");
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
std::string unhex(const std::string & s) {
    require(s.size() % 2 == 0, "hex length");
    std::string out;
    const std::string digits = "0123456789abcdef";
    for (size_t i = 0; i < s.size(); i += 2) {
        auto a = digits.find(s[i]), b = digits.find(s[i + 1]);
        require(a < 16 && b < 16, "hex digit");
        out += char(a * 16 + b);
    }
    return out;
}
struct input {
    std::string id, task, text;
};
input parse(const std::string & s) {
    auto a = s.find('\t'), b = s.find('\t', a == std::string::npos ? 0 : a + 1);
    require(a != std::string::npos && b != std::string::npos, "raw text row");
    return { s.substr(0, a), s.substr(a + 1, b - a - 1), unhex(s.substr(b + 1)) };
}
mt::role task(const input & i) {
    return i.task == "query" ? mt::role::query : i.task == "document" ? mt::role::document : static_cast<mt::role>(99);
}
bool equal(const mt::result & a, const mt::result & b) {
    return a.ids == b.ids && a.supplied_mask == b.supplied_mask && a.effective_mask == b.effective_mask &&
           a.scoring_mask == b.scoring_mask && a.expansion_positions == b.expansion_positions &&
           a.all_vectors.size() == b.all_vectors.size() && a.selected_vectors.size() == b.selected_vectors.size() &&
           std::memcmp(a.all_vectors.data(), b.all_vectors.data(), a.all_vectors.size() * 4) == 0 &&
           std::memcmp(a.selected_vectors.data(), b.selected_vectors.data(), a.selected_vectors.size() * 4) == 0;
}
void gate(const char * v) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MOEDERN_TEXT_COMPOSED", v ? v : "");
#else
    if (v)
        setenv("CRISPEMBED_MOEDERN_TEXT_COMPOSED", v, 1);
    else
        unsetenv("CRISPEMBED_MOEDERN_TEXT_COMPOSED");
#endif
}
void bytes(const std::filesystem::path & p, const void * d, size_t n) {
    std::ofstream out(p, std::ios::binary);
    out.write(static_cast<const char *>(d), std::streamsize(n));
    require(bool(out), "array write failed");
}
template <class T> void numbers(std::ostream & o, const std::vector<T> & v) {
    o << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o << ',';
        o << +v[i];
    }
    o << ']';
}
std::string quote(const std::string & s) {
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
void metadata(const std::filesystem::path & p, const mt::result & r) {
    std::ofstream o(p);
    o << "{\"ids\":";
    numbers(o, r.ids);
    o << ",\"supplied_mask\":";
    numbers(o, r.supplied_mask);
    o << ",\"effective_mask\":";
    numbers(o, r.effective_mask);
    o << ",\"scoring_mask\":";
    numbers(o, r.scoring_mask);
    o << ",\"expansion_positions\":";
    numbers(o, r.expansion_positions);
    o << ",\"all_shape\":[" << r.ids.size() << ",128],\"selected_shape\":[" << r.selected_vectors.size() / 128
      << ",128]}\n";
    require(bool(o), "metadata write failed");
}
int run(int argc, char ** argv) {
    require(argc == 5, "usage: probe MODEL TOKENIZER RAW_TEXT_TSV OUTPUT_DIR");
    std::ifstream in(argv[3]);
    require(bool(in), "raw text input absent");
    const std::filesystem::path out(argv[4]);
    require(!std::filesystem::exists(out), "fresh output required");
    std::filesystem::create_directories(out);
    std::string error, line;
    for (const char * g : { static_cast<const char *>(nullptr), "", "0" }) {
        gate(g);
        std::unique_ptr<mt::context, decltype(&mt::free)> c(
            mt::init("must-not-open-model", "must-not-open-tokenizer", 2, error), mt::free);
        require(!c && error.find("OFF") != std::string::npos, "default-off guard failed");
    }
    gate("1");
    std::unique_ptr<mt::context, decltype(&mt::free)> ctx(mt::init(argv[1], argv[2], 2, error), mt::free);
    require(bool(ctx), error.c_str());
    size_t cases = 0;
    bool any_failed = false;
    while (std::getline(in, line)) {
        const auto row = parse(line);
        require(!row.id.empty() &&
                    row.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") == std::string::npos,
                "unsafe ID");
        const auto dir = out / row.id;
        std::filesystem::create_directory(dir);
        ++cases;
        try {
            mt::result ordinary, repeat;
            const bool encoded = mt::encode(ctx.get(), row.text, task(row), ordinary);
            require(encoded, mt::error(ctx.get()));
            require(mt::encode(ctx.get(), row.text, task(row), repeat) && equal(ordinary, repeat),
                    "ordinary repeat differs");
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
            std::ofstream stages(dir / "stages.tsv");
            stages << "stage\ttype\tshape\tfile\tbytes\n";
            std::set<std::string> names;
            size_t count = 0, total = 0;
            auto observe = [&](const std::string & name, const moedern_text_test::tensor_view & view) {
                require(names.insert(name).second, "duplicate stage");
                size_t elements = 1;
                for (auto dim : view.shape) {
                    require(dim > 0 && dim <= 512, "bad stage shape");
                    elements *= size_t(dim);
                }
                require(view.bytes == elements * (view.type == moedern_text_test::scalar::f32 ? 4 : 8),
                        "bad stage bytes");
                total += view.bytes;
                require(total <= 128 * 1024 * 1024, "stage budget");
                std::string file =
                    std::to_string(count++) + (view.type == moedern_text_test::scalar::f32 ? ".f32" : ".i64");
                bytes(dir / file, view.data, view.bytes);
                stages << name << '\t' << (view.type == moedern_text_test::scalar::f32 ? "f32" : "i64") << '\t';
                for (size_t i = 0; i < view.shape.size(); ++i) {
                    if (i) stages << ',';
                    stages << view.shape[i];
                }
                stages << '\t' << file << '\t' << view.bytes << '\n';
                return bool(stages);
            };
            mt::result observed;
            require(moedern_text_test::encode_observed(ctx.get(), row.text, task(row), observed, observe) &&
                        equal(ordinary, observed) && count == 55,
                    "ordinary/observer/stage coverage differs");
            mt::result failed;
            failed.all_vectors = { 123 };
            require(!moedern_text_test::encode_observed(
                        ctx.get(), row.text, task(row), failed,
                        [](const std::string &, const moedern_text_test::tensor_view &) { return false; }) &&
                        failed.all_vectors.empty() && failed.selected_vectors.empty() && failed.ids.empty(),
                    "observer rejection/output failed");
            require(mt::encode(ctx.get(), row.text, task(row), repeat) && equal(ordinary, repeat),
                    "observer recovery differs");
#endif
            gate("0");
            repeat.all_vectors = { 123 };
            require(!mt::encode(ctx.get(), row.text, task(row), repeat) && repeat.ids.empty() &&
                        repeat.all_vectors.empty() && repeat.selected_vectors.empty(),
                    "gate rejection stale");
            gate("1");
            require(mt::encode(ctx.get(), row.text, task(row), repeat) && equal(ordinary, repeat),
                    "gate recovery differs");
            repeat.all_vectors = { 123 };
            require(!mt::encode(ctx.get(), row.text, static_cast<mt::role>(99), repeat) && repeat.all_vectors.empty() &&
                        repeat.ids.empty(),
                    "invalid role accepted");
            require(!mt::encode(ctx.get(), std::string(65537, 'a'), task(row), repeat) && repeat.all_vectors.empty(),
                    "oversize accepted");
            require(mt::encode(ctx.get(), row.text, task(row), repeat) && equal(ordinary, repeat),
                    "input recovery differs");
            {
                std::unique_ptr<mt::context, decltype(&mt::free)> other(mt::init(argv[1], argv[2], 1, error), mt::free);
                require(bool(other), error.c_str());
                require(mt::encode(other.get(), row.text, task(row), repeat), "second owner encode failed");
            }
            require(mt::encode(ctx.get(), row.text, task(row), repeat) && equal(ordinary, repeat),
                    "two-context ownership failed");
            bytes(dir / "all.f32", ordinary.all_vectors.data(), ordinary.all_vectors.size() * 4);
            bytes(dir / "selected.f32", ordinary.selected_vectors.data(), ordinary.selected_vectors.size() * 4);
            metadata(dir / "tokens.json", ordinary);
            std::ofstream(dir / "controls.json")
                << "{\"repeat\":true,\"gate_recovery\":true,\"role_bounds_recovery\":true,\"context_ownership\":true"
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
                << ",\"observer_neutrality\":true,\"observer_recovery\":true,\"stages\":55"
#endif
                << "}\n";
        } catch (const std::exception & e) {
            any_failed = true;
            gate("1");
            std::ofstream(dir / "rejection.json") << "{\"error\":" << quote(e.what()) << "}\n";
        }
    }
    require(cases == 10, "exact frozen ten text cases required");
    std::ifstream pair_input(std::string(argv[3]) + ".pairs");
    require(bool(pair_input), "raw text score pairs missing");
    std::ofstream scores(out / "scores.jsonl");
    size_t pairs = 0;
    scores << std::setprecision(17);
    while (std::getline(pair_input, line)) {
        const auto pair = parse(line);
        require(!pair.id.empty() &&
                    pair.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") == std::string::npos,
                "unsafe score ID");
        ++pairs;
        try {
            const auto query = unhex(pair.task);
            float value = 0;
            const bool computed = mt::score(ctx.get(), query, pair.text, value);
            require(computed, mt::error(ctx.get()));
            require(std::isfinite(value), "nonfinite raw-text score");
            gate("0");
            float rejected = 123;
            require(!mt::score(ctx.get(), query, pair.text, rejected) && rejected == 0, "score gate output stale");
            gate("1");
            float repeated = 0;
            require(mt::score(ctx.get(), query, pair.text, repeated) && std::memcmp(&repeated, &value, 4) == 0,
                    "score recovery differs");
            bytes(out / (pair.id + ".f32"), &value, sizeof(value));
            scores << "{\"id\":" << quote(pair.id) << ",\"score\":" << value << "}\n";
        } catch (const std::exception & e) {
            any_failed = true;
            gate("1");
            scores << "{\"id\":" << quote(pair.id) << ",\"error\":" << quote(e.what()) << "}\n";
        }
    }
    require(pairs == 3 && scores.good(), "exact three raw-text scores required");
    return any_failed ? 1 : 0;
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
