// Hosted-only component probe. Build josev2b_head.cpp with
// CRISPEMBED_JOSEV_HEAD_TESTING; never link this observer variant into products.
#include "josev2b_head.h"
#include "josev2b_head_observer.h"
#include "core/clean_exit.h"
#include "core/json.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
namespace fs = std::filesystem;
namespace {
using object = std::map<std::string, std::string>;
void require(bool value, const std::string & error) {
    if (!value) throw std::runtime_error(error);
}
size_t skip(const std::string & s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    return i;
}
object fields(const std::string & s) {
    size_t i = skip(s, 0);
    require(i < s.size() && s[i++] == '{', "JSON object required");
    object out;
    i = skip(s, i);
    if (i < s.size() && s[i] == '}') {
        require(skip(s, i + 1) == s.size(), "Trailing JSON bytes");
        return out;
    }
    while (i < s.size()) {
        std::string name;
        size_t end = 0;
        require(core_json::json_decode_string(s, i, name, end), "JSON member key required");
        i = skip(s, end);
        require(i < s.size() && s[i++] == ':', "JSON colon required");
        i = skip(s, i);
        end = core_json::json_skip_value(s, i);
        require(end != std::string::npos && end > i, "JSON value required");
        require(out.emplace(name, s.substr(i, end - i)).second, "Duplicate JSON member");
        i = skip(s, end);
        require(i < s.size(), "Truncated JSON object");
        if (s[i] == '}') {
            require(skip(s, i + 1) == s.size(), "Trailing JSON bytes");
            return out;
        }
        require(s[i++] == ',', "JSON comma required");
        i = skip(s, i);
    }
    throw std::runtime_error("Unclosed JSON object");
}
std::vector<std::string> array(const std::string & s) {
    size_t i = skip(s, 0);
    require(i < s.size() && s[i++] == '[', "JSON array required");
    std::vector<std::string> out;
    i = skip(s, i);
    if (i < s.size() && s[i] == ']') {
        require(skip(s, i + 1) == s.size(), "Trailing JSON array bytes");
        return out;
    }
    while (i < s.size()) {
        const size_t end = core_json::json_skip_value(s, i);
        require(end != std::string::npos && end > i, "JSON array value required");
        out.push_back(s.substr(i, end - i));
        i = skip(s, end);
        require(i < s.size(), "Truncated JSON array");
        if (s[i] == ']') {
            require(skip(s, i + 1) == s.size(), "Trailing JSON array bytes");
            return out;
        }
        require(s[i++] == ',', "JSON array comma required");
        i = skip(s, i);
    }
    throw std::runtime_error("Unclosed JSON array");
}
void keys(const object & o, std::set<std::string> expected) {
    std::set<std::string> actual;
    for (const auto & e : o) actual.insert(e.first);
    require(actual == expected, "Exact manifest key set required");
}
std::string text(const std::string & s) {
    std::string value;
    size_t end = 0;
    require(core_json::json_decode_string(s, 0, value, end) && skip(s, end) == s.size(), "JSON string required");
    return value;
}
int integer(const std::string & s) {
    // json_skip_value retains trailing primitive whitespace. Trim exactly JSON's
    // four whitespace bytes, never locale whitespace or bytes inside a number.
    const auto first = s.find_first_not_of(" \t\r\n");
    require(first != std::string::npos, "Nonnegative integer required");
    const auto token = s.substr(first, s.find_last_not_of(" \t\r\n") + 1 - first);
    require(std::all_of(token.begin(), token.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "Nonnegative integer required");
    const auto value = std::stoul(token);
    require(value <= 65536, "Manifest integer bound exceeded");
    return int(value);
}
std::string read_text(const fs::path & path, size_t bound) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file && file.tellg() > 0 && file.tellg() <= std::streamoff(bound),
            "Bounded regular input manifest required");
    std::string data(size_t(file.tellg()), '\0');
    file.seekg(0);
    require(bool(file.read(data.data(), data.size())), "Manifest read failed");
    return data;
}
std::vector<float> floats(const fs::path & root, const std::string & name, size_t count) {
    fs::path relative(name);
    require(!name.empty() && !relative.is_absolute() && !relative.has_parent_path() &&
                relative.filename() == relative && name != ".." && name != ".",
            "Raw input must be a flat relative filename");
    const auto path = root / relative;
    require(!fs::is_symlink(path) && fs::is_regular_file(path), "Regular nonsymlink F32 input required");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file && file.tellg() == std::streamoff(count * sizeof(float)), "Exact raw F32 input extent required");
    std::vector<float> result(count);
    file.seekg(0);
    require(bool(file.read(reinterpret_cast<char *>(result.data()), count * sizeof(float))),
            "Raw F32 input read failed");
    return result;
}
josev2b_head::input load(const fs::path & path) {
    const auto o = fields(read_text(path, 1024 * 1024));
    keys(o, { "schema", "tokens", "hidden_f32", "fields" });
    require(text(o.at("schema")) == "crispembed-josev-head-input-v1", "Head input schema differs");
    josev2b_head::input values;
    values.tokens = integer(o.at("tokens"));
    require(values.tokens >= 1 && values.tokens <= 512, "Token bound exceeded");
    values.hidden = floats(path.parent_path(), text(o.at("hidden_f32")), size_t(values.tokens) * 2048);
    const auto rows = array(o.at("fields"));
    require(!rows.empty() && rows.size() <= 16, "Field bound exceeded");
    size_t count = 0;
    for (const auto & raw : rows) {
        const auto f = fields(raw);
        keys(f, { "id", "type", "question_start", "question_end", "options", "lexical_f32" });
        josev2b_head::field field;
        field.id = text(f.at("id"));
        field.type = integer(f.at("type"));
        field.question = { integer(f.at("question_start")), integer(f.at("question_end")) };
        const auto options = array(f.at("options"));
        count += options.size();
        require(!options.empty() && count <= 128, "Option bound exceeded");
        for (const auto & raw_option : options) {
            const auto option = fields(raw_option);
            keys(option, { "start", "end" });
            field.options.push_back({ integer(option.at("start")), integer(option.at("end")) });
        }
        field.lexical_means = floats(path.parent_path(), text(f.at("lexical_f32")), field.options.size() * 2048);
        values.fields.push_back(std::move(field));
    }
    std::string error;
    require(josev2b_head::validate_input(values, error), error);
    return values;
}
void raw(const fs::path & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    require(bool(file), "Probe F32 output write failed");
}
bool exact(const josev2b_head::output & a, const josev2b_head::output & b) {
    if (a.backend != b.backend || a.fields.size() != b.fields.size()) return false;
    for (size_t i = 0; i < a.fields.size(); ++i) {
        const auto & x = a.fields[i];
        const auto & y = b.fields[i];
        if (x.id != y.id || x.logits.size() != y.logits.size() || x.probabilities.size() != y.probabilities.size() ||
            std::memcmp(x.logits.data(), y.logits.data(), x.logits.size() * sizeof(float)) ||
            std::memcmp(x.probabilities.data(), y.probabilities.data(), x.probabilities.size() * sizeof(float)))
            return false;
    }
    return true;
}
bool empty_output(const josev2b_head::output & out) {
    return out.fields.empty() && out.backend.empty();
}
josev2b_head::output stale_output() {
    josev2b_head::output out;
    out.backend = "stale-backend";
    out.fields.push_back({ "stale-field", { 1.0f }, { 1.0f } });
    return out;
}
void set_gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_JOSEV_HEAD", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_JOSEV_HEAD", value, 1);
    else
        unsetenv("CRISPEMBED_JOSEV_HEAD");
#endif
}
void contracts() {
    std::string error;
    require(josev2b_head::test::metadata_controls(error), error);
    josev2b_head::input good;
    good.tokens = 1;
    good.hidden.resize(2048);
    josev2b_head::field f;
    f.id = "owned";
    f.question = { 0, 1 };
    f.options = { { 0, 1 } };
    f.lexical_means.resize(2048);
    good.fields = { f };
    require(josev2b_head::validate_input(good, error), error);
    for (int control = 0; control < 8; ++control) {
        auto bad = good;
        if (control == 0) bad.tokens = 513;
        if (control == 1) bad.hidden.pop_back();
        if (control == 2) bad.hidden[0] = std::numeric_limits<float>::quiet_NaN();
        if (control == 3) bad.fields[0].question = { 0, 0 };
        if (control == 4) bad.fields[0].type = 3;
        if (control == 5) bad.fields[0].lexical_means.pop_back();
        if (control == 6) bad.fields.push_back(f);
        if (control == 7) bad.fields[0].options[0] = { 0, 2 };
        require(!josev2b_head::validate_input(bad, error), "Invalid input contract accepted");
    }
    require(fields("{\"id\":\"a\",\"value\":[1,2]}").size() == 2, "Parser positive failed");
    // Exercise the actual nested parser path used by the pretty-printed exporter.
    const auto pretty = fields(R"JSON({
  "schema": "crispembed-josev-head-input-v1",
  "tokens": 512 ,
  "hidden_f32": "hidden.f32",
  "fields": [
    {
      "id": "Bestellung",
      "type": 0 ,
      "question_start": 430 ,
      "question_end": 437 ,
      "options": [
        {
          "start": 448 ,
          "end": 466
        }
      ],
      "lexical_f32": "lexical-00.f32"
    }
  ]
})JSON");
    keys(pretty, { "schema", "tokens", "hidden_f32", "fields" });
    require(integer(pretty.at("tokens")) == 512, "Pretty token integer failed");
    const auto pretty_field = fields(array(pretty.at("fields")).at(0));
    require(integer(pretty_field.at("type")) == 0 && integer(pretty_field.at("question_start")) == 430 &&
                integer(pretty_field.at("question_end")) == 437,
            "Pretty field integers failed");
    const auto pretty_option = fields(array(pretty_field.at("options")).at(0));
    require(integer(pretty_option.at("start")) == 448 && integer(pretty_option.at("end")) == 466,
            "Pretty trailing option integer failed");
    require(integer(" \t\r\n0 \t\r\n") == 0 && integer(" \t\r\n65536 \t\r\n") == 65536,
            "JSON whitespace/boundary integer failed");
    size_t malformed_integer_negatives = 0;
    for (const char * bad : { "", " \t\r\n", "1.0", "1e2", "-1", "+1", "1 2", "1\n2", "\v1", "1\f", "true", "65537",
                              "999999999999999999999999999999999" }) {
        bool rejected_number = false;
        try {
            integer(bad);
        } catch (...) {
            rejected_number = true;
        }
        require(rejected_number, "Malformed/out-of-bound integer accepted");
        ++malformed_integer_negatives;
    }
    bool rejected = false;
    try {
        fields("{\"id\":1,\"id\":2}");
    } catch (...) {
        rejected = true;
    }
    require(rejected, "Duplicate manifest member accepted");
    std::printf("{\"weight_free_metadata_controls\":8,\"input_negatives\":8,\"manifest_duplicate_rejected\":true,"
                "\"pretty_json_integer_controls\":true,\"malformed_integer_negatives\":%zu,\"passed\":true}\n",
                malformed_integer_negatives);
}
} // namespace
static int crispembed_test_main(int argc, char ** argv) {
    try {
        const uint32_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4 &&
                    std::numeric_limits<float>::is_iec559,
                "Little-endian F32 probe required");
        if (argc == 2 && std::string(argv[1]) == "--contracts") {
            contracts();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--gates") {
            for (const char * flag : { static_cast<const char *>(nullptr), "", "0" }) {
                set_gate(flag);
                std::string error;
                auto out = stale_output();
                auto * ctx = josev2b_head::init("absent.gguf", nullptr, nullptr, "cpu", 2, error);
                require(!ctx && error == "Josev head requires CRISPEMBED_JOSEV_HEAD=1", "Init gate rejected too late");
                require(!josev2b_head::run(nullptr, {}, out, error) &&
                            error == "Josev head requires CRISPEMBED_JOSEV_HEAD=1",
                        "Run gate rejected too late");
                require(empty_output(out), "Run gate failure retained stale output");
                out = stale_output();
                require(!josev2b_head::test::run_observed(
                            nullptr, {}, out, [](const auto &) {}, error) &&
                            error == "Josev head requires CRISPEMBED_JOSEV_HEAD=1" && empty_output(out),
                        "Observer gate failure retained stale output");
            }
            std::printf("{\"unset_empty_zero_rejected_before_load\":true,\"gate_failure_output_empty\":true}\n");
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--reject") {
            set_gate("1");
            std::string error;
            auto * ctx = josev2b_head::init(argv[2], argv[3], argv[4], "cpu", 2, error);
            const bool rejected = !ctx;
            josev2b_head::free(ctx);
            std::printf("{\"rejected\":%s,\"error\":\"%s\"}\n", rejected ? "true" : "false",
                        core_json::json_escape_strict(error).c_str());
            return rejected ? 0 : 1;
        }
        require(argc == 7, "usage: test-josev2b-head MODEL INPUT.json OUTDIR cpu|CUDA0 RECEIPT_SHA WORKER_SHA");
        const fs::path out = argv[3];
        require(!fs::exists(out), "Fresh output required");
        fs::create_directories(out);
        const auto values = load(argv[2]);
        std::string error;
        std::unique_ptr<josev2b_head::context, decltype(&josev2b_head::free)> model(
            josev2b_head::init(argv[1], argv[5], argv[6], argv[4], 2, error), josev2b_head::free);
        require(bool(model), error);
        std::ofstream events(out / "stages.jsonl");
        size_t sequence = 0;
        josev2b_head::output ordinary, observed, repeat;
        const auto begin = std::chrono::steady_clock::now();
        require(josev2b_head::run(model.get(), values, ordinary, error), error);
        const auto ordinary_end = std::chrono::steady_clock::now();
        require(josev2b_head::test::run_observed(
                    model.get(), values, observed,
                    [&](const josev2b_head::test::stage & stage) {
                        char filename[32];
                        std::snprintf(filename, sizeof(filename), "%05zu.f32", sequence++);
                        raw(out / filename, stage.values);
                        events << "{\"stage\":\"" << core_json::json_escape_strict(stage.label)
                               << "\",\"occurrence\":" << stage.occurrence << ",\"file\":\"" << filename
                               << "\",\"shape\":[";
                        for (size_t i = 0; i < stage.shape.size(); ++i) events << (i ? "," : "") << stage.shape[i];
                        events << "],\"dtype\":\"float32\",\"elements\":" << stage.values.size() << "}\n";
                        events.flush();
                        require(bool(events), "Observer JSONL write failed");
                    },
                    error),
                error);
        const auto observed_end = std::chrono::steady_clock::now();
        require(josev2b_head::run(model.get(), values, repeat, error), error);
        require(exact(ordinary, observed) && exact(ordinary, repeat), "Ordinary/observer/repeat output bits differ");
        // Seed valid previous outputs and reject before publication on every failure.
        auto invalid = values;
        invalid.hidden.pop_back();
        auto rejected = ordinary;
        require(!josev2b_head::run(model.get(), invalid, rejected, error) && empty_output(rejected),
                "Run input failure retained previous valid output");
        rejected = ordinary;
        require(!josev2b_head::test::run_observed(
                    model.get(), invalid, rejected, [](const auto &) {}, error) &&
                    empty_output(rejected),
                "Observer input failure retained previous valid output");
        rejected = ordinary;
        size_t rejected_callbacks = 0;
        require(!josev2b_head::test::run_observed(
                    model.get(), values, rejected,
                    [&](const auto &) {
                        ++rejected_callbacks;
                        throw std::runtime_error("owned observer rejection control");
                    },
                    error) &&
                    rejected_callbacks == 1 && error == "owned observer rejection control" && empty_output(rejected),
                "Observer exception published a computed result");
        josev2b_head::output recovered;
        require(josev2b_head::run(model.get(), values, recovered, error) && exact(ordinary, recovered),
                "Ordinary output changed after input/observer failures");
        std::ofstream proof(out / "probe.json");
        proof << "{\"completed\":true,\"full_model\":false,\"qualification\":false,\"backend\":\""
              << core_json::json_escape_strict(ordinary.backend) << "\",\"stages\":" << sequence
              << ",\"observer_bit_exact\":true,\"repeat_bit_exact\":true,\"input_failure_output_empty\":true,"
                 "\"observer_failure_output_empty\":true,\"post_failure_recovery_bit_exact\":true,\"ordinary_seconds\":"
              << std::chrono::duration<double>(ordinary_end - begin).count()
              << ",\"observed_seconds\":" << std::chrono::duration<double>(observed_end - ordinary_end).count()
              << ",\"fields\":[";
        for (size_t i = 0; i < ordinary.fields.size(); ++i) {
            const auto & field = ordinary.fields[i];
            const auto prefix = std::string("field-") + std::to_string(i);
            raw(out / (prefix + "-ordinary-logits.f32"), field.logits);
            raw(out / (prefix + "-ordinary-probabilities.f32"), field.probabilities);
            raw(out / (prefix + "-repeat-logits.f32"), repeat.fields[i].logits);
            raw(out / (prefix + "-repeat-probabilities.f32"), repeat.fields[i].probabilities);
            proof << (i ? "," : "") << "{\"id\":\"" << core_json::json_escape_strict(field.id) << "\",\"raw_prefix\":\""
                  << prefix << "\",\"logits\":[";
            for (size_t j = 0; j < field.logits.size(); ++j) {
                char value[64];
                std::snprintf(value, sizeof(value), "%.9g", field.logits[j]);
                proof << (j ? "," : "") << value;
            }
            proof << "],\"probabilities\":[";
            for (size_t j = 0; j < field.probabilities.size(); ++j) {
                char value[64];
                std::snprintf(value, sizeof(value), "%.9g", field.probabilities[j]);
                proof << (j ? "," : "") << value;
            }
            proof << "]}";
        }
        proof << "]}\n";
        require(bool(proof), "Probe receipt write failed");
        std::printf("{\"completed\":true,\"observer_bit_exact\":true,\"repeat_bit_exact\":true,\"stages\":%zu}\n",
                    sequence);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Josev head probe: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
