// Model-free ABI/admission controls using explicitly synthetic zero weights.
// Passing these tests does not establish trained-model inference parity.
#include "core/clean_exit.h"
#include "core/sha256.h"
#include "core/temp_file.h"
#include "ggml.h"
#include "gguf.h"
#include "spacy_pipeline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int assertions = 0;

void check(bool condition, const char * message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}

void gate(const char * value) {
#if defined(_WIN32)
    _putenv_s("CRISPEMBED_SPACY", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_SPACY", value, 1);
    else
        unsetenv("CRISPEMBED_SPACY");
#endif
}

struct GateRestore {
    bool existed = std::getenv("CRISPEMBED_SPACY") != nullptr;
    std::string original = existed ? std::getenv("CRISPEMBED_SPACY") : "";
    ~GateRestore() { gate(existed ? original.c_str() : nullptr); }
};

struct TemporaryFile {
    std::string path = core_tmp::make_private(".gguf");
    TemporaryFile() { check(!path.empty(), "Cannot create private synthetic fixture"); }
    ~TemporaryFile() { std::remove(path.c_str()); }
    void write(const std::vector<uint8_t> & bytes) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        check(static_cast<bool>(out), "Cannot write synthetic fixture");
    }
};

using Context = std::unique_ptr<crispembed_spacy_context, decltype(&crispembed_spacy_free)>;
using Result = std::unique_ptr<crispembed_spacy_result, decltype(&crispembed_spacy_result_free)>;

std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    check(static_cast<bool>(input), "Cannot read synthetic fixture");
    const auto size = input.tellg();
    std::vector<uint8_t> data(static_cast<size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(data.data()), size);
    check(static_cast<bool>(input), "Synthetic fixture truncated");
    return data;
}

// gguf's independent writer supplies a complete structural fixture. It carries
// the closed profile's metadata, but its all-zero weights are NOT source-model
// weights. The test, as trusted host, supplies this synthetic payload's digest.
std::vector<uint8_t> synthetic_model(const std::string & fault = "") {
    std::map<std::string, std::vector<int64_t>> shapes;
    const int64_t rows[] = { 5000, 1000, 2500, 2500, 50, 50 };
    for (int i = 0; i < 6; ++i) shapes["embed." + std::to_string(i) + ".E"] = { 96, rows[i] };
    shapes["embed.mix.W"] = { 576, 3, 96 };
    shapes["embed.mix.b"] = { 3, 96 };
    shapes["embed.norm.G"] = { 96 };
    shapes["embed.norm.b"] = { 96 };
    for (int i = 0; i < 4; ++i) {
        const auto name = "encoder." + std::to_string(i);
        shapes[name + ".maxout.W"] = { 288, 3, 96 };
        shapes[name + ".maxout.b"] = { 3, 96 };
        shapes[name + ".norm.G"] = { 96 };
        shapes[name + ".norm.b"] = { 96 };
    }
    for (const auto & item :
         std::map<std::string, int64_t>{ { "tagger", 55 }, { "morphologizer", 671 }, { "lemmatizer", 1311 } }) {
        shapes[item.first + ".W"] = { 96, item.second };
        shapes[item.first + ".b"] = { item.second };
    }
    shapes["parser.reduce.W"] = { 96, 64 };
    shapes["parser.reduce.b"] = { 64 };
    shapes["parser.lower.W"] = { 64, 2, 64, 8 };
    shapes["parser.lower.b"] = { 2, 64 };
    shapes["parser.lower.pad"] = { 2, 64, 8, 1 };
    shapes["parser.upper.W"] = { 64, 137 };
    shapes["parser.upper.b"] = { 137 };
    if (fault == "missing") shapes.erase("tagger.b");
    if (fault == "shape") shapes["tagger.b"] = { 54 };
    auto * gguf = gguf_init_empty();
    auto * ggml = ggml_init({ 64 * ggml_tensor_overhead() + 1024, nullptr, true });
    check(gguf && ggml, "Cannot allocate synthetic GGUF metadata");
    gguf_set_val_str(gguf, "general.architecture", "spacy_de_core_news_sm");
    if (fault == "type")
        gguf_set_val_i32(gguf, "spacy.format_version", 1);
    else
        gguf_set_val_u32(gguf, "spacy.format_version", 1);
    gguf_set_val_str(gguf, "spacy.model", "de_core_news_sm");
    gguf_set_val_str(gguf, "spacy.model_version", "3.8.0");
    gguf_set_val_str(gguf, "spacy.source.sha256",
                     fault == "source" ? "unknown"
                                       : "fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7");
    gguf_set_val_str(gguf, "spacy.resources.sha256",
                     "b42496cc373ada57a67ed2f4c2c25f8d22f80c08db5841a7cdb91c8a6e18562a");
    gguf_set_val_str(gguf, "spacy.components", "tok2vec,tagger,morphologizer,parser,lemmatizer,attribute_ruler");
    gguf_set_val_str(gguf, "spacy.feature_order", "NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE");
    gguf_set_val_str(gguf, "spacy.precision", "f32");
    gguf_set_val_u32(gguf, "spacy.width", 96);
    gguf_set_val_u32(gguf, "spacy.encoder_depth", 4);
    gguf_set_val_u32(gguf, "spacy.encoder_padding", 4);
    gguf_set_val_f32(gguf, "spacy.layer_norm_epsilon", 1e-8f);
    std::vector<std::vector<float>> weights;
    weights.reserve(shapes.size());
    for (const auto & item : shapes) {
        size_t elements = 1;
        for (const auto dim : item.second) elements *= static_cast<size_t>(dim);
        weights.emplace_back(elements, 0.0f);
        if (item.first == "embed.0.E") {
            if (fault == "nonfinite") weights.back()[0] = std::numeric_limits<float>::quiet_NaN();
            if (fault == "other-model") weights.back()[0] = 0.125f;
        }
        const auto type = fault == "precision" && item.first == "embed.0.E" ? GGML_TYPE_F16 : GGML_TYPE_F32;
        auto * tensor = ggml_new_tensor(ggml, type, static_cast<int>(item.second.size()), item.second.data());
        check(tensor != nullptr, "Cannot construct synthetic tensor");
        ggml_set_name(tensor, item.first.c_str());
        tensor->data = weights.back().data();
        gguf_add_tensor(gguf, tensor);
    }
    TemporaryFile file;
    const size_t data_offset = gguf_get_meta_size(gguf);
    const bool wrote = gguf_write_to_file(gguf, file.path.c_str(), false);
    gguf_free(gguf);
    ggml_free(ggml);
    check(wrote, "Cannot serialize synthetic GGUF");
    const auto bytes = read_bytes(file.path);
    // The C writer trims trailing dimensions of size one. The frozen Python
    // converter preserves parser.lower.pad's explicit [2,64,8,1] GGUF rank.
    // Restore that representation in this independently generated fixture.
    const auto & last = *shapes.rbegin();
    const auto last_name =
        std::search(bytes.begin(), bytes.begin() + data_offset, last.first.begin(), last.first.end());
    check(last_name != bytes.begin() + data_offset, "Synthetic final tensor missing");
    const size_t metadata_end =
        static_cast<size_t>(last_name - bytes.begin()) + last.first.size() + 4 + 8 * last.second.size() + 4 + 8;
    std::vector<uint8_t> metadata(bytes.begin(), bytes.begin() + metadata_end);
    const std::string pad_name = "parser.lower.pad";
    const auto pad = std::search(metadata.begin(), metadata.end(), pad_name.begin(), pad_name.end());
    check(pad != metadata.end(), "Synthetic parser padding missing");
    const size_t rank_offset = static_cast<size_t>(pad - metadata.begin()) + pad_name.size();
    check(metadata[rank_offset] == 3, "Unexpected synthetic parser padding rank");
    metadata[rank_offset] = 4;
    const size_t added_dimension = rank_offset + 4 + 3 * 8;
    metadata.insert(metadata.begin() + added_dimension, 8, 0);
    metadata[added_dimension] = 1;
    metadata.resize((metadata.size() + 31) & ~size_t(31), 0);
    metadata.insert(metadata.end(), bytes.begin() + data_offset, bytes.end());
    return metadata;
}

void reject_bytes(const std::vector<uint8_t> & bytes, const char * reason) {
    TemporaryFile file;
    file.write(bytes);
    char error[256] = {};
    const auto sha = core_sha256::bytes(bytes.data(), bytes.size());
    Context context(crispembed_spacy_create(file.path.c_str(), sha.c_str(), error, sizeof(error)),
                    crispembed_spacy_free);
    check(!context && error[0], reason);
}

int always_cancel(void *) {
    return 1;
}
int abort_trace(void *, const char *, const float *, const size_t *, size_t) {
    return 1;
}

struct CancelAfterStage {
    std::atomic<bool> cancelled{ false };
};

int cancel_after_trace(void * user, const char * stage, const float *, const size_t *, size_t) {
    auto & state = *static_cast<CancelAfterStage *>(user);
    if (std::strcmp(stage, "tok2vec.encoder.layer.0.residual") == 0) state.cancelled.store(true);
    return 0;
}

int check_cancelled(void * user) {
    return static_cast<CancelAfterStage *>(user)->cancelled.load() ? 1 : 0;
}

int forward(crispembed_spacy_context * context, const uint64_t * features, size_t tokens, size_t count,
            crispembed_spacy_result ** result) {
    return crispembed_spacy_forward_features(context, features, tokens, count, result, nullptr, nullptr, nullptr,
                                             nullptr, nullptr, 0);
}

void test_admission(const std::vector<uint8_t> & valid) {
    for (const char * value : { static_cast<const char *>(nullptr), "", "0", "false", "OFF", "No" }) {
        gate(value);
        char error[256] = {};
        Context context(crispembed_spacy_create("unused.gguf", nullptr, error, sizeof(error)), crispembed_spacy_free);
        check(!context && std::strstr(error, "disabled"), "Off spelling enabled the spaCy backend");
    }
    gate("1");
    char tiny[1] = { 'x' };
    check(!crispembed_spacy_create(nullptr, nullptr, tiny, sizeof(tiny)) && tiny[0] == '\0',
          "Tiny error buffer was not terminated");
    check(!crispembed_spacy_create("unused", "A", nullptr, 0), "Invalid SHA admitted");
    check(!crispembed_spacy_create("unused", nullptr, nullptr, 1), "Invalid error pointer/capacity admitted");
    TemporaryFile file;
    file.write(valid);
    const std::string wrong_sha(64, '0');
    char error[256] = {};
    check(!crispembed_spacy_create(file.path.c_str(), wrong_sha.c_str(), error, sizeof(error)) &&
              std::strstr(error, "SHA-256"),
          "Untrusted hash admitted");
    for (const auto & fault : { "missing", "shape", "type", "source", "nonfinite", "precision" }) {
        reject_bytes(synthetic_model(fault), "Malformed synthetic profile admitted");
    }
    auto bytes = valid;
    bytes.pop_back();
    reject_bytes(bytes, "Truncated GGUF admitted");
    bytes = valid;
    bytes.push_back(0);
    reject_bytes(bytes, "GGUF trailing payload admitted");
    bytes = valid;
    bytes[4] = 2;
    reject_bytes(bytes, "Unknown GGUF version admitted");
    bytes = valid;
    bytes[8] = 0xff;
    reject_bytes(bytes, "Malformed tensor count admitted");
    bytes = valid;
    const std::string original = "embed.1.E";
    const auto found = std::search(bytes.begin(), bytes.end(), original.begin(), original.end());
    check(found != bytes.end(), "Synthetic tensor name missing");
    *(found + 6) = '0';
    reject_bytes(bytes, "Duplicate tensor name admitted");
}

void test_inference_contract(const std::vector<uint8_t> & valid) {
    gate("1");
    TemporaryFile file;
    file.write(valid);
    const auto sha = core_sha256::bytes(valid.data(), valid.size());
    char error[256] = {};
    Context context(crispembed_spacy_create(file.path.c_str(), sha.c_str(), error, sizeof(error)),
                    crispembed_spacy_free);
    check(context != nullptr, error[0] ? error : "Synthetic model load failed");
    // The model is copied; replacing the file after successful admission must
    // not mutate an existing context's owned weights.
    file.write(std::vector<uint8_t>{ 0 });
    std::array<uint64_t, 6> features{ 1, 2, 3, 4, 1, 0 };
    auto * raw = reinterpret_cast<crispembed_spacy_result *>(1);
    check(forward(nullptr, features.data(), 1, 6, &raw) == CRISPEMBED_SPACY_INVALID_ARGUMENT && !raw,
          "Null-context result leaked");
    check(forward(context.get(), features.data(), 2049, 2049 * 6, &raw) == CRISPEMBED_SPACY_INVALID_ARGUMENT && !raw,
          "Oversized input admitted");
    check(forward(context.get(), nullptr, 1, 6, &raw) == CRISPEMBED_SPACY_INVALID_ARGUMENT && !raw,
          "Null features admitted");
    check(forward(context.get(), features.data(), 1, 5, &raw) == CRISPEMBED_SPACY_INVALID_ARGUMENT && !raw,
          "Mismatched feature count admitted");
    features[5] = 2;
    check(forward(context.get(), features.data(), 1, 6, &raw) == CRISPEMBED_SPACY_INVALID_ARGUMENT && !raw,
          "Non-Boolean lexical ID admitted");
    features[5] = 0;
    check(crispembed_spacy_forward_features(context.get(), features.data(), 1, 6, &raw, nullptr, nullptr, always_cancel,
                                            nullptr, error, sizeof(error)) == CRISPEMBED_SPACY_CANCELLED &&
              !raw,
          "Cancelled request published a result");
    check(crispembed_spacy_forward_features(context.get(), features.data(), 1, 6, &raw, abort_trace, nullptr, nullptr,
                                            nullptr, error, sizeof(error)) == CRISPEMBED_SPACY_TRACE_ERROR &&
              !raw,
          "Aborted trace published a result");
    CancelAfterStage mid_cancel;
    check(crispembed_spacy_forward_features(context.get(), features.data(), 1, 6, &raw, cancel_after_trace, &mid_cancel,
                                            check_cancelled, &mid_cancel, error,
                                            sizeof(error)) == CRISPEMBED_SPACY_CANCELLED &&
              !raw,
          "Mid-graph cancellation published a result");
    check(forward(context.get(), nullptr, 0, 0, &raw) == CRISPEMBED_SPACY_OK && raw, "Empty result rejected");
    Result empty(raw, crispembed_spacy_result_free);
    const float * values = reinterpret_cast<const float *>(1);
    size_t rows = 1, columns = 1;
    check(crispembed_spacy_result_matrix(empty.get(), "tok2vec", &values, &rows, &columns) == CRISPEMBED_SPACY_OK &&
              !values && rows == 0 && columns == 96,
          "Empty matrix contract differs");
    check(forward(context.get(), features.data(), 1, 6, &raw) == CRISPEMBED_SPACY_OK && raw,
          "Recovery inference failed");
    Result result(raw, crispembed_spacy_result_free);
    check(crispembed_spacy_result_matrix(result.get(), "tagger", &values, &rows, &columns) == CRISPEMBED_SPACY_OK &&
              rows == 1 && columns == 55 && values,
          "Classifier output shape differs");
    check(std::all_of(values, values + columns, [](float value) { return value == 0.0f; }),
          "Synthetic zero model returned nonzero scores");
    check(crispembed_spacy_result_matrix(result.get(), "unsupported", &values, &rows, &columns) ==
                  CRISPEMBED_SPACY_INVALID_ARGUMENT &&
              !values && rows == 0 && columns == 0,
          "Invalid matrix query returned stale pointers");
    std::array<int32_t, 8> state{ 0, -1, -1, -1, -1, -1, -1, -1 };
    std::array<float, 137> scores;
    scores.fill(42.0f);
    check(crispembed_spacy_parser_scores(context.get(), result.get(), state.data(), state.size(), scores.data(), 136,
                                         nullptr, nullptr, nullptr, nullptr, error,
                                         sizeof(error)) == CRISPEMBED_SPACY_INVALID_ARGUMENT,
          "Short parser output admitted");
    state[0] = 1;
    check(crispembed_spacy_parser_scores(context.get(), result.get(), state.data(), state.size(), scores.data(),
                                         scores.size(), nullptr, nullptr, nullptr, nullptr, error,
                                         sizeof(error)) == CRISPEMBED_SPACY_INVALID_ARGUMENT,
          "Out-of-range parser feature admitted");
    state[0] = 0;
    check(crispembed_spacy_parser_scores(context.get(), result.get(), state.data(), state.size(), scores.data(),
                                         scores.size(), abort_trace, nullptr, nullptr, nullptr, error,
                                         sizeof(error)) == CRISPEMBED_SPACY_TRACE_ERROR,
          "Parser trace abortion ignored");
    check(std::all_of(scores.begin(), scores.end(), [](float value) { return value == 42.0f; }),
          "Failed parser call changed caller output");
    check(crispembed_spacy_parser_scores(context.get(), result.get(), state.data(), state.size(), scores.data(),
                                         scores.size(), nullptr, nullptr, nullptr, nullptr, error,
                                         sizeof(error)) == CRISPEMBED_SPACY_OK,
          "Parser scorer recovery failed");
    check(std::all_of(scores.begin(), scores.end(), [](float value) { return value == 0.0f; }),
          "Synthetic parser scores differ");
    const auto other = synthetic_model("other-model");
    TemporaryFile other_file;
    other_file.write(other);
    const auto other_sha = core_sha256::bytes(other.data(), other.size());
    Context other_context(crispembed_spacy_create(other_file.path.c_str(), other_sha.c_str(), error, sizeof(error)),
                          crispembed_spacy_free);
    check(other_context != nullptr, "Second synthetic model failed");
    check(crispembed_spacy_parser_scores(other_context.get(), result.get(), state.data(), state.size(), scores.data(),
                                         scores.size(), nullptr, nullptr, nullptr, nullptr, error,
                                         sizeof(error)) == CRISPEMBED_SPACY_INVALID_ARGUMENT,
          "Result from another model was accepted");
    gate("0");
    check(forward(context.get(), features.data(), 1, 6, &raw) == CRISPEMBED_SPACY_DISABLED && !raw,
          "Disabling an open context did not stop new inference");
    gate("1");
    std::atomic<int> concurrent_success{ 0 };
    auto worker = [&]() {
        crispembed_spacy_result * owned = nullptr;
        if (forward(context.get(), features.data(), 1, 6, &owned) == CRISPEMBED_SPACY_OK) ++concurrent_success;
        crispembed_spacy_result_free(owned);
    };
    std::thread first(worker), second(worker);
    first.join();
    second.join();
    check(concurrent_success == 2, "Serialized concurrent callers failed");
    context.reset();
    check(crispembed_spacy_result_matrix(result.get(), "lemmatizer", &values, &rows, &columns) == CRISPEMBED_SPACY_OK &&
              rows == 1 && columns == 1311 && values,
          "Result did not outlive its context");
    check(std::all_of(values, values + columns, [](float value) { return value == 0.0f; }),
          "Result ownership corrupted after context destruction");
    crispembed_spacy_free(nullptr);
    crispembed_spacy_result_free(nullptr);
}

int crispembed_test_main() {
    GateRestore restore;
    try {
        check(crispembed_spacy_abi_version() == 1, "ABI version differs");
        check(crispembed_spacy_capabilities() == 3, "Numeric capability declaration differs");
        const auto valid = synthetic_model();
        test_admission(valid);
        test_inference_contract(valid);
        std::printf(
            "PASS: %d model-free spaCy ABI/admission assertions; synthetic weights, no inference qualification\n",
            assertions);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "FAIL after %d assertions: %s\n", assertions, exception.what());
        return 1;
    }
}

} // namespace

int main() {
    core_util::clean_exit(crispembed_test_main());
}
