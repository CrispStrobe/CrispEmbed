// Closed CPU/F32 backend for de_core_news_sm 3.8.0.
#include "spacy_pipeline.h"

#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"
#include "ggml-cpu.h"
#include "spacy_ggml.h"
#include "spacy_numeric.h"
#include "spacy_lexical.h"
#include "spacy_decode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t k_max_model_bytes = 32u * 1024u * 1024u;
constexpr size_t k_alignment = 32;
constexpr const char * k_source_sha = "fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7";
constexpr const char * k_resources_sha = "b42496cc373ada57a67ed2f4c2c25f8d22f80c08db5841a7cdb91c8a6e18562a";
// Quantized derivatives (CRISPEMBED_SPACY_QUANT) must name this exact admitted
// F32 export as their source; models/quantize-spacy-gguf.py writes them.
constexpr const char * k_f32_model_sha = "fae2e6ab8d6c611d7a9dddd4b3259a871a1d2f95cacc8259f49c1e4ff31ff6cd";
constexpr const char * k_quant_policy = "spacy-matrices-v1";

// Policy spacy-matrices-v1: only the large matrices may be stored below F32.
// Biases, LayerNorm and the per-step parser tensors (lower.b, lower.pad,
// upper.*) always stay F32 so that parser state scoring is unchanged.
bool quantizable_tensor(const std::string & name) {
    static const std::set<std::string> names = {
        "embed.0.E", "embed.1.E", "embed.2.E", "embed.3.E", "embed.4.E", "embed.5.E", "embed.mix.W",
        "encoder.0.maxout.W", "encoder.1.maxout.W", "encoder.2.maxout.W", "encoder.3.maxout.W",
        "tagger.W", "morphologizer.W", "lemmatizer.W", "parser.reduce.W", "parser.lower.W" };
    return names.count(name) != 0;
}

const std::map<std::string, ggml_type> & quant_precisions() {
    static const std::map<std::string, ggml_type> precisions = {
        { "f16", GGML_TYPE_F16 }, { "bf16", GGML_TYPE_BF16 }, { "q8_0", GGML_TYPE_Q8_0 }, { "q5_0", GGML_TYPE_Q5_0 }, { "q4_0", GGML_TYPE_Q4_0 } };
    return precisions;
}

bool quant_type_admitted(uint32_t type) {
    for (const auto & item : quant_precisions())
        if (static_cast<uint32_t>(item.second) == type) return true;
    return false;
}

using Shapes = std::map<std::string, std::vector<size_t>>;

Shapes expected_shapes() {
    Shapes shapes;
    const size_t rows[] = { 5000, 1000, 2500, 2500, 50, 50 };
    for (size_t i = 0; i < 6; ++i) shapes["embed." + std::to_string(i) + ".E"] = { rows[i], 96 };
    shapes["embed.mix.W"] = { 96, 3, 576 };
    shapes["embed.mix.b"] = { 96, 3 };
    shapes["embed.norm.G"] = { 96 };
    shapes["embed.norm.b"] = { 96 };
    for (size_t i = 0; i < 4; ++i) {
        const std::string prefix = "encoder." + std::to_string(i);
        shapes[prefix + ".maxout.W"] = { 96, 3, 288 };
        shapes[prefix + ".maxout.b"] = { 96, 3 };
        shapes[prefix + ".norm.G"] = { 96 };
        shapes[prefix + ".norm.b"] = { 96 };
    }
    for (const auto & item :
         std::map<std::string, size_t>{ { "tagger", 55 }, { "morphologizer", 671 }, { "lemmatizer", 1311 } }) {
        shapes[item.first + ".W"] = { item.second, 96 };
        shapes[item.first + ".b"] = { item.second };
    }
    shapes["parser.reduce.W"] = { 64, 96 };
    shapes["parser.reduce.b"] = { 64 };
    shapes["parser.lower.W"] = { 8, 64, 2, 64 };
    shapes["parser.lower.b"] = { 64, 2 };
    shapes["parser.lower.pad"] = { 1, 8, 64, 2 };
    shapes["parser.upper.W"] = { 137, 64 };
    shapes["parser.upper.b"] = { 137 };
    return shapes;
}

void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

void error_text(char * error, size_t capacity, const char * message) noexcept {
    if (!error || !capacity) return;
    size_t length = 0;
    while (length + 1 < capacity && message[length]) ++length;
    if (length) std::memcpy(error, message, length);
    error[length] = '\0';
}

bool valid_sha(const char * text) {
    if (!text) return false;
    for (size_t i = 0; i < 64; ++i) {
        const char value = text[i];
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) return false;
    }
    return text[64] == '\0';
}

size_t aligned(size_t size) {
    require(size <= k_max_model_bytes, "GGUF allocation bound exceeded");
    return (size + k_alignment - 1) & ~(k_alignment - 1);
}

// The closed profile is preflighted before handing any metadata to ggml.
// This prevents untrusted dimensions/counts from reaching allocator/assert paths.
class Reader {
public:
    explicit Reader(const std::vector<uint8_t> & bytes) : bytes_(bytes) {}
    uint32_t u32() {
        need(4);
        uint32_t value = 0;
        for (size_t i = 0; i < 4; ++i) value |= static_cast<uint32_t>(bytes_[offset_++]) << (8 * i);
        return value;
    }
    uint64_t u64() {
        need(8);
        uint64_t value = 0;
        for (size_t i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes_[offset_++]) << (8 * i);
        return value;
    }
    std::string string(size_t limit) {
        const uint64_t length = u64();
        require(length <= limit, "Oversized GGUF string");
        need(static_cast<size_t>(length));
        const std::string result(reinterpret_cast<const char *>(bytes_.data() + offset_), static_cast<size_t>(length));
        require(result.find('\0') == std::string::npos, "NUL in GGUF metadata");
        offset_ += static_cast<size_t>(length);
        return result;
    }
    size_t position() const { return offset_; }

private:
    void need(size_t count) {
        require(offset_ <= bytes_.size() && count <= bytes_.size() - offset_, "Truncated GGUF metadata");
    }
    const std::vector<uint8_t> & bytes_;
    size_t offset_ = 0;
};

struct Extent {
    std::vector<size_t> shape;
    size_t offset;
    size_t bytes;
    ggml_type type;
};

struct Profile {
    size_t data_offset;
    std::string precision;
    std::map<std::string, Extent> tensors;
};

// Dequantizes one admitted tensor (any type) to F32 and rejects nonfinite values.
std::vector<float> dequantized_f32(const uint8_t * data, const Extent & extent) {
    size_t elements = 1;
    for (auto dimension : extent.shape) elements *= dimension;
    std::vector<float> values(elements);
    if (extent.type == GGML_TYPE_F32) {
        std::memcpy(values.data(), data, elements * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(extent.type);
        require(traits && traits->to_float, "spaCy tensor type cannot be dequantized");
        traits->to_float(data, values.data(), static_cast<int64_t>(elements));
    }
    for (const float value : values) require(std::isfinite(value), "Nonfinite spaCy tensor");
    return values;
}

Profile preflight(const std::vector<uint8_t> & bytes) {
    require(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE-754 float32 required");
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "Little-endian host required by this profile");
    Reader reader(bytes);
    require(reader.u32() == 0x46554747u, "Invalid GGUF magic");
    require(reader.u32() == 3, "Only GGUF version 3 is supported");
    const auto shapes = expected_shapes();
    require(reader.u64() == shapes.size(), "Expected exactly 39 spaCy tensors");
    // 13 keys for the admitted F32 export; quantized derivatives add exactly
    // spacy.quantization.policy and spacy.quantization.source_gguf_sha256.
    const uint64_t metadata_count = reader.u64();
    require(metadata_count == 13 || metadata_count == 15, "Unexpected spaCy GGUF metadata inventory");
    const std::map<std::string, std::string> strings = {
        { "general.architecture", "spacy_de_core_news_sm" },
        { "spacy.model", "de_core_news_sm" },
        { "spacy.model_version", "3.8.0" },
        { "spacy.source.sha256", k_source_sha },
        { "spacy.resources.sha256", k_resources_sha },
        { "spacy.components", "tok2vec,tagger,morphologizer,parser,lemmatizer,attribute_ruler" },
        { "spacy.feature_order", "NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE" }
    };
    const std::map<std::string, std::string> quant_strings = {
        { "spacy.quantization.policy", k_quant_policy },
        { "spacy.quantization.source_gguf_sha256", k_f32_model_sha } };
    Profile profile{};
    const std::map<std::string, uint32_t> integers = { { "spacy.format_version", 1 },
                                                       { "spacy.width", 96 },
                                                       { "spacy.encoder_depth", 4 },
                                                       { "spacy.encoder_padding", 4 } };
    std::set<std::string> seen_keys;
    for (size_t i = 0; i < metadata_count; ++i) {
        const auto key = reader.string(128);
        require(seen_keys.insert(key).second, "Duplicate GGUF metadata key");
        const uint32_t type = reader.u32();
        const auto text = strings.find(key);
        const auto integer = integers.find(key);
        const auto quant_text = quant_strings.find(key);
        if (text != strings.end()) {
            require(type == GGUF_TYPE_STRING, "Mistyped required GGUF string");
            require(reader.string(256) == text->second, "Unsupported spaCy GGUF identity or configuration");
        } else if (key == "spacy.precision") {
            require(type == GGUF_TYPE_STRING, "Mistyped required GGUF string");
            profile.precision = reader.string(16);
            require(profile.precision == "f32" || profile.precision == "mixed" ||
                        quant_precisions().count(profile.precision),
                    "Unsupported spaCy GGUF precision");
        } else if (quant_text != quant_strings.end()) {
            require(type == GGUF_TYPE_STRING, "Mistyped required GGUF string");
            require(reader.string(256) == quant_text->second, "Unsupported spaCy quantization policy or source");
        } else if (integer != integers.end()) {
            require(type == GGUF_TYPE_UINT32, "Mistyped required GGUF uint32");
            require(reader.u32() == integer->second, "Unsupported spaCy GGUF dimension or schema");
        } else if (key == "spacy.layer_norm_epsilon") {
            require(type == GGUF_TYPE_FLOAT32, "Mistyped LayerNorm epsilon");
            const uint32_t raw = reader.u32();
            float epsilon;
            std::memcpy(&epsilon, &raw, sizeof(epsilon));
            require(epsilon == 1e-8f, "Unsupported LayerNorm epsilon");
        } else {
            throw std::invalid_argument("Unexpected GGUF metadata key");
        }
    }
    // Completeness is explicit: every identity/schema key must be present, and the
    // two quantization keys exactly when the precision is not f32.
    for (const auto & item : strings) require(seen_keys.count(item.first) != 0, "Missing spaCy GGUF metadata key");
    for (const auto & item : integers) require(seen_keys.count(item.first) != 0, "Missing spaCy GGUF metadata key");
    require(seen_keys.count("spacy.layer_norm_epsilon") != 0 && seen_keys.count("spacy.precision") != 0,
            "Missing spaCy GGUF metadata key");
    const bool quantized = profile.precision != "f32";
    for (const auto & item : quant_strings)
        require(seen_keys.count(item.first) == (quantized ? 1u : 0u),
                "spaCy precision and quantization metadata disagree");
    require(!quantized || core_env::on("CRISPEMBED_SPACY_QUANT"),
            "Quantized spaCy GGUF requires CRISPEMBED_SPACY_QUANT=1");
    std::set<ggml_type> matrix_types;
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto name = reader.string(63);
        const auto found = shapes.find(name);
        require(found != shapes.end(), "Unknown spaCy tensor name");
        require(profile.tensors.find(name) == profile.tensors.end(), "Duplicate spaCy tensor name");
        const auto & shape = found->second;
        require(reader.u32() == shape.size(), "Incorrect spaCy tensor rank");
        size_t elements = 1;
        for (auto dimension = shape.rbegin(); dimension != shape.rend(); ++dimension) {
            require(reader.u64() == *dimension, "Incorrect spaCy tensor shape");
            elements *= *dimension; // dimensions are fixed by the closed profile
        }
        const uint32_t raw_type = reader.u32();
        const bool matrix = quantizable_tensor(name);
        require(raw_type == GGML_TYPE_F32 || (quantized && matrix && quant_type_admitted(raw_type)),
                quantized ? "spaCy tensor type outside quantization policy" : "Only F32 spaCy tensors are admitted");
        const auto tensor_type = static_cast<ggml_type>(raw_type);
        if (matrix) matrix_types.insert(tensor_type);
        const size_t row = shape.back();
        require(row % static_cast<size_t>(ggml_blck_size(tensor_type)) == 0, "spaCy tensor row not block aligned");
        const uint64_t offset = reader.u64();
        const size_t tensor_bytes = ggml_row_size(tensor_type, static_cast<int64_t>(row)) * (elements / row);
        require(offset <= k_max_model_bytes && offset % k_alignment == 0, "Invalid GGUF tensor offset");
        require(tensor_bytes <= k_max_model_bytes - static_cast<size_t>(offset), "GGUF tensor extent exceeds bound");
        profile.tensors.emplace(name, Extent{ shape, static_cast<size_t>(offset), tensor_bytes, tensor_type });
        ranges.emplace_back(static_cast<size_t>(offset), aligned(static_cast<size_t>(offset) + tensor_bytes));
    }
    // A single-type precision stores every policy matrix in that type; "mixed"
    // must actually mix at least two matrix types.
    if (profile.precision == "mixed") {
        require(matrix_types.size() >= 2, "Mixed spaCy precision requires at least two matrix types");
    } else if (quantized) {
        require(matrix_types.size() == 1 && *matrix_types.begin() == quant_precisions().at(profile.precision),
                "spaCy quantized tensor types disagree with precision");
    }
    profile.data_offset = aligned(reader.position());
    require(profile.data_offset <= bytes.size(), "Truncated GGUF metadata alignment");
    for (size_t i = reader.position(); i < profile.data_offset; ++i)
        require(bytes[i] == 0, "Nonzero GGUF header padding");
    std::sort(ranges.begin(), ranges.end());
    size_t end = 0;
    for (const auto & range : ranges) {
        require(range.first == end, "GGUF tensor extents overlap or leave a gap");
        end = range.second;
    }
    require(end == bytes.size() - profile.data_offset, "Truncated GGUF tensors or unexpected trailing bytes");
    for (const auto & item : profile.tensors) {
        const auto & extent = item.second;
        const size_t begin = profile.data_offset + extent.offset;
        if (extent.type == GGML_TYPE_F32) {
            for (size_t i = 0; i < extent.bytes; i += sizeof(float)) {
                float value;
                std::memcpy(&value, bytes.data() + begin + i, sizeof(value));
                require(std::isfinite(value), "Nonfinite spaCy tensor");
            }
        } else {
            (void) dequantized_f32(bytes.data() + begin, extent);
        }
        for (size_t i = extent.bytes; i < aligned(extent.bytes); ++i) {
            require(bytes[begin + i] == 0, "Nonzero GGUF tensor padding");
        }
    }
    return profile;
}

std::vector<uint8_t> read_model(const char * path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(stream), "Cannot open spaCy GGUF file");
    const auto length = stream.tellg();
    require(length >= 24 && length <= static_cast<std::streamoff>(k_max_model_bytes),
            "GGUF file size outside admitted bounds");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream) && stream.peek() == std::char_traits<char>::eof(),
            "Cannot read stable complete GGUF file");
    return bytes;
}

std::string read_resources(const char * path, const char * expected_sha256) {
    require(path && *path && valid_sha(expected_sha256),
            "Explicit resource path and trusted lowercase SHA-256 required");
    require(std::strcmp(expected_sha256, k_resources_sha) == 0,
            "Resource hash differs from the admitted model profile");
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(stream), "Cannot open spaCy resources file");
    const auto length = stream.tellg();
    require(length > 0 && length <= static_cast<std::streamoff>(k_max_model_bytes),
            "Resource file size outside admitted bounds");
    std::string bytes(static_cast<size_t>(length), '\0');
    stream.seekg(0);
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream) && stream.peek() == std::char_traits<char>::eof(),
            "Cannot read complete spaCy resources");
    require(core_sha256::bytes(bytes.data(), bytes.size()) == expected_sha256, "Resource SHA-256 mismatch");
    return bytes;
}

class TraceAborted : public std::runtime_error {
public:
    TraceAborted() : std::runtime_error("spaCy trace callback requested abortion") {}
};

crispembed::spacy::TraceSink trace_sink(crispembed_spacy_trace_callback callback, void * user) {
    if (!callback) return {};
    return [callback, user](const std::string & stage, const crispembed::spacy::TensorView & tensor) {
        if (callback(user, stage.c_str(), tensor.data, tensor.shape.data(), tensor.shape.size())) throw TraceAborted();
    };
}

crispembed::spacy::CancelCheck cancellation(crispembed_spacy_cancel_callback callback, void * user) {
    if (!callback) return {};
    return [callback, user]() { return callback(user) != 0; };
}

// CRISPEMBED_SPACY_THREADS: ggml CPU threads for the opt-in graph engine.
// Unset/invalid keeps the historical single thread; values clamp to [1, 64].
int spacy_threads() {
    const char * value = std::getenv("CRISPEMBED_SPACY_THREADS");
    if (!value || !*value) return 1;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed < 1) return 1;
    return static_cast<int>(std::min<long>(parsed, 64));
}

void check_cancel(const crispembed::spacy::CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw crispembed::spacy::NumericCancelled();
}

} // namespace

struct crispembed_spacy_context {
    std::mutex mutex;
    std::string model_sha256;
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    // Scalar-reference F32 copies of non-F32 matrices (quantized GGUF only).
    std::vector<std::vector<float>> dequantized;
    std::unique_ptr<crispembed::spacy::NumericModel> model;
    // Opt-in CRISPEMBED_SPACY_GGML document engine over the same weights.
    std::unique_ptr<crispembed::spacy::GgmlNumeric> ggml;
    std::unique_ptr<crispembed::spacy::LexicalModel> lexical;
    std::unique_ptr<crispembed::spacy::DecodeResources> decoder;

    ~crispembed_spacy_context() {
        ggml.reset();
        model.reset();
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};

struct crispembed_spacy_result {
    std::string model_sha256;
    crispembed::spacy::NumericOutput output;
};

struct crispembed_spacy_document {
    std::vector<crispembed::spacy::LexicalToken> tokens;
    crispembed::spacy::DecodeOutput annotations;
    bool analyzed = false;
};

namespace {

// The ggml engine handles nonempty documents; the scalar reference keeps the
// empty-document contract and remains the default (CRISPEMBED_SPACY_GGML off).
crispembed::spacy::NumericOutput run_numeric(const crispembed_spacy_context & context,
                                             const std::vector<crispembed::spacy::FeatureRow> & features,
                                             const crispembed::spacy::TraceSink & sink,
                                             const crispembed::spacy::CancelCheck & cancelled) {
    if (context.ggml && !features.empty())
        return context.ggml->run(features, CRISPEMBED_SPACY_MAX_TOKENS, sink, cancelled);
    return context.model->run(features, CRISPEMBED_SPACY_MAX_TOKENS, sink, cancelled);
}

} // namespace

extern "C" {

uint32_t crispembed_spacy_abi_version(void) {
    return CRISPEMBED_SPACY_ABI_VERSION;
}

uint32_t crispembed_spacy_capabilities(void) {
    return CRISPEMBED_SPACY_PREPARED_FEATURE_SCORES | CRISPEMBED_SPACY_PARSER_STATE_SCORES;
}

uint32_t crispembed_spacy_text_abi_version(void) {
    return CRISPEMBED_SPACY_TEXT_ABI_VERSION;
}

uint32_t crispembed_spacy_context_capabilities(const crispembed_spacy_context * context) {
    if (!context) return 0;
    return crispembed_spacy_capabilities() |
           (context->lexical ? static_cast<uint32_t>(CRISPEMBED_SPACY_TOKENIZE_UTF8) : 0u) |
           (context->lexical && context->decoder ? static_cast<uint32_t>(CRISPEMBED_SPACY_ANALYZE_UTF8) : 0u);
}

crispembed_spacy_context * crispembed_spacy_create(const char * model_path, const char * expected_sha256, char * error,
                                                   size_t error_capacity) {
    error_text(error, error_capacity, "");
    if (error_capacity && !error) return nullptr;
    try {
        if (!core_env::on("CRISPEMBED_SPACY")) {
            error_text(error, error_capacity,
                       "Native spaCy numeric development backend is disabled; set CRISPEMBED_SPACY=1");
            return nullptr;
        }
        require(model_path && *model_path && valid_sha(expected_sha256),
                "Explicit model path and trusted lowercase SHA-256 required");
        const auto bytes = read_model(model_path);
        require(core_sha256::bytes(bytes.data(), bytes.size()) == expected_sha256, "GGUF SHA-256 mismatch");
        const Profile profile = preflight(bytes);
        auto context = std::make_unique<crispembed_spacy_context>();
        context->model_sha256 = expected_sha256;
        context->backend = ggml_backend_cpu_init();
        if (!context->backend) throw std::bad_alloc();
        ggml_backend_cpu_set_n_threads(context->backend, spacy_threads());
        require(core_gguf::load_weights(model_path, context->backend, "spacy", context->weights, false),
                "Cannot load admitted spaCy weights");
        require(!context->weights.used_mmap && context->weights.tensors.size() == profile.tensors.size(),
                "Unexpected spaCy weight ownership/inventory");
        crispembed::spacy::TensorMap tensors;
        std::map<std::string, ggml_tensor *> ggml_weights;
        context->dequantized.reserve(profile.tensors.size());
        for (const auto & item : profile.tensors) {
            const auto loaded = context->weights.tensors.find(item.first);
            require(loaded != context->weights.tensors.end(), "Loaded GGUF tensor missing");
            const auto * tensor = loaded->second;
            const auto & extent = item.second;
            require(tensor->type == extent.type && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) == extent.bytes,
                    "Loaded GGUF tensor type/size differs from admitted profile");
            for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                const size_t expected = i < extent.shape.size() ? extent.shape[extent.shape.size() - 1 - i] : 1;
                require(tensor->ne[i] == static_cast<int64_t>(expected),
                        "Loaded GGUF tensor shape changed during admission");
            }
            require(tensor->data && std::memcmp(tensor->data, bytes.data() + profile.data_offset + extent.offset,
                                                extent.bytes) == 0,
                    "GGUF tensor bytes changed during admission");
            const float * values = static_cast<const float *>(tensor->data);
            if (extent.type != GGML_TYPE_F32) {
                context->dequantized.push_back(dequantized_f32(static_cast<const uint8_t *>(tensor->data), extent));
                values = context->dequantized.back().data();
            }
            size_t elements = 1;
            for (auto dimension : extent.shape) elements *= dimension;
            tensors.emplace(item.first, crispembed::spacy::TensorView{ extent.shape, values, elements });
            ggml_weights.emplace(item.first, loaded->second);
        }
        require(core_sha256::file(model_path) == expected_sha256, "GGUF file changed while opening");
        context->model = std::make_unique<crispembed::spacy::NumericModel>(std::move(tensors));
        if (core_env::on("CRISPEMBED_SPACY_GGML"))
            context->ggml = std::make_unique<crispembed::spacy::GgmlNumeric>(std::move(ggml_weights), context->backend);
        return context.release();
    } catch (const std::bad_alloc &) {
        error_text(error, error_capacity, "Cannot allocate spaCy model resources");
    } catch (const std::exception & exception) {
        error_text(error, error_capacity, exception.what());
    } catch (...) {
        error_text(error, error_capacity, "Unknown spaCy model admission error");
    }
    return nullptr;
}

void crispembed_spacy_free(crispembed_spacy_context * context) {
    delete context;
}

crispembed_spacy_context * crispembed_spacy_create_with_resources(const char * model_path,
                                                                  const char * expected_model_sha256,
                                                                  const char * resources_path,
                                                                  const char * expected_resources_sha256, char * error,
                                                                  size_t error_capacity) {
    error_text(error, error_capacity, "");
    if (error_capacity && !error) return nullptr;
    try {
        if (!core_env::on("CRISPEMBED_SPACY")) {
            error_text(error, error_capacity, "Native spaCy backend is disabled; set CRISPEMBED_SPACY=1");
            return nullptr;
        }
        // Authenticate resource bytes before either resource parser sees them.
        const auto resources = read_resources(resources_path, expected_resources_sha256);
        std::unique_ptr<crispembed_spacy_context> context(
            crispembed_spacy_create(model_path, expected_model_sha256, error, error_capacity));
        if (!context) return nullptr;
        context->lexical = std::make_unique<crispembed::spacy::LexicalModel>(resources);
        context->decoder = std::make_unique<crispembed::spacy::DecodeResources>(
            crispembed::spacy::DecodeResources::from_json(resources));
        return context.release();
    } catch (const std::bad_alloc &) {
        error_text(error, error_capacity, "Cannot allocate spaCy text resources");
    } catch (const std::exception & exception) {
        error_text(error, error_capacity, exception.what());
    } catch (...) {
        error_text(error, error_capacity, "Unknown spaCy text resource admission error");
    }
    return nullptr;
}

int crispembed_spacy_tokenize_utf8(crispembed_spacy_context * context, const char * text, size_t text_bytes,
                                   crispembed_spacy_document ** document, crispembed_spacy_cancel_callback cancel,
                                   void * cancel_user, char * error, size_t error_capacity) {
    if (document) *document = nullptr;
    error_text(error, error_capacity, "");
    if (!context || !document || (error_capacity && !error) || (text_bytes && !text) ||
        text_bytes > CRISPEMBED_SPACY_MAX_TEXT_BYTES || !context->lexical) {
        error_text(error, error_capacity, "Invalid text pointers/length or context without admitted text resources");
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    }
    try {
        if (!core_env::on("CRISPEMBED_SPACY")) {
            error_text(error, error_capacity, "Native spaCy backend is disabled");
            return CRISPEMBED_SPACY_DISABLED;
        }
        std::lock_guard<std::mutex> lock(context->mutex);
        const auto cancelled = cancellation(cancel, cancel_user);
        check_cancel(cancelled);
        auto owned = std::make_unique<crispembed_spacy_document>();
        owned->tokens = context->lexical->tokenize(std::string_view(text ? text : "", text_bytes), cancelled);
        check_cancel(cancelled);
        *document = owned.release();
        return CRISPEMBED_SPACY_OK;
    } catch (const crispembed::spacy::NumericCancelled & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_CANCELLED;
    } catch (const std::invalid_argument & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        error_text(error, error_capacity, "Cannot allocate spaCy tokenization resources");
        return CRISPEMBED_SPACY_ALLOCATION_ERROR;
    } catch (const std::exception & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    } catch (...) {
        error_text(error, error_capacity, "Unknown spaCy tokenization error");
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    }
}

int crispembed_spacy_analyze_utf8(crispembed_spacy_context * context, const char * text, size_t text_bytes,
                                  crispembed_spacy_document ** document, crispembed_spacy_trace_callback trace,
                                  void * trace_user, crispembed_spacy_cancel_callback cancel, void * cancel_user,
                                  char * error, size_t error_capacity) {
    if (document) *document = nullptr;
    error_text(error, error_capacity, "");
    if (!context || !document || (error_capacity && !error) || (text_bytes && !text) ||
        text_bytes > CRISPEMBED_SPACY_MAX_TEXT_BYTES || !context->lexical || !context->decoder) {
        error_text(error, error_capacity,
                   "Invalid text pointers/length or context without admitted analysis resources");
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    }
    try {
        if (!core_env::on("CRISPEMBED_SPACY")) {
            error_text(error, error_capacity, "Native spaCy backend is disabled");
            return CRISPEMBED_SPACY_DISABLED;
        }
        std::lock_guard<std::mutex> lock(context->mutex);
        const auto cancelled = cancellation(cancel, cancel_user);
        check_cancel(cancelled);
        auto owned = std::make_unique<crispembed_spacy_document>();
        owned->tokens = context->lexical->tokenize(std::string_view(text ? text : "", text_bytes), cancelled);
        std::vector<crispembed::spacy::FeatureRow> features;
        std::vector<crispembed::spacy::DecodeToken> tokens;
        features.reserve(owned->tokens.size());
        tokens.reserve(owned->tokens.size());
        for (const auto & token : owned->tokens) {
            features.push_back(token.features);
            tokens.push_back({ token.text, token.is_space, {} });
        }
        const auto sink = trace_sink(trace, trace_user);
        const auto numeric = run_numeric(*context, features, sink, cancelled);
        const crispembed::spacy::ParserScorer scorer = [&](const std::array<int32_t, 8> & ids, size_t step) {
            crispembed::spacy::TraceSink parser_sink;
            if (sink) {
                parser_sink = [&](const std::string & name, const crispembed::spacy::TensorView & view) {
                    require(name.rfind("parser.", 0) == 0, "Invalid parser trace stage");
                    sink("parser.step." + std::to_string(step) + "." + name.substr(7), view);
                };
            }
            return context->model->parser_scores(numeric.parser_cache, ids, parser_sink, cancelled);
        };
        crispembed::spacy::ParserTrace parser_trace;
        if (sink) {
            parser_trace = [&](const crispembed::spacy::ParserStep & step) {
                const auto prefix = "parser.step." + std::to_string(step.step) + ".state.";
                const std::vector<float> ids(step.features.begin(), step.features.end());
                const std::vector<float> mask(step.valid_actions.begin(), step.valid_actions.end());
                const float choice = static_cast<float>(step.chosen_action);
                sink(prefix + "features", { { ids.size() }, ids.data(), ids.size() });
                sink(prefix + "valid_actions", { { mask.size() }, mask.data(), mask.size() });
                sink(prefix + "chosen_action", { { 1 }, &choice, 1 });
            };
        }
        owned->annotations =
            crispembed::spacy::decode_annotations(*context->decoder, numeric, tokens, scorer, parser_trace, cancelled);
        require(owned->annotations.tokens.size() == owned->tokens.size(), "Incomplete spaCy annotation output");
        check_cancel(cancelled);
        owned->analyzed = true;
        *document = owned.release();
        return CRISPEMBED_SPACY_OK;
    } catch (const crispembed::spacy::NumericCancelled & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_CANCELLED;
    } catch (const TraceAborted & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_TRACE_ERROR;
    } catch (const std::invalid_argument & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        error_text(error, error_capacity, "Cannot allocate spaCy text analysis resources");
        return CRISPEMBED_SPACY_ALLOCATION_ERROR;
    } catch (const std::exception & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    } catch (...) {
        error_text(error, error_capacity, "Unknown spaCy text analysis error");
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    }
}

size_t crispembed_spacy_document_token_count(const crispembed_spacy_document * document) {
    return document ? document->tokens.size() : 0;
}

int crispembed_spacy_document_is_analyzed(const crispembed_spacy_document * document) {
    return document && document->analyzed ? 1 : 0;
}

int crispembed_spacy_document_token(const crispembed_spacy_document * document, size_t index,
                                    crispembed_spacy_token_view * token) {
    if (token) *token = {};
    if (!document || !token || index >= document->tokens.size()) return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    const auto & value = document->tokens[index];
    token->text = { value.text.data(), value.text.size() };
    token->norm = { value.norm.data(), value.norm.size() };
    token->prefix = { value.prefix.data(), value.prefix.size() };
    token->suffix = { value.suffix.data(), value.suffix.size() };
    token->shape = { value.shape.data(), value.shape.size() };
    token->byte_start = value.byte_start;
    token->byte_end = value.byte_end;
    token->scalar_start = value.scalar_start;
    token->scalar_end = value.scalar_end;
    token->utf16_start = value.utf16_start;
    token->utf16_end = value.utf16_end;
    token->orth_id = value.orth_id;
    std::copy(value.features.begin(), value.features.end(), token->features);
    token->head = -1;
    token->flags = (value.space_after ? static_cast<uint32_t>(CRISPEMBED_SPACY_SPACE_AFTER) : 0u) |
                   (value.is_space ? static_cast<uint32_t>(CRISPEMBED_SPACY_IS_SPACE) : 0u);
    if (document->analyzed) {
        const auto & annotation = document->annotations.tokens[index];
        token->lemma = { annotation.lemma.data(), annotation.lemma.size() };
        token->tag = { annotation.tag.data(), annotation.tag.size() };
        token->pos = { annotation.pos.data(), annotation.pos.size() };
        token->morph = { annotation.morph.data(), annotation.morph.size() };
        token->dependency = { annotation.dep.data(), annotation.dep.size() };
        token->head = annotation.head;
        if (annotation.is_sent_start) token->flags |= CRISPEMBED_SPACY_SENTENCE_START;
        if (index + 1 == document->tokens.size() || document->annotations.tokens[index + 1].is_sent_start)
            token->flags |= CRISPEMBED_SPACY_SENTENCE_END;
    }
    return CRISPEMBED_SPACY_OK;
}

void crispembed_spacy_document_free(crispembed_spacy_document * document) {
    delete document;
}

int crispembed_spacy_forward_features(crispembed_spacy_context * context, const uint64_t * features, size_t token_count,
                                      size_t feature_count, crispembed_spacy_result ** result,
                                      crispembed_spacy_trace_callback trace, void * trace_user,
                                      crispembed_spacy_cancel_callback cancel, void * cancel_user, char * error,
                                      size_t error_capacity) {
    if (result) *result = nullptr;
    error_text(error, error_capacity, "");
    if (!context || !result || (error_capacity && !error) || token_count > CRISPEMBED_SPACY_MAX_TOKENS ||
        feature_count != token_count * CRISPEMBED_SPACY_FEATURE_COUNT || (feature_count && !features)) {
        error_text(error, error_capacity, "Invalid spaCy feature pointers/counts or token limit");
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    }
    try {
        if (!core_env::on("CRISPEMBED_SPACY")) {
            error_text(error, error_capacity, "Native spaCy numeric development backend is disabled");
            return CRISPEMBED_SPACY_DISABLED;
        }
        std::lock_guard<std::mutex> lock(context->mutex);
        const auto cancelled = cancellation(cancel, cancel_user);
        check_cancel(cancelled);
        std::vector<crispembed::spacy::FeatureRow> rows(token_count);
        for (size_t i = 0; i < token_count; ++i) {
            for (size_t j = 0; j < 6; ++j) rows[i][j] = features[i * 6 + j];
            require(rows[i][4] <= 1 && rows[i][5] <= 1, "SPACY and IS_SPACE feature IDs must be Boolean");
        }
        auto owned = std::make_unique<crispembed_spacy_result>();
        owned->model_sha256 = context->model_sha256;
        owned->output = run_numeric(*context, rows, trace_sink(trace, trace_user), cancelled);
        check_cancel(cancelled);
        *result = owned.release();
        return CRISPEMBED_SPACY_OK;
    } catch (const crispembed::spacy::NumericCancelled & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_CANCELLED;
    } catch (const TraceAborted & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_TRACE_ERROR;
    } catch (const std::invalid_argument & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        error_text(error, error_capacity, "Cannot allocate spaCy inference resources");
        return CRISPEMBED_SPACY_ALLOCATION_ERROR;
    } catch (const std::exception & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    } catch (...) {
        error_text(error, error_capacity, "Unknown spaCy inference error");
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    }
}

int crispembed_spacy_result_matrix(const crispembed_spacy_result * result, const char * name, const float ** values,
                                   size_t * rows, size_t * columns) {
    if (values) *values = nullptr;
    if (rows) *rows = 0;
    if (columns) *columns = 0;
    if (!result || !name || !values || !rows || !columns) return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    const crispembed::spacy::Matrix * matrix = nullptr;
    if (std::strcmp(name, "tok2vec") == 0)
        matrix = &result->output.tok2vec;
    else if (std::strcmp(name, "tagger") == 0)
        matrix = &result->output.tagger;
    else if (std::strcmp(name, "morphologizer") == 0)
        matrix = &result->output.morphologizer;
    else if (std::strcmp(name, "lemmatizer") == 0)
        matrix = &result->output.lemmatizer;
    else
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    *values = matrix->data.empty() ? nullptr : matrix->data.data();
    *rows = matrix->rows;
    *columns = matrix->cols;
    return CRISPEMBED_SPACY_OK;
}

void crispembed_spacy_result_free(crispembed_spacy_result * result) {
    delete result;
}

int crispembed_spacy_parser_scores(crispembed_spacy_context * context, const crispembed_spacy_result * result,
                                   const int32_t * state_features, size_t state_feature_count, float * output,
                                   size_t output_capacity, crispembed_spacy_trace_callback trace, void * trace_user,
                                   crispembed_spacy_cancel_callback cancel, void * cancel_user, char * error,
                                   size_t error_capacity) {
    error_text(error, error_capacity, "");
    if (!context || !result || !state_features || state_feature_count != 8 || !output || output_capacity < 137 ||
        (error_capacity && !error) || !result->output.tok2vec.rows || result->model_sha256 != context->model_sha256) {
        error_text(error, error_capacity, "Invalid parser state/result/model identity or output capacity");
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    }
    try {
        if (!core_env::on("CRISPEMBED_SPACY")) {
            error_text(error, error_capacity, "Native spaCy numeric development backend is disabled");
            return CRISPEMBED_SPACY_DISABLED;
        }
        std::lock_guard<std::mutex> lock(context->mutex);
        const auto cancelled = cancellation(cancel, cancel_user);
        check_cancel(cancelled);
        std::array<int32_t, 8> ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = state_features[i];
            require(ids[i] >= -1 && (ids[i] == -1 || static_cast<size_t>(ids[i]) < result->output.tok2vec.rows),
                    "Parser feature index outside the document");
        }
        const auto scores =
            context->model->parser_scores(result->output.parser_cache, ids, trace_sink(trace, trace_user), cancelled);
        require(scores.size() == 137, "Unexpected parser score dimension");
        check_cancel(cancelled);
        std::copy(scores.begin(), scores.end(), output);
        return CRISPEMBED_SPACY_OK;
    } catch (const crispembed::spacy::NumericCancelled & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_CANCELLED;
    } catch (const TraceAborted & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_TRACE_ERROR;
    } catch (const std::invalid_argument & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        error_text(error, error_capacity, "Cannot allocate spaCy parser resources");
        return CRISPEMBED_SPACY_ALLOCATION_ERROR;
    } catch (const std::exception & exception) {
        error_text(error, error_capacity, exception.what());
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    } catch (...) {
        error_text(error, error_capacity, "Unknown spaCy parser scoring error");
        return CRISPEMBED_SPACY_RUNTIME_ERROR;
    }
}

} // extern "C"
