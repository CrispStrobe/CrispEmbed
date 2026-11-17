// Closed CPU/F32 numeric development backend for de_core_news_sm 3.8.0.
// Tokenization and linguistic decoding are deliberately outside this ABI.
#include "spacy_pipeline.h"

#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"
#include "ggml-cpu.h"
#include "spacy_numeric.h"

#include <algorithm>
#include <array>
#include <cmath>
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
};

struct Profile {
    size_t data_offset;
    std::map<std::string, Extent> tensors;
};

Profile preflight(const std::vector<uint8_t> & bytes) {
    require(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE-754 float32 required");
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "Little-endian host required by this profile");
    Reader reader(bytes);
    require(reader.u32() == 0x46554747u, "Invalid GGUF magic");
    require(reader.u32() == 3, "Only GGUF version 3 is supported");
    const auto shapes = expected_shapes();
    require(reader.u64() == shapes.size(), "Expected exactly 39 spaCy tensors");
    require(reader.u64() == 13, "Unexpected spaCy GGUF metadata inventory");
    const std::map<std::string, std::string> strings = {
        { "general.architecture", "spacy_de_core_news_sm" },
        { "spacy.model", "de_core_news_sm" },
        { "spacy.model_version", "3.8.0" },
        { "spacy.source.sha256", k_source_sha },
        { "spacy.resources.sha256", k_resources_sha },
        { "spacy.components", "tok2vec,tagger,morphologizer,parser,lemmatizer,attribute_ruler" },
        { "spacy.feature_order", "NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE" },
        { "spacy.precision", "f32" }
    };
    const std::map<std::string, uint32_t> integers = { { "spacy.format_version", 1 },
                                                       { "spacy.width", 96 },
                                                       { "spacy.encoder_depth", 4 },
                                                       { "spacy.encoder_padding", 4 } };
    std::set<std::string> seen_keys;
    for (size_t i = 0; i < 13; ++i) {
        const auto key = reader.string(128);
        require(seen_keys.insert(key).second, "Duplicate GGUF metadata key");
        const uint32_t type = reader.u32();
        const auto text = strings.find(key);
        const auto integer = integers.find(key);
        if (text != strings.end()) {
            require(type == GGUF_TYPE_STRING, "Mistyped required GGUF string");
            require(reader.string(256) == text->second, "Unsupported spaCy GGUF identity or configuration");
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
    Profile profile{};
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
        require(reader.u32() == GGML_TYPE_F32, "Only F32 spaCy tensors are admitted");
        const uint64_t offset = reader.u64();
        const size_t tensor_bytes = elements * sizeof(float);
        require(offset <= k_max_model_bytes && offset % k_alignment == 0, "Invalid GGUF tensor offset");
        require(tensor_bytes <= k_max_model_bytes - static_cast<size_t>(offset), "GGUF tensor extent exceeds bound");
        profile.tensors.emplace(name, Extent{ shape, static_cast<size_t>(offset), tensor_bytes });
        ranges.emplace_back(static_cast<size_t>(offset), aligned(static_cast<size_t>(offset) + tensor_bytes));
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
        for (size_t i = 0; i < extent.bytes; i += sizeof(float)) {
            float value;
            std::memcpy(&value, bytes.data() + begin + i, sizeof(value));
            require(std::isfinite(value), "Nonfinite spaCy tensor");
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

void check_cancel(const crispembed::spacy::CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw crispembed::spacy::NumericCancelled();
}

} // namespace

struct crispembed_spacy_context {
    std::mutex mutex;
    std::string model_sha256;
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    std::unique_ptr<crispembed::spacy::NumericModel> model;

    ~crispembed_spacy_context() {
        model.reset();
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};

struct crispembed_spacy_result {
    std::string model_sha256;
    crispembed::spacy::NumericOutput output;
};

extern "C" {

uint32_t crispembed_spacy_abi_version(void) {
    return CRISPEMBED_SPACY_ABI_VERSION;
}

uint32_t crispembed_spacy_capabilities(void) {
    return CRISPEMBED_SPACY_PREPARED_FEATURE_SCORES | CRISPEMBED_SPACY_PARSER_STATE_SCORES;
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
        ggml_backend_cpu_set_n_threads(context->backend, 1);
        require(core_gguf::load_weights(model_path, context->backend, "spacy", context->weights, false),
                "Cannot load admitted spaCy weights");
        require(!context->weights.used_mmap && context->weights.tensors.size() == profile.tensors.size(),
                "Unexpected spaCy weight ownership/inventory");
        crispembed::spacy::TensorMap tensors;
        for (const auto & item : profile.tensors) {
            const auto loaded = context->weights.tensors.find(item.first);
            require(loaded != context->weights.tensors.end(), "Loaded GGUF tensor missing");
            const auto * tensor = loaded->second;
            const auto & extent = item.second;
            require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) == extent.bytes,
                    "Loaded GGUF tensor type/size differs from admitted profile");
            for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                const size_t expected = i < extent.shape.size() ? extent.shape[extent.shape.size() - 1 - i] : 1;
                require(tensor->ne[i] == static_cast<int64_t>(expected),
                        "Loaded GGUF tensor shape changed during admission");
            }
            require(tensor->data && std::memcmp(tensor->data, bytes.data() + profile.data_offset + extent.offset,
                                                extent.bytes) == 0,
                    "GGUF tensor bytes changed during admission");
            tensors.emplace(item.first,
                            crispembed::spacy::TensorView{ extent.shape, static_cast<const float *>(tensor->data),
                                                           extent.bytes / sizeof(float) });
        }
        require(core_sha256::file(model_path) == expected_sha256, "GGUF file changed while opening");
        context->model = std::make_unique<crispembed::spacy::NumericModel>(std::move(tensors));
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
        owned->output =
            context->model->run(rows, CRISPEMBED_SPACY_MAX_TOKENS, trace_sink(trace, trace_user), cancelled);
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
