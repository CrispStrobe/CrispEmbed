#include "spacy_ggml.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// Graph form of spacy_numeric.cpp (same upstream semantics, see that file).
namespace crispembed {
namespace spacy {
namespace {

constexpr int64_t width = 96;
constexpr int64_t pieces = 3;
constexpr int64_t features_count = 6;
constexpr int64_t padding = 4;
constexpr int64_t parser_columns = 8 * 64 * 2;
constexpr std::size_t embedding_rows[features_count] = { 5000, 1000, 2500, 2500, 50, 50 };
constexpr float layer_norm_epsilon = 1e-8f;

struct Stage {
    std::string name;
    ggml_tensor * tensor;
    std::vector<std::size_t> shape;
};

struct ContextDeleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

struct AbortState {
    const CancelCheck * cancelled;
    bool fired = false;
};

bool abort_requested(void * data) {
    auto * state = static_cast<AbortState *>(data);
    if (!state->fired && *state->cancelled && (*state->cancelled)()) state->fired = true;
    return state->fired;
}

void check_cancelled(const CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw NumericCancelled();
}

std::vector<float> read(const ggml_tensor * tensor) {
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void require_finite(const std::vector<float> & values) {
    for (const float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("spaCy numeric inference produced a nonfinite activation");
}

Matrix to_matrix(std::vector<float> values, std::size_t rows, std::size_t columns) {
    require_finite(values);
    Matrix result;
    result.rows = rows;
    result.cols = columns;
    result.data = std::move(values);
    return result;
}

} // namespace

GgmlNumeric::GgmlNumeric(std::map<std::string, ggml_tensor *> weights, ggml_backend_t backend)
    : weights_(std::move(weights)), backend_(backend) {
    if (!backend_ || !ggml_backend_is_cpu(backend_))
        throw std::invalid_argument("spaCy ggml engine requires a CPU backend");
    allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocator_) throw std::bad_alloc();
}

GgmlNumeric::~GgmlNumeric() {
    if (allocator_) ggml_gallocr_free(allocator_);
}

ggml_tensor * GgmlNumeric::weight(const std::string & name) const {
    const auto found = weights_.find(name);
    if (found == weights_.end() || !found->second) throw std::invalid_argument("Missing spaCy ggml tensor: " + name);
    return found->second;
}

NumericOutput GgmlNumeric::run(const std::vector<FeatureRow> & features, std::size_t max_tokens,
                               const TraceSink & trace, const CancelCheck & cancelled) const {
    check_cancelled(cancelled);
    const auto count = features.size();
    if (count == 0) throw std::invalid_argument("spaCy ggml engine requires a nonempty document");
    if (count > max_tokens || count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / 8)
        throw std::length_error("spaCy input token count exceeds the admitted limit");
    const int64_t n = static_cast<int64_t>(count);
    const int64_t padded = n + 2 * padding;

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    params.no_alloc = true;
    std::unique_ptr<ggml_context, ContextDeleter> owned(ggml_init(params));
    if (!owned) throw std::bad_alloc();
    ggml_context * ctx = owned.get();
    std::vector<Stage> stages;
    auto stage = [&](const std::string & name, ggml_tensor * tensor, std::vector<std::size_t> shape) {
        if (trace) {
            ggml_set_output(tensor);
            stages.push_back({ name, tensor, std::move(shape) });
        }
        return tensor;
    };
    const auto rows = static_cast<std::size_t>(n);
    const auto padded_rows = static_cast<std::size_t>(padded);

    // MaxoutWindowEncoder/MultiHashEmbed mix: affine -> maxout(3) -> LayerNorm.
    auto maxout_normalized = [&](ggml_tensor * input, const std::string & prefix, const std::string & weights,
                                 std::size_t stage_rows) {
        ggml_tensor * w = weight(weights + "W");
        w = ggml_reshape_2d(ctx, w, w->ne[0], width * pieces);
        ggml_tensor * affine = ggml_mul_mat(ctx, w, input);
        affine = ggml_add(ctx, affine, ggml_reshape_1d(ctx, weight(weights + "b"), width * pieces));
        stage(prefix + ".affine", affine,
              { stage_rows, static_cast<std::size_t>(width), static_cast<std::size_t>(pieces) });
        ggml_tensor * maximum = ggml_pool_1d(ctx, affine, GGML_OP_POOL_MAX, pieces, pieces, 0);
        stage(prefix + ".maxout", maximum, { stage_rows, static_cast<std::size_t>(width) });
        return maximum;
    };
    auto layer_norm = [&](ggml_tensor * input, const std::string & prefix, const std::string & weights,
                          std::size_t stage_rows) {
        ggml_tensor * normalized = ggml_norm(ctx, input, layer_norm_epsilon);
        normalized = ggml_add(ctx, ggml_mul(ctx, normalized, weight(weights + "G")), weight(weights + "b"));
        stage(prefix + ".norm", normalized, { stage_rows, static_cast<std::size_t>(width) });
        return normalized;
    };

    // HashEmbed: four Murmur keys per feature, rows summed in key order.
    ggml_tensor * index_inputs[features_count];
    ggml_tensor * concatenated = nullptr;
    for (int64_t feature = 0; feature < features_count; ++feature) {
        index_inputs[feature] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
        ggml_set_input(index_inputs[feature]);
        ggml_tensor * gathered =
            ggml_get_rows(ctx, weight("embed." + std::to_string(feature) + ".E"), index_inputs[feature]);
        ggml_tensor * sum = nullptr;
        for (int64_t key = 0; key < 4; ++key) {
            ggml_tensor * view = ggml_view_2d(ctx, gathered, width, n, gathered->nb[1] * 4, gathered->nb[1] * key);
            sum = sum ? ggml_add(ctx, sum, view) : ggml_cont(ctx, view);
        }
        concatenated = concatenated ? ggml_concat(ctx, concatenated, sum, 0) : sum;
    }
    stage("tok2vec.embed.concat", concatenated, { rows, static_cast<std::size_t>(features_count * width) });
    ggml_tensor * mixed = layer_norm(maxout_normalized(concatenated, "tok2vec.embed", "embed.mix.", rows),
                                     "tok2vec.embed", "embed.norm.", rows);

    // with_array(pad=4): padding rows participate in every residual layer.
    ggml_tensor * encoded = ggml_pad_ext(ctx, mixed, 0, 0, padding, padding, 0, 0, 0, 0);
    stage("tok2vec.encoder.input", encoded, { padded_rows, static_cast<std::size_t>(width) });
    for (int layer = 0; layer < 4; ++layer) {
        const auto prefix = "tok2vec.encoder.layer." + std::to_string(layer);
        const auto weights = "encoder." + std::to_string(layer) + ".";
        // expand_window(1): [previous, current, next], zero beyond the padded ends.
        ggml_tensor * bordered = ggml_pad_ext(ctx, encoded, 0, 0, 1, 1, 0, 0, 0, 0);
        ggml_tensor * window = nullptr;
        for (int64_t slot = 0; slot < 3; ++slot) {
            ggml_tensor * view = ggml_view_2d(ctx, bordered, width, padded, bordered->nb[1], bordered->nb[1] * slot);
            window = window ? ggml_concat(ctx, window, view, 0) : ggml_cont(ctx, view);
        }
        stage(prefix + ".window", window, { padded_rows, static_cast<std::size_t>(3 * width) });
        ggml_tensor * delta = layer_norm(maxout_normalized(window, prefix, weights + "maxout.", padded_rows), prefix,
                                         weights + "norm.", padded_rows);
        encoded = ggml_add(ctx, encoded, delta);
        stage(prefix + ".residual", encoded, { padded_rows, static_cast<std::size_t>(width) });
    }
    ggml_tensor * tok2vec =
        ggml_cont(ctx, ggml_view_2d(ctx, encoded, width, n, encoded->nb[1], encoded->nb[1] * padding));
    ggml_set_output(tok2vec);
    // Classifier heads are optional for v2 pipelines (e.g. English has no
    // morphologizer or trainable lemmatizer); a head without weights is skipped
    // and its output matrix stays empty. The German v1 model has all of them.
    auto head = [&](const char * name) -> ggml_tensor * {
        if (weights_.find(std::string(name) + ".W") == weights_.end()) return nullptr;
        ggml_tensor * scores = ggml_add(ctx, ggml_mul_mat(ctx, weight(std::string(name) + ".W"), tok2vec),
                                        weight(std::string(name) + ".b"));
        ggml_set_output(scores);
        return scores;
    };
    ggml_tensor * tagger = head("tagger");
    ggml_tensor * morphologizer = head("morphologizer");
    ggml_tensor * lemmatizer = head("lemmatizer");
    ggml_tensor * reduced = head("parser.reduce");
    if (!reduced) throw std::invalid_argument("Missing spaCy parser reduce layer");
    ggml_tensor * lower_w = weight("parser.lower.W");
    ggml_tensor * lower = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, lower_w, lower_w->ne[0], parser_columns), reduced);
    ggml_set_output(lower);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    for (const auto & item : stages) ggml_build_forward_expand(graph, item.tensor);
    for (ggml_tensor * output : { tagger, morphologizer, lemmatizer, lower })
        if (output) ggml_build_forward_expand(graph, output);
    if (!ggml_gallocr_alloc_graph(allocator_, graph)) throw std::bad_alloc();

    std::vector<std::int32_t> indices(static_cast<std::size_t>(4 * n));
    for (int64_t feature = 0; feature < features_count; ++feature) {
        check_cancelled(cancelled);
        for (std::size_t token = 0; token < count; ++token) {
            const auto keys = hash_keys(features[token][feature], static_cast<std::uint32_t>(8 + feature));
            for (std::size_t key = 0; key < 4; ++key)
                indices[token * 4 + key] = static_cast<std::int32_t>(keys[key] % embedding_rows[feature]);
        }
        ggml_backend_tensor_set(index_inputs[feature], indices.data(), 0, indices.size() * sizeof(std::int32_t));
    }

    AbortState abort_state{ &cancelled };
    ggml_backend_cpu_set_abort_callback(backend_, cancelled ? abort_requested : nullptr, &abort_state);
    const auto status = ggml_backend_graph_compute(backend_, graph);
    ggml_backend_cpu_set_abort_callback(backend_, nullptr, nullptr);
    if (abort_state.fired || status == GGML_STATUS_ABORTED) throw NumericCancelled();
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("spaCy ggml graph computation failed");

    NumericOutput output;
    output.tok2vec = to_matrix(read(tok2vec), rows, static_cast<std::size_t>(width));
    auto head_matrix = [&](ggml_tensor * scores) {
        return scores ? to_matrix(read(scores), rows, static_cast<std::size_t>(scores->ne[0])) : Matrix{};
    };
    output.tagger = head_matrix(tagger);
    output.morphologizer = head_matrix(morphologizer);
    output.lemmatizer = head_matrix(lemmatizer);
    const auto lower_values = read(lower);
    require_finite(lower_values);
    output.parser_cache.rows = rows + 1;
    output.parser_cache.cols = static_cast<std::size_t>(parser_columns);
    output.parser_cache.data.resize(output.parser_cache.rows * output.parser_cache.cols);
    ggml_backend_tensor_get(weight("parser.lower.pad"), output.parser_cache.data.data(), 0,
                            static_cast<std::size_t>(parser_columns) * sizeof(float));
    std::memcpy(output.parser_cache.data.data() + parser_columns, lower_values.data(),
                lower_values.size() * sizeof(float));

    if (trace) {
        for (const auto & item : stages) {
            const auto values = read(item.tensor);
            trace(item.name, TensorView{ item.shape, values.data(), values.size() });
        }
        trace("tok2vec.output", output.tok2vec.view());
        if (tagger) trace("tagger.scores", output.tagger.view());
        if (morphologizer) trace("morphologizer.scores", output.morphologizer.view());
        if (lemmatizer) trace("lemmatizer.scores", output.lemmatizer.view());
        const auto reduced_values = read(reduced);
        trace("parser.reduce", TensorView{ { rows, static_cast<std::size_t>(reduced->ne[0]) },
                                           reduced_values.data(),
                                           reduced_values.size() });
        trace("parser.lower_cache",
              TensorView{ { rows + 1, 8, 64, 2 }, output.parser_cache.data.data(), output.parser_cache.data.size() });
    }
    check_cancelled(cancelled);
    return output;
}

} // namespace spacy
} // namespace crispembed
