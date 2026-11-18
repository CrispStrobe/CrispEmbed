#include "spacy_ner.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace crispembed {
namespace spacy {
namespace {

constexpr int64_t width = 96;
constexpr int64_t pieces = 3;
constexpr int64_t features_count = 4;
constexpr int64_t padding = 4;
constexpr std::size_t state_features = 3;
constexpr std::size_t hidden = 64;
constexpr std::size_t hidden_pieces = 2;
constexpr std::size_t slot = hidden * hidden_pieces;         // 128 per state feature
constexpr std::size_t cache_columns = state_features * slot; // 384
constexpr std::size_t embedding_rows[features_count] = { 5000, 1000, 2500, 2500 };
constexpr float layer_norm_epsilon = 1e-8f;

// Class names follow spaCy's BiluoPushDown order (get_class_name): "B-X",
// "I-X", "L-X", "U-X", "U-" (label 0, valid only for preset blocks) and "O".
enum Move { BEGIN, IN, LAST, UNIT, UNIT_BLOCKED, OUT };

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
    for (const float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("spaCy NER produced a nonfinite activation");
    return values;
}

} // namespace

const std::vector<std::string> & NerModel::labels() const {
    return labels_;
}

NerModel::NerModel(std::map<std::string, ggml_tensor *> weights, ggml_backend_t backend,
                   const std::vector<std::string> & class_names)
    : weights_(std::move(weights)), backend_(backend) {
    if (class_names.empty() || class_names.size() > 1024)
        throw std::invalid_argument("Invalid spaCy NER class inventory");
    for (const auto & name : class_names) {
        std::uint32_t label = 0;
        int move = -1;
        if (name == "O")
            move = OUT;
        else if (name == "U-")
            move = UNIT_BLOCKED;
        else if (name.size() > 2 && name[1] == '-') {
            const char kind = name[0];
            move = kind == 'B' ? BEGIN : kind == 'I' ? IN : kind == 'L' ? LAST : kind == 'U' ? UNIT : -1;
            const auto text = name.substr(2);
            const auto found = std::find(labels_.begin(), labels_.end(), text);
            label = static_cast<std::uint32_t>(found - labels_.begin());
            if (found == labels_.end()) labels_.push_back(text);
        }
        if (move < 0) throw std::invalid_argument("Unsupported spaCy NER class " + name);
        class_moves_.push_back(move);
        class_labels_.push_back(label);
    }
    if (!backend_ || !ggml_backend_is_cpu(backend_)) throw std::invalid_argument("spaCy NER requires a CPU backend");
    for (const auto & item : weights_)
        if (!item.second || item.second->type != GGML_TYPE_F32)
            throw std::invalid_argument("spaCy NER tensors must be F32: " + item.first);
    lower_bias_ = read(weight("ner.lower.b"));
    lower_pad_ = read(weight("ner.lower.pad"));
    upper_w_ = read(weight("ner.upper.W"));
    upper_b_ = read(weight("ner.upper.b"));
    const std::size_t classes = class_moves_.size();
    if (lower_bias_.size() != slot || lower_pad_.size() != cache_columns || upper_w_.size() != classes * hidden ||
        upper_b_.size() != classes)
        throw std::invalid_argument("Invalid spaCy NER parser tensor geometry");
    allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocator_) throw std::bad_alloc();
}

NerModel::~NerModel() {
    if (allocator_) ggml_gallocr_free(allocator_);
}

ggml_tensor * NerModel::weight(const std::string & name) const {
    const auto found = weights_.find(name);
    if (found == weights_.end() || !found->second) throw std::invalid_argument("Missing spaCy NER tensor: " + name);
    return found->second;
}

std::vector<Entity> NerModel::run(const std::vector<FeatureRow> & features,
                                  const std::vector<std::uint8_t> & sentence_starts, const TraceSink & trace,
                                  const CancelCheck & cancelled) const {
    check_cancelled(cancelled);
    const auto count = features.size();
    if (sentence_starts.size() != count) throw std::invalid_argument("spaCy NER sentence-boundary count mismatch");
    if (count == 0) return {};
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / 8)
        throw std::length_error("spaCy NER token count exceeds the admitted limit");
    const int64_t n = static_cast<int64_t>(count);
    const int64_t padded = n + 2 * padding;

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
    params.no_alloc = true;
    std::unique_ptr<ggml_context, ContextDeleter> owned(ggml_init(params));
    if (!owned) throw std::bad_alloc();
    ggml_context * ctx = owned.get();

    auto maxout_norm = [&](ggml_tensor * input, const std::string & prefix) {
        ggml_tensor * w = weight(prefix + "maxout.W");
        w = ggml_reshape_2d(ctx, w, w->ne[0], width * pieces);
        ggml_tensor * affine = ggml_add(ctx, ggml_mul_mat(ctx, w, input),
                                        ggml_reshape_1d(ctx, weight(prefix + "maxout.b"), width * pieces));
        ggml_tensor * maximum = ggml_pool_1d(ctx, affine, GGML_OP_POOL_MAX, pieces, pieces, 0);
        ggml_tensor * normalized = ggml_norm(ctx, maximum, layer_norm_epsilon);
        return ggml_add(ctx, ggml_mul(ctx, normalized, weight(prefix + "norm.G")), weight(prefix + "norm.b"));
    };

    // HashEmbed: four Murmur keys per feature (seeds 8..11), rows summed in key order.
    ggml_tensor * index_inputs[features_count];
    ggml_tensor * concatenated = nullptr;
    for (int64_t feature = 0; feature < features_count; ++feature) {
        index_inputs[feature] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
        ggml_set_input(index_inputs[feature]);
        ggml_tensor * gathered =
            ggml_get_rows(ctx, weight("ner.embed." + std::to_string(feature) + ".E"), index_inputs[feature]);
        ggml_tensor * sum = nullptr;
        for (int64_t key = 0; key < 4; ++key) {
            ggml_tensor * view = ggml_view_2d(ctx, gathered, width, n, gathered->nb[1] * 4, gathered->nb[1] * key);
            sum = sum ? ggml_add(ctx, sum, view) : ggml_cont(ctx, view);
        }
        concatenated = concatenated ? ggml_concat(ctx, concatenated, sum, 0) : sum;
    }
    ggml_tensor * w_mix = weight("ner.embed.mix.W");
    ggml_tensor * mixed =
        ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w_mix, w_mix->ne[0], width * pieces), concatenated),
                 ggml_reshape_1d(ctx, weight("ner.embed.mix.b"), width * pieces));
    mixed = ggml_pool_1d(ctx, mixed, GGML_OP_POOL_MAX, pieces, pieces, 0);
    mixed = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, mixed, layer_norm_epsilon), weight("ner.embed.norm.G")),
                     weight("ner.embed.norm.b"));

    // with_array(pad=4): padding rows evolve across every residual layer.
    ggml_tensor * encoded = ggml_pad_ext(ctx, mixed, 0, 0, padding, padding, 0, 0, 0, 0);
    for (int layer = 0; layer < 4; ++layer) {
        ggml_tensor * bordered = ggml_pad_ext(ctx, encoded, 0, 0, 1, 1, 0, 0, 0, 0);
        ggml_tensor * window = nullptr;
        for (int64_t s = 0; s < 3; ++s) {
            ggml_tensor * view = ggml_view_2d(ctx, bordered, width, padded, bordered->nb[1], bordered->nb[1] * s);
            window = window ? ggml_concat(ctx, window, view, 0) : ggml_cont(ctx, view);
        }
        encoded = ggml_add(ctx, encoded, maxout_norm(window, "ner.encoder." + std::to_string(layer) + "."));
    }
    ggml_tensor * tok2vec =
        ggml_cont(ctx, ggml_view_2d(ctx, encoded, width, n, encoded->nb[1], encoded->nb[1] * padding));
    ggml_set_output(tok2vec);
    ggml_tensor * reduced = ggml_add(ctx, ggml_mul_mat(ctx, weight("ner.reduce.W"), tok2vec), weight("ner.reduce.b"));
    ggml_set_output(reduced);
    ggml_tensor * lower_w = weight("ner.lower.W");
    ggml_tensor * lower = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, lower_w, lower_w->ne[0], cache_columns), reduced);
    ggml_set_output(lower);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, lower);
    if (!ggml_gallocr_alloc_graph(allocator_, graph)) throw std::bad_alloc();
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4 * n));
    for (int64_t feature = 0; feature < features_count; ++feature) {
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
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("spaCy NER graph computation failed");

    // Row 0 is learned padding; row i+1 belongs to token i.
    std::vector<float> cache(lower_pad_);
    const auto lower_values = read(lower);
    cache.insert(cache.end(), lower_values.begin(), lower_values.end());
    if (trace) {
        const auto t2v = read(tok2vec);
        trace("ner.tok2vec.output", TensorView{ { count, static_cast<std::size_t>(width) }, t2v.data(), t2v.size() });
        const auto red = read(reduced);
        trace("ner.reduce", TensorView{ { count, hidden }, red.data(), red.size() });
        trace("ner.lower_cache",
              TensorView{ { count + 1, state_features, hidden, hidden_pieces }, cache.data(), cache.size() });
    }

    // Greedy BILUO transitions (no preset entities, beam width 1).
    std::vector<Entity> entities;
    bool open = false;
    Entity current;
    std::array<float, slot> summed{};
    std::array<float, hidden> activation{};
    const std::size_t classes = class_moves_.size();
    std::vector<float> scores(classes);
    for (std::size_t b0 = 0; b0 < count; ++b0) {
        check_cancelled(cancelled);
        const std::size_t remaining = count - b0;
        const bool next_starts_sentence = b0 + 1 < count && sentence_starts[b0 + 1] != 0;
        const bool is_space = features[b0][5] != 0;
        const std::array<std::int32_t, state_features> ids = { static_cast<std::int32_t>(b0),
                                                               open ? static_cast<std::int32_t>(current.start) : -1,
                                                               open ? static_cast<std::int32_t>(b0) - 1 : -1 };
        summed.fill(0.0f);
        for (std::size_t f = 0; f < state_features; ++f) {
            const float * source =
                ids[f] < 0 ? lower_pad_.data() + f * slot
                           : cache.data() + (static_cast<std::size_t>(ids[f]) + 1) * cache_columns + f * slot;
            for (std::size_t i = 0; i < slot; ++i) summed[i] += source[i];
        }
        for (std::size_t i = 0; i < slot; ++i) summed[i] += lower_bias_[i];
        for (std::size_t h = 0; h < hidden; ++h) {
            const float first = summed[h * hidden_pieces];
            const float second = summed[h * hidden_pieces + 1];
            activation[h] = second > first ? second : first;
        }
        for (std::size_t c = 0; c < classes; ++c) {
            const float * w = upper_w_.data() + c * hidden;
            float sum = 0.0f;
            for (std::size_t h = 0; h < hidden; ++h) sum += activation[h] * w[h];
            scores[c] = sum + upper_b_[c];
            if (!std::isfinite(scores[c])) throw std::runtime_error("spaCy NER produced a nonfinite score");
        }
        std::vector<float> valid(classes);
        int best = -1;
        for (std::size_t c = 0; c < classes; ++c) {
            const auto label = class_labels_[c];
            bool ok = false;
            switch (static_cast<Move>(class_moves_[c])) {
            case BEGIN:
                ok = !open && remaining >= 2 && !next_starts_sentence && !is_space;
                break;
            case IN:
                ok = open && remaining >= 2 && current.label == label && !next_starts_sentence;
                break;
            case LAST:
                ok = open && current.label == label;
                break;
            case UNIT:
                ok = !open && !is_space;
                break;
            case UNIT_BLOCKED:
                ok = false;
                break;
            case OUT:
                ok = !open;
                break;
            }
            valid[c] = ok ? 1.0f : 0.0f;
            if (ok && (best < 0 || scores[c] > scores[static_cast<std::size_t>(best)])) best = static_cast<int>(c);
        }
        if (best < 0) throw std::runtime_error("spaCy NER state has no valid transition");
        if (trace) {
            const auto prefix = "ner.step." + std::to_string(b0) + ".";
            const std::vector<float> id_values(ids.begin(), ids.end());
            const float choice = static_cast<float>(best);
            trace(prefix + "state.features", TensorView{ { state_features }, id_values.data(), id_values.size() });
            trace(prefix + "scores", TensorView{ { classes }, scores.data(), classes });
            trace(prefix + "state.valid_actions", TensorView{ { classes }, valid.data(), classes });
            trace(prefix + "state.chosen_action", TensorView{ { 1 }, &choice, 1 });
        }
        const auto chosen = static_cast<std::size_t>(best);
        const auto label = class_labels_[chosen];
        switch (static_cast<Move>(class_moves_[chosen])) {
        case BEGIN:
            open = true;
            current = { static_cast<std::uint32_t>(b0), 0, label };
            break;
        case LAST:
            current.end = static_cast<std::uint32_t>(b0 + 1);
            entities.push_back(current);
            open = false;
            break;
        case UNIT:
            entities.push_back({ static_cast<std::uint32_t>(b0), static_cast<std::uint32_t>(b0 + 1), label });
            break;
        default:
            break;
        }
    }
    if (open) throw std::runtime_error("spaCy NER finished with an open entity");
    return entities;
}

} // namespace spacy
} // namespace crispembed
