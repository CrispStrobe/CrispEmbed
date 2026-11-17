#include "spacy_numeric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// Inference semantics: spaCy release-v3.8.18 (43311999028032fd81bef77a75fdb4fa96803c98),
// spacy/ml/models/{tok2vec,tagger,parser}.py and spacy/ml/parser_model.pyx;
// Thinc v8.3.14 (e617de4ea2530c7091ea2110e011807c33c366f0),
// thinc/layers/{hashembed,maxout,layernorm,with_array}.py and numpy_ops.pyx.
// Both upstream projects are MIT licensed. This bounded scalar F32 baseline
// intentionally keeps reduction order visible for differential diagnosis.
namespace crispembed {
namespace spacy {
namespace {

constexpr std::size_t width = 96;
constexpr std::size_t pieces = 3;
constexpr std::size_t features_count = 6;
constexpr std::size_t padding = 4;
constexpr std::size_t parser_features = 8;
constexpr std::size_t parser_width = 64;
constexpr std::size_t parser_pieces = 2;
constexpr std::size_t parser_columns = parser_features * parser_width * parser_pieces;
constexpr std::array<std::size_t, features_count> embedding_rows{ { 5000, 1000, 2500, 2500, 50, 50 } };

void check_cancelled(const CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw NumericCancelled();
}

std::size_t checked_product(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::length_error("spaCy numeric matrix size overflow");
    return rows * columns;
}

Matrix matrix(std::size_t rows, std::size_t columns) {
    Matrix result;
    result.rows = rows;
    result.cols = columns;
    const auto count = checked_product(rows, columns);
    if (count > result.data.max_size()) throw std::length_error("spaCy numeric matrix exceeds allocation limit");
    result.data.assign(count, 0.0f);
    return result;
}

float finite(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("spaCy numeric inference produced a nonfinite activation");
    return value;
}

void emit(const TraceSink & trace, const std::string & name, const Matrix & value) {
    if (trace) trace(name, value.view());
}

void emit(const TraceSink & trace, const std::string & name, const Matrix & value, std::vector<std::size_t> shape) {
    if (trace) trace(name, TensorView{ std::move(shape), value.data.data(), value.data.size() });
}

// W is original C row-major, with its input axis last. Rank-3 maxout and
// rank-4 parser weights flatten only their output axes, never transpose bytes.
Matrix affine(const Matrix & input, const TensorView & weights, const TensorView * bias,
              const CancelCheck & cancelled) {
    check_cancelled(cancelled);
    const auto columns = weights.size / input.cols;
    auto output = matrix(input.rows, columns);
    for (std::size_t row = 0; row < input.rows; ++row) {
        check_cancelled(cancelled);
        const float * x = input.data.data() + row * input.cols;
        for (std::size_t column = 0; column < columns; ++column) {
            const float * w = weights.data + column * input.cols;
            float sum = 0.0f;
            for (std::size_t k = 0; k < input.cols; ++k) sum += x[k] * w[k];
            if (bias) sum += bias->data[column];
            output.data[row * columns + column] = finite(sum);
        }
    }
    return output;
}

Matrix maxout_normalized(const Matrix & input, const TensorView & weights, const TensorView & bias,
                         const TensorView & gain, const TensorView & shift, const std::string & trace_prefix,
                         const TraceSink & trace, const CancelCheck & cancelled) {
    const auto projected = affine(input, weights, &bias, cancelled);
    emit(trace, trace_prefix + ".affine", projected, { input.rows, width, pieces });
    check_cancelled(cancelled);
    auto maximum = matrix(input.rows, width);
    for (std::size_t row = 0; row < input.rows; ++row) {
        check_cancelled(cancelled);
        for (std::size_t column = 0; column < width; ++column) {
            const float * candidates = projected.data.data() + (row * width + column) * pieces;
            float best = candidates[0];
            for (std::size_t piece = 1; piece < pieces; ++piece)
                if (candidates[piece] > best) best = candidates[piece];
            maximum.data[row * width + column] = best;
        }
    }
    emit(trace, trace_prefix + ".maxout", maximum);
    check_cancelled(cancelled);
    auto normalized = matrix(input.rows, width);
    for (std::size_t row = 0; row < input.rows; ++row) {
        check_cancelled(cancelled);
        const float * x = maximum.data.data() + row * width;
        float sum = 0.0f;
        for (std::size_t column = 0; column < width; ++column) sum += x[column];
        const float mean = finite(sum / static_cast<float>(width));
        float squared = 0.0f;
        for (std::size_t column = 0; column < width; ++column) {
            const float delta = x[column] - mean;
            squared += delta * delta;
        }
        // Thinc uses population variance, per row, with epsilon inside sqrt.
        const float variance = finite(squared / static_cast<float>(width) + 1e-8f);
        const float inverse = 1.0f / std::sqrt(variance);
        for (std::size_t column = 0; column < width; ++column) {
            const float standardized = (x[column] - mean) * inverse;
            normalized.data[row * width + column] = finite(standardized * gain.data[column] + shift.data[column]);
        }
    }
    emit(trace, trace_prefix + ".norm", normalized);
    return normalized;
}

std::uint64_t final_mix(std::uint64_t value) {
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value *= UINT64_C(0xc4ceb9fe1a85ec53);
    return value ^ (value >> 33);
}

} // namespace

std::array<std::uint32_t, 4> hash_keys(std::uint64_t value, std::uint32_t seed) {
    std::uint64_t a = value * UINT64_C(0x87c37b91114253d5);
    a = (a << 31) | (a >> 33);
    a *= UINT64_C(0x4cf5ad432745937f);
    a ^= static_cast<std::uint64_t>(seed);
    a ^= UINT64_C(8);
    std::uint64_t b = static_cast<std::uint64_t>(seed) ^ UINT64_C(8);
    a += b;
    b += a;
    a = final_mix(a);
    b = final_mix(b);
    a += b;
    b += a;
    return { { static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(a >> 32), static_cast<std::uint32_t>(b),
               static_cast<std::uint32_t>(b >> 32) } };
}

NumericModel::NumericModel(TensorMap tensors) : tensors_(std::move(tensors)) {
    if (tensors_.size() != 39) throw std::invalid_argument("spaCy numeric model requires exactly 39 tensors");
    auto expect = [&](const std::string & name, std::vector<std::size_t> shape) {
        auto found = tensors_.find(name);
        if (found == tensors_.end()) throw std::invalid_argument("Missing spaCy numeric tensor: " + name);
        const auto & value = found->second;
        std::size_t count = 1;
        for (auto dimension : shape) count = checked_product(count, dimension);
        if (value.shape != shape || value.size != count || !value.data)
            throw std::invalid_argument("Invalid spaCy numeric tensor layout: " + name);
        for (std::size_t i = 0; i < count; ++i)
            if (!std::isfinite(value.data[i])) throw std::invalid_argument("Nonfinite spaCy numeric tensor: " + name);
    };
    for (std::size_t i = 0; i < features_count; ++i)
        expect("embed." + std::to_string(i) + ".E", { embedding_rows[i], width });
    expect("embed.mix.W", { width, pieces, features_count * width });
    expect("embed.mix.b", { width, pieces });
    expect("embed.norm.G", { width });
    expect("embed.norm.b", { width });
    for (std::size_t i = 0; i < 4; ++i) {
        const auto prefix = "encoder." + std::to_string(i);
        expect(prefix + ".maxout.W", { width, pieces, 3 * width });
        expect(prefix + ".maxout.b", { width, pieces });
        expect(prefix + ".norm.G", { width });
        expect(prefix + ".norm.b", { width });
    }
    expect("tagger.W", { 55, width });
    expect("tagger.b", { 55 });
    expect("morphologizer.W", { 671, width });
    expect("morphologizer.b", { 671 });
    expect("lemmatizer.W", { 1311, width });
    expect("lemmatizer.b", { 1311 });
    expect("parser.reduce.W", { parser_width, width });
    expect("parser.reduce.b", { parser_width });
    expect("parser.lower.W", { parser_features, parser_width, parser_pieces, parser_width });
    expect("parser.lower.b", { parser_width, parser_pieces });
    expect("parser.lower.pad", { 1, parser_features, parser_width, parser_pieces });
    expect("parser.upper.W", { 137, parser_width });
    expect("parser.upper.b", { 137 });
}

const TensorView & NumericModel::tensor(const std::string & name) const {
    return tensors_.at(name);
}

NumericOutput NumericModel::run(const std::vector<FeatureRow> & features, std::size_t max_tokens,
                                const TraceSink & trace, const CancelCheck & cancelled) const {
    check_cancelled(cancelled);
    const auto count = features.size();
    if (count > max_tokens || count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - 1)
        throw std::length_error("spaCy input token count exceeds the admitted limit");
    NumericOutput output;
    if (count == 0) {
        output.tok2vec = matrix(0, width);
        output.tagger = matrix(0, 55);
        output.morphologizer = matrix(0, 671);
        output.lemmatizer = matrix(0, 1311);
        output.parser_cache = matrix(0, parser_columns);
        emit(trace, "tok2vec.output", output.tok2vec);
        check_cancelled(cancelled);
        emit(trace, "tagger.scores", output.tagger);
        check_cancelled(cancelled);
        emit(trace, "morphologizer.scores", output.morphologizer);
        check_cancelled(cancelled);
        emit(trace, "lemmatizer.scores", output.lemmatizer);
        check_cancelled(cancelled);
        return output;
    }

    auto embedded = matrix(count, features_count * width);
    for (std::size_t token = 0; token < count; ++token) {
        check_cancelled(cancelled);
        for (std::size_t feature = 0; feature < features_count; ++feature) {
            const auto & table = tensor("embed." + std::to_string(feature) + ".E");
            float * target = embedded.data.data() + (token * features_count + feature) * width;
            for (const auto key : hash_keys(features[token][feature], static_cast<std::uint32_t>(8 + feature))) {
                const auto row = static_cast<std::size_t>(key) % embedding_rows[feature];
                const float * embedding = table.data + row * width;
                // Four ordered additions, including every repeated bucket.
                for (std::size_t column = 0; column < width; ++column)
                    target[column] = finite(target[column] + embedding[column]);
            }
        }
    }
    emit(trace, "tok2vec.embed.concat", embedded);
    auto mixed = maxout_normalized(embedded, tensor("embed.mix.W"), tensor("embed.mix.b"), tensor("embed.norm.G"),
                                   tensor("embed.norm.b"), "tok2vec.embed", trace, cancelled);
    check_cancelled(cancelled);
    // with_array(pad=4) pads once. Padding activations then participate in all
    // residual layers; resetting boundary rows between layers changes outputs.
    auto encoded = matrix(count + 2 * padding, width);
    std::copy(mixed.data.begin(), mixed.data.end(), encoded.data.begin() + padding * width);
    emit(trace, "tok2vec.encoder.input", encoded);
    for (std::size_t layer = 0; layer < 4; ++layer) {
        check_cancelled(cancelled);
        auto window = matrix(encoded.rows, 3 * width);
        for (std::size_t row = 0; row < encoded.rows; ++row) {
            check_cancelled(cancelled);
            for (std::size_t slot = 0; slot < 3; ++slot) {
                if ((slot == 0 && row == 0) || (slot == 2 && row + 1 == encoded.rows)) continue;
                const auto source = row + slot - 1;
                std::copy_n(encoded.data.data() + source * width, width, window.data.data() + (row * 3 + slot) * width);
            }
        }
        const auto weights_prefix = "encoder." + std::to_string(layer);
        const auto trace_prefix = "tok2vec.encoder.layer." + std::to_string(layer);
        emit(trace, trace_prefix + ".window", window);
        auto delta = maxout_normalized(window, tensor(weights_prefix + ".maxout.W"),
                                       tensor(weights_prefix + ".maxout.b"), tensor(weights_prefix + ".norm.G"),
                                       tensor(weights_prefix + ".norm.b"), trace_prefix, trace, cancelled);
        for (std::size_t row = 0; row < encoded.rows; ++row) {
            check_cancelled(cancelled);
            for (std::size_t column = 0; column < width; ++column) {
                const auto i = row * width + column;
                encoded.data[i] = finite(encoded.data[i] + delta.data[i]);
            }
        }
        emit(trace, trace_prefix + ".residual", encoded);
    }
    check_cancelled(cancelled);
    output.tok2vec = matrix(count, width);
    std::copy_n(encoded.data.data() + padding * width, output.tok2vec.data.size(), output.tok2vec.data.data());
    emit(trace, "tok2vec.output", output.tok2vec);
    output.tagger = affine(output.tok2vec, tensor("tagger.W"), &tensor("tagger.b"), cancelled);
    emit(trace, "tagger.scores", output.tagger);
    output.morphologizer = affine(output.tok2vec, tensor("morphologizer.W"), &tensor("morphologizer.b"), cancelled);
    emit(trace, "morphologizer.scores", output.morphologizer);
    output.lemmatizer = affine(output.tok2vec, tensor("lemmatizer.W"), &tensor("lemmatizer.b"), cancelled);
    emit(trace, "lemmatizer.scores", output.lemmatizer);

    const auto reduced = affine(output.tok2vec, tensor("parser.reduce.W"), &tensor("parser.reduce.b"), cancelled);
    emit(trace, "parser.reduce", reduced);
    const auto lower = affine(reduced, tensor("parser.lower.W"), nullptr, cancelled);
    output.parser_cache = matrix(count + 1, parser_columns);
    std::copy_n(tensor("parser.lower.pad").data, parser_columns, output.parser_cache.data.data());
    std::copy(lower.data.begin(), lower.data.end(), output.parser_cache.data.begin() + parser_columns);
    emit(trace, "parser.lower_cache", output.parser_cache, { count + 1, parser_features, parser_width, parser_pieces });
    check_cancelled(cancelled);
    return output;
}

std::vector<float> NumericModel::parser_scores(const Matrix & cache, const std::array<std::int32_t, 8> & feature_ids,
                                               const TraceSink & trace, const CancelCheck & cancelled) const {
    check_cancelled(cancelled);
    if (cache.rows == 0 || cache.cols != parser_columns || cache.data.size() != checked_product(cache.rows, cache.cols))
        throw std::invalid_argument("Invalid spaCy parser cache geometry");
    for (const auto id : feature_ids)
        if (id < -1 || (id >= 0 && static_cast<std::size_t>(id) >= cache.rows - 1))
            throw std::out_of_range("spaCy parser feature token ID is out of range");
    auto summed = matrix(1, parser_width * parser_pieces);
    for (std::size_t feature = 0; feature < parser_features; ++feature) {
        check_cancelled(cancelled);
        const auto row = feature_ids[feature] == -1 ? 0 : static_cast<std::size_t>(feature_ids[feature]) + 1;
        const float * source = cache.data.data() + row * parser_columns + feature * summed.cols;
        for (std::size_t i = 0; i < summed.cols; ++i) summed.data[i] = finite(summed.data[i] + source[i]);
    }
    const auto & bias = tensor("parser.lower.b");
    for (std::size_t i = 0; i < summed.cols; ++i) summed.data[i] = finite(summed.data[i] + bias.data[i]);
    emit(trace, "parser.sum", summed, { 1, parser_width, parser_pieces });
    check_cancelled(cancelled);
    auto hidden = matrix(1, parser_width);
    for (std::size_t i = 0; i < parser_width; ++i) {
        const auto first = summed.data[i * parser_pieces];
        const auto second = summed.data[i * parser_pieces + 1];
        hidden.data[i] = second > first ? second : first;
    }
    emit(trace, "parser.hidden", hidden);
    const auto scores = affine(hidden, tensor("parser.upper.W"), &tensor("parser.upper.b"), cancelled);
    emit(trace, "parser.scores", scores);
    check_cancelled(cancelled);
    return scores.data;
}

} // namespace spacy
} // namespace crispembed
