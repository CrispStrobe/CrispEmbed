#include "spacy_numeric.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using namespace crispembed::spacy;
std::size_t assertions = 0;

void require(bool condition, const std::string & message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function> void requires_exception(Function operation, const std::string & message) {
    bool caught = false;
    try {
        operation();
    } catch (const Exception &) {
        caught = true;
    }
    require(caught, message);
}

struct SyntheticWeights {
    struct Owned {
        std::vector<std::size_t> shape;
        std::vector<float> data;
    };
    std::unordered_map<std::string, Owned> tensors;

    void add(const std::string & name, std::vector<std::size_t> shape, float value = 0.0f) {
        std::size_t count = 1;
        for (auto dimension : shape) count *= dimension;
        tensors.emplace(name, Owned{ shape, std::vector<float>(count, value) });
    }

    SyntheticWeights() {
        const std::array<std::size_t, 6> rows{ { 5000, 1000, 2500, 2500, 50, 50 } };
        for (std::size_t i = 0; i < rows.size(); ++i) add("embed." + std::to_string(i) + ".E", { rows[i], 96 });
        add("embed.mix.W", { 96, 3, 576 });
        add("embed.mix.b", { 96, 3 });
        add("embed.norm.G", { 96 }, 1.0f);
        add("embed.norm.b", { 96 });
        for (std::size_t i = 0; i < 4; ++i) {
            const auto prefix = "encoder." + std::to_string(i);
            add(prefix + ".maxout.W", { 96, 3, 288 });
            add(prefix + ".maxout.b", { 96, 3 });
            add(prefix + ".norm.G", { 96 }, 1.0f);
            add(prefix + ".norm.b", { 96 });
        }
        for (const auto & head : std::vector<std::pair<std::string, std::size_t>>{
                 { "tagger", 55 }, { "morphologizer", 671 }, { "lemmatizer", 1311 } }) {
            add(head.first + ".W", { head.second, 96 });
            add(head.first + ".b", { head.second });
        }
        add("parser.reduce.W", { 64, 96 });
        add("parser.reduce.b", { 64 });
        add("parser.lower.W", { 8, 64, 2, 64 });
        add("parser.lower.b", { 64, 2 });
        add("parser.lower.pad", { 1, 8, 64, 2 });
        add("parser.upper.W", { 137, 64 });
        add("parser.upper.b", { 137 });
    }

    TensorMap views() const {
        TensorMap result;
        for (const auto & item : tensors)
            result.emplace(item.first,
                           TensorView{ item.second.shape, item.second.data.data(), item.second.data.size() });
        return result;
    }
};

void hash_controls() {
    // Frozen output from thinc 8.3.14 NumpyOps.hash, source e617de4ea253.
    // Values above INT64_MAX detect accidental signed conversion/overflow.
    const std::array<std::uint64_t, 7> values{ { 0, 1, 2, UINT64_C(0xffffffff), UINT64_C(0x100000000),
                                                 UINT64_C(0x8000000000000000), UINT64_C(0xffffffffffffffff) } };
    const std::array<std::array<std::uint32_t, 4>, 7> seed8{
        { { { 0, 0, 0, 0 } },
          { { 1161550108u, 2247783286u, 1742325162u, 1224191281u } },
          { { 180622166u, 2914303317u, 2418416897u, 2223971327u } },
          { { 1183447596u, 2168351004u, 1775171394u, 1105042858u } },
          { { 4243829122u, 2271772160u, 2070776387u, 3407658241u } },
          { { 2354736462u, 2184247079u, 1384621045u, 1128886971u } },
          { { 2488739950u, 2974852181u, 1585626277u, 2314794624u } } }
    };
    const std::array<std::array<std::uint32_t, 4>, 7> seed13{
        { { { 1041741114u, 1039425791u, 1457952615u, 394743738u } },
          { { 1266879892u, 1356517611u, 318478807u, 3343674364u } },
          { { 2428954001u, 3947668691u, 2620113271u, 4290702076u } },
          { { 4135855152u, 3396580483u, 3385019210u, 3561501817u } },
          { { 2587945398u, 2070709741u, 1124313452u, 2432145691u } },
          { { 3538663004u, 3291391371u, 1894261537u, 1309555066u } },
          { { 3199409384u, 1062937581u, 752279700u, 3061442444u } } }
    };
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(hash_keys(values[i], 8) == seed8[i], "Thinc hash seed 8 mismatch");
        require(hash_keys(values[i], 13) == seed13[i], "Thinc hash seed 13 mismatch");
    }
}

void admission_controls(SyntheticWeights & weights) {
    auto missing = weights.views();
    missing.erase("embed.0.E");
    requires_exception<std::invalid_argument>([&] { NumericModel model(missing); }, "Missing weight was accepted");
    auto extra = weights.views();
    extra.emplace("unexpected.weight", extra.at("tagger.b"));
    requires_exception<std::invalid_argument>([&] { NumericModel model(extra); }, "Unexpected weight was accepted");
    auto wrong_shape = weights.views();
    wrong_shape.at("embed.mix.W").shape = { 288, 576 };
    requires_exception<std::invalid_argument>([&] { NumericModel model(wrong_shape); },
                                              "Wrong rank with same size was accepted");
    auto truncated = weights.views();
    --truncated.at("parser.upper.W").size;
    requires_exception<std::invalid_argument>([&] { NumericModel model(truncated); },
                                              "Truncated tensor view was accepted");
    auto null_data = weights.views();
    null_data.at("morphologizer.b").data = nullptr;
    requires_exception<std::invalid_argument>([&] { NumericModel model(null_data); }, "Null tensor data was accepted");
    auto & bad = weights.tensors.at("lemmatizer.b").data[0];
    bad = std::numeric_limits<float>::quiet_NaN();
    requires_exception<std::invalid_argument>([&] { NumericModel model(weights.views()); },
                                              "Nonfinite parameter was accepted");
    bad = 0.0f;
}

void empty_and_cancel_controls(const NumericModel & model) {
    std::set<std::string> stages;
    auto sink = [&](const std::string & name, const TensorView & view) {
        require(stages.insert(name).second, "Repeated empty trace stage");
        require(view.shape.size() == 2 && view.shape[0] == 0 && view.size == 0,
                "Empty inference produced nonempty tensor");
    };
    const auto empty = model.run({}, 0, sink);
    require(stages ==
                std::set<std::string>{ "tok2vec.output", "tagger.scores", "morphologizer.scores", "lemmatizer.scores" },
            "Empty input executed fake embedding/CNN/parser stages");
    require(empty.tok2vec.rows == 0 && empty.tok2vec.cols == 96 && empty.tagger.cols == 55 &&
                empty.morphologizer.cols == 671 && empty.lemmatizer.cols == 1311 && empty.parser_cache.rows == 0,
            "Empty output dimensions differ from the contract");
    const std::vector<FeatureRow> one{ { { 0, 0, 0, 0, 0, 0 } } };
    requires_exception<std::length_error>([&] { model.run(one, 0); }, "Over-limit input was accepted");
    std::size_t emitted = 0;
    auto count = [&](const std::string &, const TensorView &) { ++emitted; };
    requires_exception<NumericCancelled>([&] { model.run(one, 1, count, [] { return true; }); },
                                         "Pre-cancelled inference ran");
    require(emitted == 0, "Pre-cancelled inference emitted a trace");
    requires_exception<NumericCancelled>([&] { model.run(one, 1, count, [&] { return emitted != 0; }); },
                                         "Cancellation after embedding was not observed");
    require(emitted == 1, "Inference continued after cancellation acknowledgement");
    require(model.run({}, 0).tok2vec.rows == 0, "Cancellation poisoned model reuse");
}

void duplicate_bucket_control(const NumericModel & model) {
    bool captured = false;
    const std::vector<FeatureRow> one{ { { 0, 0, 0, 0, 0, 0 } } };
    auto sink = [&](const std::string & name, const TensorView & view) {
        require(name == "tok2vec.embed.concat" && view.shape == std::vector<std::size_t>{ 1, 576 },
                "Wrong first embedding trace");
        for (std::size_t i = 0; i < 96; ++i)
            require(view.data[i] == static_cast<float>(i + 1),
                    "Repeated hash bucket was deduplicated or summed incorrectly");
        captured = true;
    };
    requires_exception<NumericCancelled>([&] { model.run(one, 1, sink, [&] { return captured; }); },
                                         "Embedding-only probe did not stop at its trace boundary");
    require(captured, "Embedding-only probe did not execute");
}

void synthetic_complete_graph_control(const NumericModel & model) {
    const std::vector<FeatureRow> one{ { { 0, 0, 0, 0, 0, 0 } } };
    const auto output = model.run(one, 1);
    require(output.tok2vec.rows == 1 && output.tok2vec.cols == 96 && output.tagger.cols == 55 &&
                output.morphologizer.cols == 671 && output.lemmatizer.cols == 1311 && output.parser_cache.rows == 2 &&
                output.parser_cache.cols == 1024,
            "Synthetic complete graph changed declared output geometry");
    for (const auto * value :
         { &output.tok2vec, &output.tagger, &output.morphologizer, &output.lemmatizer, &output.parser_cache })
        require(std::all_of(value->data.begin(), value->data.end(), [](float v) { return v == 0.0f; }),
                "Zero affine parameters produced nonzero activations");
}

void parser_controls(const NumericModel & model) {
    Matrix padding;
    padding.rows = 1;
    padding.cols = 8 * 64 * 2;
    padding.data.assign(padding.cols, 0.0f);
    const std::array<std::int32_t, 8> absent{ { -1, -1, -1, -1, -1, -1, -1, -1 } };
    const auto scores = model.parser_scores(padding, absent);
    require(scores.size() == 137 && std::all_of(scores.begin(), scores.end(), [](float v) { return v == 0.0f; }),
            "Synthetic zero parser state did not produce zero action logits");
    auto bad_ids = absent;
    bad_ids[0] = -2;
    requires_exception<std::out_of_range>([&] { model.parser_scores(padding, bad_ids); },
                                          "Negative parser ID other than -1 was accepted");
    bad_ids[0] = 0;
    requires_exception<std::out_of_range>([&] { model.parser_scores(padding, bad_ids); },
                                          "Out-of-document parser ID was accepted");
    Matrix truncated = padding;
    truncated.data.pop_back();
    requires_exception<std::invalid_argument>([&] { model.parser_scores(truncated, absent); },
                                              "Truncated parser cache was accepted");
    padding.data[0] = std::numeric_limits<float>::infinity();
    requires_exception<std::runtime_error>([&] { model.parser_scores(padding, absent); },
                                           "Nonfinite parser state produced usable scores");
    padding.data[0] = 0.0f;
    requires_exception<NumericCancelled>([&] { model.parser_scores(padding, absent, {}, [] { return true; }); },
                                         "Parser did not observe cancellation");
}

int run() {
    hash_controls();
    SyntheticWeights weights;
    admission_controls(weights);
    // Feature value 0 with seed 8 chooses row 0 four times. Distinct exactly
    // representable values make the expected unprojected sum independent.
    auto & embedding = weights.tensors.at("embed.0.E").data;
    for (std::size_t i = 0; i < 96; ++i) embedding[i] = static_cast<float>(i + 1) / 4.0f;
    const NumericModel model(weights.views());
    empty_and_cancel_controls(model);
    duplicate_bucket_control(model);
    synthetic_complete_graph_control(model);
    parser_controls(model);
    std::cout << "SPACY_NUMERIC_MODEL_FREE assertions=" << assertions
              << " actual_model_fixtures=0 qualified_inference=0\n";
    return 0;
}
} // namespace

int main() {
    try {
        core_util::clean_exit(run());
    } catch (const std::exception & error) {
        std::cerr << "SPACY_NUMERIC_MODEL_FREE FAILED: " << error.what() << '\n';
        core_util::clean_exit(1);
    }
}
