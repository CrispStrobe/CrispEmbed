// Model-free decoder controls. The optional two-path invocation replays only
// the frozen development oracle's authenticated scores, not raw-text/native
// numerical inference. Missing fixtures are an error, never a skipped pass.
#include "spacy_decode.h"
#include "core/clean_exit.h"
#include "core/json_value.h"
#include "core/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace crispembed::spacy;
using Json = core_json_value::value;
std::size_t assertions = 0;

void check(bool condition, const std::string & message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}
template <typename Exception = std::invalid_argument, typename Function>
void rejects(Function function, const std::string & message) {
    bool rejected = false;
    try {
        function();
    } catch (const Exception &) {
        rejected = true;
    }
    check(rejected, message);
}

const Json & get(const Json & value, const char * key) {
    const auto * found = value.get(key);
    check(value.kind == Json::object_t && found, std::string("Missing fixture field: ") + key);
    return *found;
}
std::size_t number(const Json & value) {
    check(value.kind == Json::int_t && value.i >= 0, "Noninteger fixture field");
    return static_cast<std::size_t>(value.i);
}
std::string read(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Missing decoder fixture: " + path);
    const auto size = file.tellg();
    check(size >= 0 && size <= 32 * 1024 * 1024, "Fixture exceeds bound");
    std::string data(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    if (!data.empty()) file.read(&data[0], static_cast<std::streamsize>(data.size()));
    check(static_cast<bool>(file), "Cannot read decoder fixture");
    return data;
}
std::string sha256(const std::string & bytes) {
    core_sha256::sha256_state state;
    core_sha256::sha256_update(state, reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
    return core_sha256::sha256_finish(state);
}
std::vector<std::int32_t> integers(const Json & value) {
    check(value.kind == Json::array_t, "Expected fixture integer list");
    std::vector<std::int32_t> result;
    for (const auto & item : value.a) {
        check(item.kind == Json::int_t && item.i >= -1 && item.i <= 2048, "Invalid fixture token index");
        result.push_back(static_cast<std::int32_t>(item.i));
    }
    return result;
}

void lemma_controls() {
    EditTree identity;
    EditTree suffix;
    suffix.is_match = false;
    suffix.original = "en";
    suffix.replacement = "";
    EditTree strip;
    strip.suffix_length = 2;
    strip.suffix_tree = 1;
    std::vector<EditTree> trees{ identity, suffix, strip };
    validate_edit_trees(trees);
    auto result = apply_lemma(trees, 2, "Äpfelen");
    check(result.applicable && result.applied == "Äpfel" && result.lemma == "Äpfel", "Unicode affix slicing");
    result = apply_lemma(trees, 2, "😀en");
    check(result.applicable && result.lemma == "😀", "Astral scalar affix slicing");
    result = apply_lemma(trees, 2, "gehen", "kept");
    check(result.applicable && result.applied == "geh" && result.lemma == "kept", "Preserve existing lemma");
    result = apply_lemma(trees, 2, "am");
    check(!result.applicable && result.applied.empty() && result.lemma == "am",
          "Failed substitution falls back to orth");
    check(!apply_lemma(trees, 2, "x").applicable, "Oversized affix is not applicable");
    check(apply_lemma(trees, 0, "").applicable, "Empty identity application");
    const std::string nul("a\0b", 3);
    check(apply_lemma(trees, 0, nul).lemma == nul, "Counted NUL in lemma");
    for (const auto & bad :
         std::vector<std::string>{ "\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x82", "\x80" })
        rejects([&] { (void)apply_lemma(trees, 0, bad); }, "Malformed UTF-8 rejected");
    rejects([&] { (void)apply_lemma(trees, 3, "x"); }, "Unknown tree rejected");
    trees[0].prefix_tree = 0;
    rejects([&] { validate_edit_trees(trees); }, "Tree cycle rejected");
    rejects([&] { (void)apply_lemma(trees, 0, ""); }, "Unvalidated tree cycle bounded");
    trees[0].prefix_tree = EditTree::none;
    trees[0].prefix_length = 1;
    rejects([&] { validate_edit_trees(trees); }, "Missing affix child rejected");
    trees[0].prefix_length = 0;
    trees[0].prefix_tree = 100;
    rejects([&] { validate_edit_trees(trees); }, "Out-of-range child rejected");
    std::vector<EditTree> deep(65);
    for (std::uint32_t i = 0; i < 64; ++i) deep[i].prefix_tree = i + 1;
    rejects([&] { validate_edit_trees(deep); }, "Overdeep edit tree rejected");
}

const std::vector<std::string> controls_moves{ "S", "D", "L-dep", "R-dep", "B-ROOT" };

void parser_controls() {
    ParserState empty(0, controls_moves);
    check(empty.is_final() && empty.finish({}).heads.empty(), "Explicit empty parse");
    check(empty.valid_actions() == std::vector<std::uint8_t>(5, 0), "No transitions after final");
    rejects([&] { empty.step(0); }, "Final state rejects transition");
    ParserState pair(2, controls_moves);
    check(pair.features() == std::array<std::int32_t, 8>{ 0, 1, -1, -1, -1, -1, -1, -1 }, "Initial parser features");
    check(pair.valid_actions() == std::vector<std::uint8_t>{ 1, 0, 0, 0, 1 }, "Initial parser mask");
    rejects([&] { pair.step(2); }, "Illegal left transition rejected");
    rejects([&] { pair.finish({ 0, 0 }); }, "Partial parse cannot publish");
    pair.step(0);
    check(pair.valid_actions() == std::vector<std::uint8_t>{ 0, 1, 1, 1, 0 }, "One-buffer-token action mask");
    pair.step(2);
    check(pair.features()[5] == 0, "Latest left child enters buffer feature");
    pair.step(0);
    pair.step(1);
    const auto parsed = pair.finish({ 0, 0 });
    check(parsed.heads == std::vector<std::int32_t>{ 1, 1 }, "Left arc/root heads");
    check(parsed.dependencies == std::vector<std::string>{ "dep", "ROOT" }, "Left arc/root labels");
    check(parsed.sentence_starts == std::vector<std::uint8_t>{ 1, 0 }, "Root subtree sentence boundary");

    ParserState unshift(3, controls_moves);
    unshift.step(0);
    unshift.step(0);
    unshift.step(1);
    check(unshift.stack() == std::vector<std::int32_t>{ 0 } && unshift.queue() == std::vector<std::int32_t>{ 1, 2 },
          "Headless reduce re-buffers top token");
    check(!unshift.valid_actions()[0], "Rebuffered token cannot shift again without structural change");
    unshift.step(2);
    check(unshift.valid_actions()[0], "Left transition permits shifting changed state");
    unshift.step(0);
    unshift.step(3);
    check(unshift.features()[3] == 1, "Right transition pushes child");
    unshift.step(1);
    unshift.step(1);
    check(unshift.finish({ 0, 0, 0 }).heads == std::vector<std::int32_t>{ 1, 1, 1 }, "Unshift sequence closes");

    ParserState boundary(3, controls_moves);
    boundary.step(4);
    boundary.step(0);
    check(!boundary.valid_actions()[2] && !boundary.valid_actions()[3], "Arc cannot cross declared sentence boundary");
    boundary.step(1);
    boundary.step(0);
    boundary.step(3);
    boundary.step(1);
    boundary.step(1);
    check(boundary.finish({ 0, 0, 0 }).sentence_starts == std::vector<std::uint8_t>{ 1, 1, 0 },
          "Multiple sentence roots");
    ParserState constrained(2, controls_moves, { 1, -1 });
    check(!constrained.valid_actions()[4], "Explicit forbidden sentence start respected");
    constrained.step(0);
    check(!constrained.valid_actions()[1], "Root reduce respects forbidden next sentence start");
    rejects([&] { ParserState(2049, controls_moves); }, "Token bound rejected");
    rejects([&] { ParserState(1, { "S", "D", "B-ROOT", "L-subtok" }); }, "Token learning rejected");
    rejects([&] { ParserState(1, { "S", "D", "B-ROOT", "L-a||b||c" }); }, "Decorated label shape rejected");
}

void deprojection_controls() {
    ParsedDocument document{ { 3, 3, 1, 3 }, { "oa||mo", "mo", "nk", "ROOT" }, {} };
    deprojectivize(document, { 0, 0, 0, 0 });
    check(document.heads == std::vector<std::int32_t>{ 1, 3, 1, 3 } && document.dependencies[0] == "oa",
          "Decorated arc reattaches to target label");
    check(document.sentence_starts == std::vector<std::uint8_t>{ 1, 0, 0, 0 }, "Deprojection recomputes boundaries");
    document = { { 3, 3, 1, 3 }, { "oa||mo", "mo", "mo", "ROOT" }, {} };
    deprojectivize(document, { 0, 1, 0, 0 });
    check(document.heads[0] == 3 && document.dependencies[0] == "oa", "Whitespace subtree skipped during search");
    document = { { 3, 3, 3, 3 }, { "oa||mo", "mo", "mo", "ROOT" }, {} };
    deprojectivize(document, { 0, 0, 0, 0 });
    check(document.heads[0] == 1, "Leftmost equal-depth target wins");
    document = { { 3, 0, 3, 3 }, { "oa||mo", "mo", "nk", "ROOT" }, {} };
    deprojectivize(document, { 0, 0, 0, 0 });
    check(document.heads[0] == 3, "Dependent subtree excluded to prevent self-attachment");
    document = { { 1, 0 }, { "dep", "dep" }, {} };
    rejects([&] { deprojectivize(document, { 0, 0 }); }, "Cyclic dependencies fail closed");
}

void cancellation_controls() {
    // A longest-admitted chain makes the first ancestor walk substantial.
    // Cancellation must be observed inside that walk, before reconstruction.
    ParsedDocument chain;
    chain.heads.resize(2048);
    chain.dependencies.assign(2048, "dep");
    for (std::size_t i = 0; i < chain.heads.size(); ++i)
        chain.heads[i] = static_cast<std::int32_t>(std::min(i + 1, chain.heads.size() - 1));
    chain.dependencies.back() = "ROOT";
    std::size_t polls = 0;
    rejects<NumericCancelled>(
        [&] { deprojectivize(chain, std::vector<std::uint8_t>(2048), [&] { return ++polls == 64; }); },
        "Ancestor validation cancellation propagates");
    check(polls == 64 && chain.sentence_starts.empty(), "Ancestor cancellation stops without sentence output");

    // First complete a decorated token, then cancel inside the next broad
    // child search. This cannot pass merely by checking cancellation on entry.
    ParsedDocument broad;
    broad.heads.assign(2048, 0);
    broad.dependencies.assign(2048, "dep||missing");
    broad.dependencies[0] = "ROOT";
    std::size_t after_first = 0;
    rejects<NumericCancelled>(
        [&] {
            deprojectivize(broad, std::vector<std::uint8_t>(2048),
                           [&] { return broad.dependencies[1] == "dep" && ++after_first == 5; });
        },
        "Repeated decorated dependency search cancellation propagates");
    check(after_first == 5 && broad.dependencies[1] == "dep" && broad.dependencies[2] == "dep||missing",
          "Cancellation interrupts a later decorated search");
    check(broad.sentence_starts.empty(), "Cancelled reconstruction publishes no sentence result");

    // Let every decorated label finish, then cancel the final forest/extent
    // recomputation. The caller must discard the temporary in-place document.
    ParsedDocument final_edges;
    final_edges.heads.assign(64, 63);
    final_edges.dependencies.assign(64, "dep||missing");
    final_edges.dependencies.back() = "ROOT";
    final_edges.sentence_starts = { 7 };
    std::size_t after_labels = 0;
    rejects<NumericCancelled>(
        [&] {
            deprojectivize(final_edges, std::vector<std::uint8_t>(64),
                           [&] { return final_edges.dependencies[62] == "dep" && ++after_labels == 4; });
        },
        "Final forest recomputation cancellation propagates");
    check(after_labels == 4 && final_edges.sentence_starts == std::vector<std::uint8_t>{ 7 },
          "Final recomputation stops before sentence assignment");

    ParserState state(2, controls_moves);
    state.step(0);
    state.step(3);
    state.step(1);
    state.step(1);
    rejects<NumericCancelled>([&] { (void)state.finish({ 0, 0 }, [] { return true; }); },
                              "Final parser state forwards cancellation");
    check(state.is_final() && state.finish({ 0, 0 }).heads == std::vector<std::int32_t>{ 0, 0 },
          "Final parser state remains usable after cancelled output");
    ParsedDocument empty;
    rejects<NumericCancelled>([&] { deprojectivize(empty, {}, [] { return true; }); },
                              "Empty reconstruction honors cancellation");
}

Matrix matrix(const Json & fixture, const std::string & binary, const std::string & name) {
    const auto & item = get(get(fixture, "traces"), name.c_str());
    const auto & shape = get(item, "shape");
    check(shape.kind == Json::array_t && shape.a.size() == 2, "Invalid score matrix shape");
    const auto offset = number(get(item, "offset_bytes"));
    const auto size = number(get(item, "size_bytes"));
    Matrix result{ number(shape.a[0]), number(shape.a[1]), {} };
    check(size == result.rows * result.cols * 4 && offset <= binary.size() && size <= binary.size() - offset,
          "Invalid score matrix extent");
    check(get(item, "dtype").s == "float32-le" && get(item, "order").s == "C" && get(item, "file").s == "traces.f32",
          "Unsupported score matrix encoding");
    check(sha256(binary.substr(offset, size)) == get(item, "sha256").s, "Changed score matrix bytes");
    result.data.resize(result.rows * result.cols);
    for (std::size_t i = 0; i < result.data.size(); ++i) {
        std::uint32_t bits = 0;
        for (std::size_t byte = 0; byte < 4; ++byte)
            bits |= static_cast<std::uint32_t>(static_cast<unsigned char>(binary[offset + 4 * i + byte])) << (8 * byte);
        std::memcpy(&result.data[i], &bits, sizeof(bits));
    }
    return result;
}

void fixture_replay(const std::string & resource_path, const std::string & fixture_path) {
    const auto resource_text = read(resource_path);
    check(sha256(resource_text) == "b42496cc373ada57a67ed2f4c2c25f8d22f80c08db5841a7cdb91c8a6e18562a",
          "Changed model resources");
    const auto resources = DecodeResources::from_json(resource_text);
    const auto manifest_text = read(fixture_path + "/manifest.json");
    check(sha256(manifest_text) == "9a3bb6bae6114adcf579840834ddcb64dea4e0aaa8361718dc7a551d663c8f7c",
          "Changed development oracle");
    const auto manifest = core_json_value::parser(manifest_text).parse();
    check(get(manifest, "split").s == "development" && get(manifest, "complete").b &&
              !get(manifest, "acceptance_outputs_used_for_tuning").b,
          "Invalid development qualification split");
    std::size_t cases = 0, token_count = 0, step_count = 0, mask_count = 0, fallback_count = 0;
    std::size_t whitespace_count = 0, sentence_count = 0;
    for (const auto & entry : get(manifest, "cases").a) {
        const auto id = get(entry, "id").s;
        check(!id.empty() && id.find('/') == std::string::npos && id.find("..") == std::string::npos,
              "Invalid case path");
        for (const auto & file : get(entry, "files").a) {
            const auto bytes = read(fixture_path + '/' + get(file, "path").s);
            check(bytes.size() == number(get(file, "size_bytes")) && sha256(bytes) == get(file, "sha256").s,
                  "Changed retained fixture file: " + id);
        }
        const auto fixture = core_json_value::parser(read(fixture_path + '/' + id + "/case.json")).parse();
        const auto binary = read(fixture_path + '/' + id + "/traces.f32");
        const auto & annotations = get(fixture, "annotations");
        const auto & expected_tokens = get(annotations, "tokens");
        const auto & steps = get(fixture, "parser_steps");
        std::vector<DecodeToken> tokens;
        for (const auto & token : expected_tokens.a)
            tokens.push_back({ get(token, "text").s, get(token, "is_space").b, {} });
        NumericOutput scores;
        scores.tagger = matrix(fixture, binary, "tagger.scores");
        scores.morphologizer = matrix(fixture, binary, "morphologizer.scores");
        scores.lemmatizer = matrix(fixture, binary, "lemmatizer.scores");
        const auto decoded = decode_annotations(
            resources, scores, tokens,
            [&](const std::array<std::int32_t, 8> & features, std::size_t step) {
                check(step < steps.a.size(), id + ": extra native parser step");
                const auto expected = integers(get(steps.a[step], "feature_token_ids"));
                check(std::vector<std::int32_t>(features.begin(), features.end()) == expected,
                      id + ": native state features differ");
                return matrix(fixture, binary, "parser.step." + std::to_string(step) + ".scores").data;
            },
            [&](const ParserStep & step) {
                const auto & expected = steps.a[step.step];
                check(step.chosen_action == number(get(expected, "chosen_action")), id + ": chosen transition differs");
                check(resources.moves[step.chosen_action] == get(expected, "chosen_name").s,
                      id + ": chosen name differs");
                const auto & valid = get(expected, "valid_actions");
                check(valid.a.size() == step.valid_actions.size(), id + ": action mask width differs");
                for (std::size_t action = 0; action < valid.a.size(); ++action) {
                    check(valid.a[action].b == (step.valid_actions[action] != 0), id + ": legal-action mask differs");
                    ++mask_count;
                }
                check(step.stack_before == integers(get(expected, "stack_before")), id + ": stack differs");
                check(step.queue_before == integers(get(expected, "queue_before")), id + ": queue differs");
                std::multiset<std::pair<std::int32_t, std::int32_t>> actual_arcs, expected_arcs;
                for (const auto & arc : step.arcs_before) actual_arcs.insert({ arc.head, arc.child });
                for (const auto & arc : get(expected, "arcs_before").a)
                    expected_arcs.insert({ static_cast<std::int32_t>(number(get(arc, "head"))),
                                           static_cast<std::int32_t>(number(get(arc, "child"))) });
                check(actual_arcs == expected_arcs, id + ": arc endpoints differ");
            });
        check(decoded.parser_steps == steps.a.size() && decoded.tokens.size() == expected_tokens.a.size(),
              id + ": output coverage differs");
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto & expected = expected_tokens.a[i];
            const auto & actual = decoded.tokens[i];
            const auto where = id + "/" + std::to_string(i);
            check(actual.tag == get(expected, "tag").s, where + ": tag differs");
            check(actual.pos == get(expected, "pos").s, where + ": POS differs");
            check(actual.morph == get(expected, "morph").s, where + ": morphology differs");
            check(actual.lemma == get(expected, "lemma").s, where + ": lemma differs");
            check(actual.dep == get(expected, "dep").s, where + ": dependency label differs");
            check(actual.head == static_cast<std::int32_t>(number(get(expected, "head"))),
                  where + ": dependency head differs");
            check(actual.is_sent_start == get(expected, "is_sent_start").b, where + ": sentence boundary differs");
            const auto & lemma = get(fixture, "lemma_decisions").a[i];
            check(actual.lemma_class == number(get(lemma, "class")) &&
                      actual.lemma_decision.tree == number(get(lemma, "tree")),
                  where + ": lemma class/tree mapping differs");
            check(actual.lemma_decision.applicable == get(lemma, "applicable").b,
                  where + ": edit-tree applicability differs");
            check(!actual.lemma_decision.applicable || actual.lemma_decision.applied == get(lemma, "applied").s,
                  where + ": applied edit tree differs");
            fallback_count += !actual.lemma_decision.applicable;
            whitespace_count += tokens[i].is_space;
        }
        const auto & sentences = get(annotations, "sentences");
        check(decoded.sentences.size() == sentences.a.size(), id + ": sentence count differs");
        for (std::size_t i = 0; i < sentences.a.size(); ++i)
            check(decoded.sentences[i].start_token == number(get(sentences.a[i], "start_token")) &&
                      decoded.sentences[i].end_token == number(get(sentences.a[i], "end_token")),
                  id + ": sentence token range differs");
        ++cases;
        token_count += tokens.size();
        step_count += decoded.parser_steps;
        sentence_count += decoded.sentences.size();
    }
    check(cases == 16 && token_count == 113 && step_count == 244 && mask_count == 33428,
          "Incomplete decoder reference denominator");
    std::cout << "SPACY_DECODE_FROZEN_SCORE_REPLAY cases=" << cases << " tokens=" << token_count
              << " parser_steps=" << step_count << " mask_decisions=" << mask_count << " sentences=" << sentence_count
              << " whitespace_tokens=" << whitespace_count << " lemma_fallbacks=" << fallback_count
              << " native_model_forward=0 raw_text_analysis=0 acceptance_used=0\n";
}

int run(int argc, char ** argv) {
    lemma_controls();
    parser_controls();
    deprojection_controls();
    cancellation_controls();
    rejects([] { (void)DecodeResources::from_json("{\"format_version\":1,\"format_version\":1}"); },
            "Duplicate resource keys rejected");
    rejects([] { (void)DecodeResources::from_json("{}"); }, "Missing decoder resources rejected");
    if (argc == 3)
        fixture_replay(argv[1], argv[2]);
    else
        check(argc == 1, "Usage: test_spacy_decode [resources.json development-fixture-directory]");
    std::cout << "SPACY_DECODE_CONTROLS assertions=" << assertions << " actual_model_inference=0\n";
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "SPACY_DECODE FAILED: " << error.what() << '\n';
        core_util::clean_exit(1);
    }
}
