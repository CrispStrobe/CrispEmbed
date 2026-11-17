#ifndef CRISPEMBED_SPACY_DECODE_H
#define CRISPEMBED_SPACY_DECODE_H

#include "spacy_numeric.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

// Internal, inference-only German spaCy decoders. The resource owner must
// authenticate resources.json against the admitted model before constructing
// DecodeResources. This module performs no I/O, tokenization, or downloads.
namespace crispembed {
namespace spacy {

struct EditTree {
    static constexpr std::uint32_t none = std::numeric_limits<std::uint32_t>::max();
    bool is_match = true;
    std::uint32_t prefix_length = 0;
    std::uint32_t suffix_length = 0;
    std::uint32_t prefix_tree = none;
    std::uint32_t suffix_tree = none;
    std::string original;
    std::string replacement;
};

struct MorphLabel {
    std::string pos;
    std::string morph;
};

struct DecodeResources {
    std::vector<std::string> tags;
    std::vector<MorphLabel> morphology;
    std::vector<std::string> moves;
    std::vector<std::uint32_t> class_to_tree;
    std::vector<EditTree> trees;

    // Closed version-1 label profile (55/671/137/1311 classes, 6092 trees).
    // Rejects malformed JSON, duplicates, unsupported policies and invalid
    // tree references/cycles. Retained resources must remain immutable.
    static DecodeResources from_json(const std::string & authenticated_json);
};

struct LemmaDecision {
    std::uint32_t tree = 0;
    bool applicable = false;
    std::string applied;
    std::string lemma;
};

// Slices affixes by Unicode scalar count, not UTF-8 bytes or UTF-16 units.
// A nonapplicable winner falls back to the unchanged token. An existing
// nonempty lemma is preserved. Invalid UTF-8/resources fail explicitly.
LemmaDecision apply_lemma(const std::vector<EditTree> & trees, std::uint32_t tree, const std::string & token,
                          const std::string & existing_lemma = {});
void validate_edit_trees(const std::vector<EditTree> & trees);

struct DependencyArc {
    std::int32_t head = -1;
    std::int32_t child = -1;
    std::string label;
};

struct ParsedDocument {
    std::vector<std::int32_t> heads;
    std::vector<std::string> dependencies;
    std::vector<std::uint8_t> sentence_starts;
};

// The source deprojectivizer visits decorated dependencies left-to-right,
// searches breadth-first in source child order and skips spaces/dependent
// subtrees. Annotation and sentence boundaries are recomputed after it.
// Cancellation may interrupt in-place reconstruction; discard this temporary
// document on failure. ParserState::finish and decode_annotations publish only
// a fully completed owned result, and remain usable after cancellation.
void deprojectivize(ParsedDocument & document, const std::vector<std::uint8_t> & is_space,
                    const CancelCheck & cancelled = {});

class ParserState {
public:
    ParserState(std::size_t token_count, const std::vector<std::string> & ordered_moves,
                const std::vector<std::int8_t> & preset_sentence_starts = {});

    bool is_final() const;
    std::array<std::int32_t, 8> features() const;
    std::vector<std::uint8_t> valid_actions() const;
    std::vector<std::int32_t> stack() const;
    std::vector<std::int32_t> queue() const;
    std::vector<DependencyArc> arcs() const;
    void step(std::size_t action);
    // Only final states can publish annotations. Missing heads become ROOT.
    ParsedDocument finish(const std::vector<std::uint8_t> & is_space, const CancelCheck & cancelled = {}) const;

private:
    struct Move {
        char type = 0;
        std::string label;
    };
    std::vector<Move> moves_;
    std::vector<std::int32_t> stack_;
    std::vector<std::int32_t> rebuffer_;
    std::vector<std::int32_t> heads_;
    std::vector<std::string> dependencies_;
    std::unordered_map<std::int32_t, std::vector<DependencyArc>> left_arcs_;
    std::unordered_map<std::int32_t, std::vector<DependencyArc>> right_arcs_;
    std::vector<std::uint8_t> unshiftable_;
    std::vector<std::int8_t> sentence_starts_;
    std::size_t next_buffer_ = 0;
    std::size_t steps_ = 0;

    std::int32_t s(std::size_t offset) const;
    std::int32_t b(std::size_t offset) const;
    std::size_t buffer_length() const;
    std::int32_t latest_child(std::int32_t head, bool left) const;
    bool valid(const Move & move) const;
    void push();
    void add_arc(std::int32_t head, std::int32_t child, const std::string & label);
};

struct DecodeToken {
    std::string text;
    bool is_space = false;
    std::string existing_lemma;
};

struct TokenAnnotation {
    std::string tag;
    std::string pos;
    std::string morph;
    std::string lemma;
    std::string dep;
    std::int32_t head = -1;
    bool is_sent_start = false;
    std::size_t tag_class = 0;
    std::size_t morph_class = 0;
    std::size_t lemma_class = 0;
    LemmaDecision lemma_decision;
};

struct SentenceRange {
    std::size_t start_token = 0;
    std::size_t end_token = 0;
};

struct DecodeOutput {
    std::vector<TokenAnnotation> tokens;
    std::vector<SentenceRange> sentences;
    std::size_t parser_steps = 0;
};

struct ParserStep {
    std::size_t step = 0;
    std::array<std::int32_t, 8> features{};
    std::vector<std::uint8_t> valid_actions;
    std::size_t chosen_action = 0;
    std::vector<float> scores;
    std::vector<std::int32_t> stack_before;
    std::vector<std::int32_t> queue_before;
    std::vector<DependencyArc> arcs_before;
};

// The scorer receives native-generated state features. Production callers
// wrap NumericModel::parser_scores with the same document's parser_cache.
// The callback may prefix numeric trace names with parser.step.<step>.
using ParserScorer = std::function<std::vector<float>(const std::array<std::int32_t, 8> &, std::size_t)>;
// The step view is borrowed for the callback duration; copy to retain it.
using ParserTrace = std::function<void(const ParserStep &)>;

// Greedy first-maximum choices, five native transitions, deprojection, tag/
// morph/POS assignment, edit trees and the two pinned whitespace rules.
// Bounds: <=2048 tokens and <=8*n+32 transitions, with no truncation or partial
// successful output. Cancellation throws NumericCancelled. Empty input is an
// explicit empty output and invokes neither scorer nor parser trace.
DecodeOutput decode_annotations(const DecodeResources & resources, const NumericOutput & scores,
                                const std::vector<DecodeToken> & tokens, const ParserScorer & scorer,
                                const ParserTrace & trace = {}, const CancelCheck & cancelled = {});

} // namespace spacy
} // namespace crispembed

#endif
