// Inference algorithms follow spaCy 3.8.18 at
// 43311999028032fd81bef77a75fdb4fa96803c98 (MIT): arc_eager.pyx,
// _state.pxd, nonproj.pyx, token.pyx and edit_trees.pyx. The original notice
// is installed from tools/spacy/notices/spaCy-LICENSE.
#include "spacy_decode.h"

#include "core/json_value.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace crispembed {
namespace spacy {
namespace {

using Json = core_json_value::value;
constexpr std::size_t max_tokens = 2048;
constexpr std::size_t max_resource_bytes = 32 * 1024 * 1024;
constexpr std::size_t max_lemma_bytes = 1024 * 1024;

void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

std::vector<std::size_t> boundaries(const std::string & text) {
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < text.size();) {
        result.push_back(i);
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t length = 1;
        std::uint32_t scalar = lead;
        if (lead >= 0xc2 && lead <= 0xdf) {
            length = 2;
            scalar = lead & 0x1f;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            length = 3;
            scalar = lead & 0x0f;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            length = 4;
            scalar = lead & 0x07;
        } else
            require(lead < 0x80, "Invalid UTF-8 in spaCy decoder input");
        require(length <= text.size() - i, "Truncated UTF-8 in spaCy decoder input");
        for (std::size_t j = 1; j < length; ++j) {
            const auto tail = static_cast<unsigned char>(text[i + j]);
            require((tail & 0xc0) == 0x80, "Invalid UTF-8 continuation in spaCy decoder input");
            scalar = (scalar << 6) | (tail & 0x3f);
        }
        require((length != 2 || scalar >= 0x80) && (length != 3 || scalar >= 0x800) &&
                    (length != 4 || scalar >= 0x10000) && scalar <= 0x10ffff && !(scalar >= 0xd800 && scalar <= 0xdfff),
                "Nonscalar or overlong UTF-8 in spaCy decoder input");
        i += length;
    }
    result.push_back(text.size());
    return result;
}

void unique_keys(const Json & value) {
    if (value.kind == Json::object_t) {
        std::set<std::string> names;
        for (const auto & item : value.o) {
            require(names.insert(item.first).second, "Duplicate spaCy resource JSON key");
            unique_keys(item.second);
        }
    } else if (value.kind == Json::array_t) {
        for (const auto & item : value.a) unique_keys(item);
    }
}

const Json & member(const Json & object, const char * name) {
    require(object.kind == Json::object_t, "Expected spaCy resource object");
    const auto * value = object.get(name);
    require(value != nullptr, "Missing spaCy decoder resource field");
    return *value;
}

void keys(const Json & object, std::initializer_list<const char *> names) {
    require(object.kind == Json::object_t && object.o.size() == names.size(), "Unexpected spaCy decoder resource keys");
    for (auto name : names) (void)member(object, name);
}

const std::string & string(const Json & value, bool empty = false) {
    require(value.kind == Json::string_t && (empty || !value.s.empty()), "Expected spaCy decoder resource string");
    require(value.s.size() <= 16384, "Oversized spaCy decoder resource string");
    (void)boundaries(value.s);
    return value.s;
}

std::uint32_t integer(const Json & value, std::uint32_t maximum) {
    require(value.kind == Json::int_t && value.i >= 0 && static_cast<unsigned long long>(value.i) <= maximum,
            "Invalid spaCy decoder resource integer");
    return static_cast<std::uint32_t>(value.i);
}

void boolean(const Json & value, bool expected) {
    require(value.kind == Json::bool_t && value.b == expected, "Unsupported spaCy decoder Boolean policy");
}

std::vector<std::string> strings(const Json & value, std::size_t count) {
    require(value.kind == Json::array_t && value.a.size() == count, "Changed spaCy decoder label count");
    std::vector<std::string> result;
    std::set<std::string> seen;
    for (const auto & entry : value.a) {
        result.push_back(string(entry));
        require(seen.insert(result.back()).second, "Duplicate spaCy decoder label");
    }
    return result;
}

void decimal_id(const Json & value) {
    const auto & text = string(value);
    require(text == "0" || text.front() != '0', "Noncanonical spaCy string ID");
    std::uint64_t number = 0;
    for (const char c : text) {
        require(c >= '0' && c <= '9', "Invalid spaCy string ID");
        const auto digit = static_cast<std::uint64_t>(c - '0');
        require(number <= (std::numeric_limits<std::uint64_t>::max() - digit) / 10, "Overflowing spaCy string ID");
        number = number * 10 + digit;
    }
}

void check_cancel(const CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw NumericCancelled();
}

void check_forest(const ParsedDocument & document, const CancelCheck & cancelled) {
    check_cancel(cancelled);
    const auto n = document.heads.size();
    require(n <= max_tokens && document.dependencies.size() == n, "Invalid spaCy dependency document");
    for (std::size_t i = 0; i < n; ++i) {
        check_cancel(cancelled);
        auto at = static_cast<std::int32_t>(i);
        std::size_t depth = 0;
        while (true) {
            check_cancel(cancelled);
            require(at >= 0 && static_cast<std::size_t>(at) < n, "Out-of-range spaCy dependency head");
            const auto head = document.heads[at];
            if (head == at) break;
            require(++depth <= n, "Cycle in spaCy dependency document");
            at = head;
        }
    }
}

struct Edges {
    std::vector<std::int32_t> left;
    std::vector<std::int32_t> right;
};

Edges subtree_edges(const ParsedDocument & document, const CancelCheck & cancelled) {
    check_forest(document, cancelled);
    const auto n = document.heads.size();
    Edges edges;
    edges.left.resize(n);
    edges.right.resize(n);
    std::iota(edges.left.begin(), edges.left.end(), 0);
    edges.right = edges.left;
    // Preserve doc.pyx's two sweeps and stopping condition, including its
    // bounded retry for nonprojective forests. Those extents also bound the
    // source Token.children iterator used during deprojection.
    for (std::size_t attempt = 0; attempt < 12; ++attempt) {
        check_cancel(cancelled);
        const auto update = [&](std::size_t child) {
            check_cancel(cancelled);
            const auto head = document.heads[child];
            edges.left[head] = std::min(edges.left[head], edges.left[child]);
            edges.right[head] = std::max(edges.right[head], edges.right[child]);
        };
        for (std::size_t i = 0; i < n; ++i) update(i);
        for (std::size_t i = n; i > 0; --i) update(i - 1);
        std::set<std::int32_t> starts;
        for (std::size_t i = 0; i < n; ++i) {
            check_cancel(cancelled);
            if (document.heads[i] == static_cast<std::int32_t>(i)) starts.insert(edges.left[i]);
        }
        std::size_t begin = 0;
        bool within = true;
        for (std::size_t i = 0; i < n && within; ++i) {
            check_cancel(cancelled);
            if ((i > 0 && starts.count(static_cast<std::int32_t>(i))) || i + 1 == n) {
                for (std::size_t j = begin; j < i; ++j) {
                    check_cancel(cancelled);
                    if (document.heads[j] < static_cast<std::int32_t>(begin) ||
                        document.heads[j] > static_cast<std::int32_t>(i)) {
                        within = false;
                        break;
                    }
                }
                begin = i;
            }
        }
        if (within) break;
    }
    return edges;
}

void set_sentence_starts(ParsedDocument & document, const CancelCheck & cancelled) {
    const auto edges = subtree_edges(document, cancelled);
    document.sentence_starts.assign(document.heads.size(), 0);
    for (std::size_t i = 0; i < document.heads.size(); ++i) {
        check_cancel(cancelled);
        if (document.heads[i] == static_cast<std::int32_t>(i)) document.sentence_starts[edges.left[i]] = 1;
    }
    require(document.heads.empty() || document.sentence_starts[0], "SpaCy parse has no initial sentence boundary");
}

std::size_t first_maximum(const Matrix & matrix, std::size_t row) {
    const auto offset = row * matrix.cols;
    std::size_t winner = 0;
    for (std::size_t i = 1; i < matrix.cols; ++i) {
        if (matrix.data[offset + i] > matrix.data[offset + winner]) winner = i;
    }
    return winner;
}

void check_matrix(const Matrix & matrix, std::size_t rows, std::size_t columns) {
    require(matrix.rows == rows && matrix.cols == columns && matrix.data.size() == rows * columns,
            "Invalid spaCy classifier matrix shape");
    for (const float value : matrix.data) require(std::isfinite(value), "Nonfinite spaCy classifier score");
}

} // namespace

DecodeResources DecodeResources::from_json(const std::string & text) {
    require(text.size() <= max_resource_bytes, "SpaCy resources exceed decoder admission bound");
    const auto root = core_json_value::parser(text).parse();
    unique_keys(root);
    require(integer(member(root, "format_version"), 1) == 1 && string(member(root, "language")) == "de",
            "Unsupported spaCy decoder resource identity");
    require(strings(member(root, "component_order"), 6) == std::vector<std::string>{ "tok2vec", "tagger",
                                                                                     "morphologizer", "parser",
                                                                                     "lemmatizer", "attribute_ruler" },
            "Unsupported spaCy component order");
    const auto & labels = member(root, "labels");
    keys(labels, { "tagger", "morphologizer", "parser", "lemmatizer", "attribute_ruler" });
    DecodeResources result;
    result.tags = strings(member(labels, "tagger"), 55);
    const auto & morph = member(labels, "morphologizer");
    keys(morph, { "labels", "morph", "pos", "overwrite", "extend" });
    boolean(member(morph, "overwrite"), true);
    boolean(member(morph, "extend"), false);
    const auto morph_labels = strings(member(morph, "labels"), 671);
    const auto & morph_map = member(morph, "morph");
    const auto & pos_map = member(morph, "pos");
    require(morph_map.kind == Json::object_t && morph_map.o.size() == 671 && pos_map.kind == Json::object_t &&
                pos_map.o.size() == 671,
            "Changed spaCy morphology map");
    const std::map<std::string, std::uint32_t> pos_ids = {
        { "ADJ", 84 },   { "ADP", 85 },   { "ADV", 86 },   { "AUX", 87 },  { "CCONJ", 89 }, { "DET", 90 },
        { "INTJ", 91 },  { "NOUN", 92 },  { "NUM", 93 },   { "PART", 94 }, { "PRON", 95 },  { "PROPN", 96 },
        { "PUNCT", 97 }, { "SCONJ", 98 }, { "VERB", 100 }, { "X", 101 },   { "SPACE", 103 }
    };
    for (const auto & label : morph_labels) {
        MorphLabel decoded;
        std::set<std::string> field_names;
        for (std::size_t at = 0; at < label.size();) {
            const auto end = label.find('|', at);
            const auto field = label.substr(at, end == std::string::npos ? end : end - at);
            const auto equal = field.find('=');
            require(equal != std::string::npos && equal > 0 && equal + 1 < field.size(), "Malformed spaCy morph label");
            const auto name = field.substr(0, equal);
            require(field_names.insert(name).second, "Duplicate spaCy morph attribute");
            if (name == "POS")
                decoded.pos = field.substr(equal + 1);
            else {
                if (!decoded.morph.empty()) decoded.morph += '|';
                decoded.morph += field;
            }
            if (end == std::string::npos) break;
            at = end + 1;
        }
        const auto pos = pos_ids.find(decoded.pos);
        require(pos != pos_ids.end() && integer(member(pos_map, label.c_str()), 103) == pos->second &&
                    string(member(morph_map, label.c_str()), true) == decoded.morph,
                "Inconsistent spaCy morphology/POS mapping");
        result.morphology.push_back(std::move(decoded));
    }
    const auto & parser = member(labels, "parser");
    keys(parser, { "moves", "labels", "features", "cfg", "moves_serialized_json" });
    result.moves = strings(member(parser, "moves"), 137);
    (void)strings(member(parser, "labels"), 42);
    require(strings(member(parser, "features"), 8) ==
                std::vector<std::string>{ "B0", "B1", "S0", "S1", "S2", "L(B0,1)", "L(S0,1)", "R(S0,1)" },
            "Unsupported spaCy parser feature order");
    const auto & cfg = member(parser, "cfg");
    keys(cfg, { "beam_density", "beam_update_prob", "beam_width", "incorrect_spans_key", "learn_tokens",
                "min_action_freq", "moves", "multitasks", "update_with_oracle_cut_size" });
    require(integer(member(cfg, "beam_width"), 1) == 1 && integer(member(cfg, "min_action_freq"), 30) == 30 &&
                integer(member(cfg, "update_with_oracle_cut_size"), 100) == 100,
            "Unsupported spaCy parser configuration");
    boolean(member(cfg, "learn_tokens"), false);
    for (const char * key : { "beam_density", "beam_update_prob" }) {
        const auto & value = member(cfg, key);
        require(value.kind == Json::float_t && value.f == 0, "Unsupported spaCy beam configuration");
    }
    require(member(cfg, "incorrect_spans_key").kind == Json::null_t && member(cfg, "moves").kind == Json::null_t &&
                member(cfg, "multitasks").kind == Json::array_t && member(cfg, "multitasks").a.empty(),
            "Unsupported spaCy parser metadata");
    (void)ParserState(0, result.moves);
    const auto & lemma = member(labels, "lemmatizer");
    keys(lemma, { "class_to_tree", "trees", "top_k", "backoff", "overwrite" });
    require(integer(member(lemma, "top_k"), 1) == 1 && string(member(lemma, "backoff")) == "orth",
            "Unsupported spaCy lemma selection");
    boolean(member(lemma, "overwrite"), false);
    const auto & trees = member(lemma, "trees");
    const auto & classes = member(lemma, "class_to_tree");
    require(trees.kind == Json::array_t && trees.a.size() == 6092 && classes.kind == Json::array_t &&
                classes.a.size() == 1311,
            "Changed spaCy edit-tree inventory");
    std::set<std::uint32_t> unique_classes;
    for (const auto & item : classes.a) {
        const auto id = integer(item, 6091);
        require(unique_classes.insert(id).second, "Duplicate spaCy lemma class tree");
        result.class_to_tree.push_back(id);
    }
    for (const auto & entry : trees.a) {
        EditTree tree;
        if (entry.get("orig")) {
            keys(entry, { "orig", "subst" });
            tree.is_match = false;
            for (const char * key : { "orig", "subst" }) {
                const auto & item = member(entry, key);
                keys(item, { "id", "text" });
                decimal_id(member(item, "id"));
                const auto & value = string(member(item, "text"), true);
                require(boundaries(value).size() <= 4097, "Oversized spaCy edit-tree substitution");
                if (key[0] == 'o')
                    tree.original = value;
                else
                    tree.replacement = value;
            }
        } else {
            keys(entry, { "prefix_len", "suffix_len", "prefix_tree", "suffix_tree" });
            tree.prefix_length = integer(member(entry, "prefix_len"), 4096);
            tree.suffix_length = integer(member(entry, "suffix_len"), 4096);
            tree.prefix_tree = integer(member(entry, "prefix_tree"), EditTree::none);
            tree.suffix_tree = integer(member(entry, "suffix_tree"), EditTree::none);
        }
        result.trees.push_back(std::move(tree));
    }
    validate_edit_trees(result.trees);
    const auto & ruler = member(labels, "attribute_ruler");
    require(ruler.kind == Json::array_t && ruler.a.size() == 2, "Changed spaCy attribute rules");
    for (std::size_t i = 0; i < 2; ++i) {
        const auto & rule = ruler.a[i];
        keys(rule, { "attrs", "index", "patterns" });
        require(integer(member(rule, "index"), 0) == 0, "Unsupported spaCy attribute-rule index");
        const auto & attrs = member(rule, "attrs");
        if (i == 0) {
            keys(attrs, { "DEP" });
            require(string(member(attrs, "DEP")) == "dep", "Changed spaCy whitespace dependency rule");
        } else {
            keys(attrs, { "MORPH", "POS", "TAG" });
            require(string(member(attrs, "MORPH")) == "_" && string(member(attrs, "POS")) == "SPACE" &&
                        string(member(attrs, "TAG")) == "_SP",
                    "Changed spaCy whitespace annotation rule");
        }
        const auto & patterns = member(rule, "patterns");
        require(patterns.kind == Json::array_t && patterns.a.size() == 1 && patterns.a[0].kind == Json::array_t &&
                    patterns.a[0].a.size() == 1,
                "Unsupported spaCy attribute-rule pattern");
        const auto & pattern = patterns.a[0].a[0];
        if (i == 0) {
            keys(pattern, { "DEP", "IS_SPACE" });
            const auto & dep = member(pattern, "DEP");
            keys(dep, { "NOT_IN" });
            const auto & excluded = member(dep, "NOT_IN");
            require(excluded.kind == Json::array_t && excluded.a.size() == 1 && string(excluded.a[0], true).empty(),
                    "Changed spaCy whitespace dependency guard");
        } else
            keys(pattern, { "IS_SPACE" });
        boolean(member(pattern, "IS_SPACE"), true);
    }
    return result;
}

void validate_edit_trees(const std::vector<EditTree> & trees) {
    require(!trees.empty() && trees.size() <= 6092, "Invalid spaCy edit-tree count");
    std::vector<std::uint8_t> state(trees.size(), 0);
    std::vector<std::size_t> depths(trees.size(), 0);
    std::function<std::size_t(std::uint32_t, std::size_t)> visit = [&](std::uint32_t id, std::size_t level) {
        require(id < trees.size() && level <= 64, "Invalid spaCy edit-tree reference/depth");
        require(state[id] != 1, "Cycle in spaCy edit trees");
        if (state[id] == 2) return depths[id];
        state[id] = 1;
        const auto & tree = trees[id];
        std::size_t depth = 1;
        if (tree.is_match) {
            require(tree.prefix_length <= 4096 && tree.suffix_length <= 4096, "Oversized spaCy edit-tree affix");
            for (const auto & side : { std::make_pair(tree.prefix_tree, tree.prefix_length),
                                       std::make_pair(tree.suffix_tree, tree.suffix_length) }) {
                if (side.first == EditTree::none)
                    require(side.second == 0, "Missing spaCy edit-tree affix child");
                else
                    depth = std::max(depth, 1 + visit(side.first, level + 1));
            }
        } else {
            require(boundaries(tree.original).size() <= 4097 && boundaries(tree.replacement).size() <= 4097,
                    "Oversized spaCy edit-tree substitution");
        }
        require(depth <= 64, "SpaCy edit-tree depth exceeds admission bound");
        state[id] = 2;
        depths[id] = depth;
        return depth;
    };
    for (std::size_t i = 0; i < trees.size(); ++i) visit(static_cast<std::uint32_t>(i), 1);
}

LemmaDecision apply_lemma(const std::vector<EditTree> & trees, std::uint32_t id, const std::string & token,
                          const std::string & existing_lemma) {
    require(id < trees.size() && token.size() <= max_lemma_bytes, "Invalid spaCy lemma input/tree");
    const auto offsets = boundaries(token);
    (void)boundaries(existing_lemma);
    std::string applied;
    std::size_t visits = 0;
    auto append = [&](const std::string & value) {
        require(value.size() <= max_lemma_bytes - applied.size(), "SpaCy lemma output exceeds bound");
        applied += value;
    };
    std::function<bool(std::uint32_t, std::size_t, std::size_t, std::size_t)> apply =
        [&](std::uint32_t tree_id, std::size_t begin, std::size_t end, std::size_t depth) {
            require(tree_id < trees.size() && depth <= 64 && ++visits <= 65536, "Invalid/unbounded spaCy edit tree");
            const auto & tree = trees[tree_id];
            if (!tree.is_match) {
                if (token.compare(offsets[begin], offsets[end] - offsets[begin], tree.original) != 0) return false;
                (void)boundaries(tree.replacement);
                append(tree.replacement);
                return true;
            }
            const std::size_t prefix = tree.prefix_length, suffix = tree.suffix_length;
            if (prefix > end - begin || suffix > end - begin - prefix) return false;
            require((tree.prefix_tree != EditTree::none || prefix == 0) &&
                        (tree.suffix_tree != EditTree::none || suffix == 0),
                    "Missing spaCy edit-tree child");
            if (tree.prefix_tree != EditTree::none && !apply(tree.prefix_tree, begin, begin + prefix, depth + 1))
                return false;
            append(token.substr(offsets[begin + prefix], offsets[end - suffix] - offsets[begin + prefix]));
            if (tree.suffix_tree != EditTree::none && !apply(tree.suffix_tree, end - suffix, end, depth + 1))
                return false;
            return true;
        };
    LemmaDecision result;
    result.tree = id;
    result.applicable = apply(id, 0, offsets.size() - 1, 1);
    if (result.applicable) result.applied = std::move(applied);
    result.lemma = existing_lemma.empty() ? (result.applicable ? result.applied : token) : existing_lemma;
    return result;
}

ParserState::ParserState(std::size_t count, const std::vector<std::string> & ordered_moves,
                         const std::vector<std::int8_t> & preset) {
    require(count <= max_tokens, "SpaCy parser token bound exceeded");
    heads_.assign(count, -1);
    dependencies_.resize(count);
    unshiftable_.assign(count, 0);
    sentence_starts_.assign(count, 0);
    require(!ordered_moves.empty() && ordered_moves.size() <= 137, "Invalid spaCy parser action inventory");
    std::set<std::string> names;
    for (const auto & name : ordered_moves) {
        require(names.insert(name).second, "Duplicate spaCy parser action");
        Move move;
        if (name == "S" || name == "D")
            move.type = name[0];
        else {
            require(name.size() > 2 && name[1] == '-' && (name[0] == 'L' || name[0] == 'R' || name == "B-ROOT"),
                    "Unsupported spaCy parser action");
            move.type = name[0];
            move.label = name.substr(2);
            require(move.label != "subtok", "Token-learning spaCy parser actions are unsupported");
            const auto bar = move.label.find("||");
            require(bar == std::string::npos ||
                        (bar > 0 && bar + 2 < move.label.size() && move.label.find("||", bar + 2) == std::string::npos),
                    "Malformed decorated dependency label");
        }
        moves_.push_back(std::move(move));
    }
    require(names.count("S") && names.count("D") && names.count("B-ROOT"), "Missing spaCy parser control action");
    if (!preset.empty()) {
        require(preset.size() == count, "Invalid preset spaCy sentence-boundary count");
        for (auto value : preset) require(value >= -1 && value <= 1, "Invalid preset spaCy sentence boundary");
        sentence_starts_ = preset;
    } else if (count)
        sentence_starts_[0] = 1;
}

std::int32_t ParserState::s(std::size_t offset) const {
    return offset < stack_.size() ? stack_[stack_.size() - 1 - offset] : -1;
}
std::int32_t ParserState::b(std::size_t offset) const {
    if (offset < rebuffer_.size()) return rebuffer_[rebuffer_.size() - 1 - offset];
    const auto index = next_buffer_ + offset - rebuffer_.size();
    return index < heads_.size() ? static_cast<std::int32_t>(index) : -1;
}
std::size_t ParserState::buffer_length() const {
    return heads_.size() - next_buffer_ + rebuffer_.size();
}
bool ParserState::is_final() const {
    return stack_.empty() && buffer_length() == 0;
}
std::int32_t ParserState::latest_child(std::int32_t head, bool left) const {
    if (head < 0) return -1;
    const auto & map = left ? left_arcs_ : right_arcs_;
    const auto found = map.find(head);
    if (found == map.end()) return -1;
    const auto & arcs = found->second;
    for (auto it = arcs.rbegin(); it != arcs.rend(); ++it)
        if (it->child >= 0) return it->child;
    return -1;
}
std::array<std::int32_t, 8> ParserState::features() const {
    return {
        b(0), b(1), s(0), s(1), s(2), latest_child(b(0), true), latest_child(s(0), true), latest_child(s(0), false)
    };
}
bool ParserState::valid(const Move & move) const {
    if (is_final()) return false;
    const auto b0 = b(0), b1 = b(1);
    switch (move.type) {
    case 'S':
        return stack_.empty() || (buffer_length() >= 2 && sentence_starts_[b0] != 1 && !unshiftable_[b0]);
    case 'D':
        return !stack_.empty() && (buffer_length() == 0 || stack_.size() != 1 || sentence_starts_[b0] != -1);
    case 'L':
    case 'R':
        return !stack_.empty() && buffer_length() > 0 && sentence_starts_[b0] != 1;
    case 'B':
        return buffer_length() >= 2 && b1 == b0 + 1 && sentence_starts_[b1] == 0;
    default:
        return false;
    }
}
std::vector<std::uint8_t> ParserState::valid_actions() const {
    std::vector<std::uint8_t> result;
    for (const auto & move : moves_) result.push_back(valid(move));
    return result;
}
std::vector<std::int32_t> ParserState::stack() const {
    return { stack_.rbegin(), stack_.rend() };
}
std::vector<std::int32_t> ParserState::queue() const {
    std::vector<std::int32_t> result;
    for (std::size_t i = 0; i < buffer_length(); ++i) result.push_back(b(i));
    return result;
}
std::vector<DependencyArc> ParserState::arcs() const {
    std::vector<DependencyArc> result;
    for (const auto * map : { &left_arcs_, &right_arcs_ }) {
        for (const auto & items : *map)
            for (const auto & arc : items.second)
                if (arc.child >= 0) result.push_back(arc);
    }
    return result;
}
void ParserState::push() {
    const auto word = b(0);
    require(word >= 0, "SpaCy parser attempted to shift an empty buffer");
    if (rebuffer_.empty())
        ++next_buffer_;
    else
        rebuffer_.pop_back();
    stack_.push_back(word);
}
void ParserState::add_arc(std::int32_t head, std::int32_t child, const std::string & label) {
    if (heads_[child] >= 0) {
        const auto previous = heads_[child];
        auto & old = previous > child ? left_arcs_[previous] : right_arcs_[previous];
        if (!old.empty() && old.back().head == previous && old.back().child == child) old.pop_back();
        // The pinned source edits a copy for a non-last arc, leaving it in
        // the bucket. Preserve that observed behavior for state/feature and
        // final-annotation parity; changing it would define another parser.
    }
    auto & target = head > child ? left_arcs_[head] : right_arcs_[head];
    target.push_back({ head, child, label });
    heads_[child] = head;
    dependencies_[child] = label;
}
void ParserState::step(std::size_t action) {
    require(action < moves_.size() && valid(moves_[action]), "Invalid/illegal spaCy parser transition");
    require(steps_ < 8 * heads_.size() + 32, "SpaCy parser transition budget exceeded");
    ++steps_;
    const auto & move = moves_[action];
    switch (move.type) {
    case 'S':
        push();
        break;
    case 'D':
        if (heads_[s(0)] >= 0 || stack_.size() == 1)
            stack_.pop_back();
        else {
            unshiftable_[s(0)] = 1;
            rebuffer_.push_back(s(0));
            stack_.pop_back();
        }
        break;
    case 'L':
        add_arc(b(0), s(0), move.label);
        unshiftable_[b(0)] = 0;
        stack_.pop_back();
        break;
    case 'R':
        add_arc(s(0), b(0), move.label);
        push();
        break;
    case 'B':
        sentence_starts_[b(1)] = 1;
        break;
    default:
        throw std::invalid_argument("Unsupported spaCy transition");
    }
}

void deprojectivize(ParsedDocument & document, const std::vector<std::uint8_t> & is_space,
                    const CancelCheck & cancelled) {
    check_cancel(cancelled);
    require(is_space.size() == document.heads.size(), "Invalid spaCy whitespace flag count");
    for (auto value : is_space) require(value <= 1, "Invalid spaCy whitespace flag");
    // Source Token.children uses extents computed once before the loop; heads
    // and labels change during the loop, but those extents change only after it.
    const auto edges = subtree_edges(document, cancelled);
    const auto n = document.heads.size();
    for (std::size_t token = 0; token < n; ++token) {
        check_cancel(cancelled);
        const auto divider = document.dependencies[token].find("||");
        if (divider == std::string::npos) continue;
        const auto label = document.dependencies[token].substr(0, divider);
        const auto head_label = document.dependencies[token].substr(divider + 2);
        require(!label.empty() && !head_label.empty() && head_label.find("||") == std::string::npos,
                "Malformed spaCy decorated dependency");
        std::vector<std::int32_t> queue{ document.heads[token] };
        std::vector<std::uint8_t> seen(n, 0);
        auto replacement = document.heads[token];
        bool found = false;
        while (!queue.empty() && !found) {
            std::vector<std::int32_t> next;
            for (const auto parent : queue) {
                check_cancel(cancelled);
                require(!seen[parent], "Cycle during spaCy deprojection");
                seen[parent] = 1;
                for (auto child = edges.left[parent]; child <= edges.right[parent]; ++child) {
                    check_cancel(cancelled);
                    if (child == parent || child == static_cast<std::int32_t>(token) || is_space[child] ||
                        document.heads[child] != parent)
                        continue;
                    if (document.dependencies[child] == head_label) {
                        replacement = child;
                        found = true;
                        break;
                    }
                    next.push_back(child);
                }
                if (found) break;
            }
            queue = std::move(next);
        }
        document.heads[token] = replacement;
        document.dependencies[token] = label;
    }
    set_sentence_starts(document, cancelled);
    check_cancel(cancelled);
}

ParsedDocument ParserState::finish(const std::vector<std::uint8_t> & is_space, const CancelCheck & cancelled) const {
    check_cancel(cancelled);
    require(is_final(), "Cannot publish an unfinished spaCy parse");
    ParsedDocument result;
    result.heads.assign(heads_.size(), -1);
    result.dependencies.resize(heads_.size());
    // Like ArcEager.set_annotations, apply the source's left/right bucket
    // enumeration. This includes its preserved non-last replacement behavior.
    for (const auto * map : { &left_arcs_, &right_arcs_ }) {
        for (const auto & items : *map) {
            check_cancel(cancelled);
            for (const auto & arc : items.second) {
                check_cancel(cancelled);
                if (arc.child < 0) continue;
                result.heads[arc.child] = arc.head;
                result.dependencies[arc.child] = arc.label;
            }
        }
    }
    for (std::size_t i = 0; i < result.heads.size(); ++i) {
        check_cancel(cancelled);
        if (result.heads[i] < 0 || result.heads[i] == static_cast<std::int32_t>(i)) {
            result.heads[i] = static_cast<std::int32_t>(i);
            result.dependencies[i] = "ROOT";
        }
    }
    deprojectivize(result, is_space, cancelled);
    return result;
}

DecodeOutput decode_annotations(const DecodeResources & resources, const NumericOutput & scores,
                                const std::vector<DecodeToken> & tokens, const ParserScorer & scorer,
                                const ParserTrace & trace, const CancelCheck & cancelled) {
    check_cancel(cancelled);
    const auto n = tokens.size();
    require(n <= max_tokens, "SpaCy decoder token bound exceeded");
    require(resources.tags.size() == 55 && resources.morphology.size() == 671 && resources.moves.size() == 137 &&
                resources.class_to_tree.size() == 1311 && resources.trees.size() == 6092,
            "Changed spaCy decoder resource profile");
    DecodeOutput result;
    if (n == 0) {
        require(scores.tagger.rows == 0 && scores.morphologizer.rows == 0 && scores.lemmatizer.rows == 0 &&
                    scores.tagger.data.empty() && scores.morphologizer.data.empty() && scores.lemmatizer.data.empty(),
                "Nonempty spaCy scores for empty document");
        return result;
    }
    require(static_cast<bool>(scorer), "Missing spaCy parser scorer");
    check_matrix(scores.tagger, n, 55);
    check_matrix(scores.morphologizer, n, 671);
    check_matrix(scores.lemmatizer, n, 1311);
    std::vector<std::uint8_t> is_space;
    for (const auto & token : tokens) {
        (void)boundaries(token.text);
        is_space.push_back(token.is_space);
    }
    ParserState state(n, resources.moves);
    while (!state.is_final()) {
        check_cancel(cancelled);
        require(result.parser_steps < 8 * n + 32, "SpaCy parser transition budget exceeded");
        ParserStep step;
        step.step = result.parser_steps;
        step.features = state.features();
        step.valid_actions = state.valid_actions();
        step.scores = scorer(step.features, step.step);
        require(step.scores.size() == resources.moves.size(), "Invalid spaCy parser score count");
        bool selected = false;
        for (std::size_t i = 0; i < step.scores.size(); ++i) {
            require(std::isfinite(step.scores[i]), "Nonfinite spaCy parser score");
            if (step.valid_actions[i] && (!selected || step.scores[i] > step.scores[step.chosen_action])) {
                step.chosen_action = i;
                selected = true;
            }
        }
        require(selected, "SpaCy parser has no legal transition");
        if (trace) {
            step.stack_before = state.stack();
            step.queue_before = state.queue();
            step.arcs_before = state.arcs();
            trace(step);
        }
        check_cancel(cancelled);
        state.step(step.chosen_action);
        ++result.parser_steps;
    }
    const auto parsed = state.finish(is_space, cancelled);
    result.tokens.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        check_cancel(cancelled);
        TokenAnnotation annotation;
        annotation.tag_class = first_maximum(scores.tagger, i);
        annotation.morph_class = first_maximum(scores.morphologizer, i);
        annotation.lemma_class = first_maximum(scores.lemmatizer, i);
        annotation.tag = resources.tags[annotation.tag_class];
        annotation.pos = resources.morphology[annotation.morph_class].pos;
        annotation.morph = resources.morphology[annotation.morph_class].morph;
        annotation.lemma_decision = apply_lemma(resources.trees, resources.class_to_tree[annotation.lemma_class],
                                                tokens[i].text, tokens[i].existing_lemma);
        annotation.lemma = annotation.lemma_decision.lemma;
        annotation.head = parsed.heads[i];
        annotation.dep = parsed.dependencies[i];
        annotation.is_sent_start = parsed.sentence_starts[i] != 0;
        if (tokens[i].is_space) {
            if (!annotation.dep.empty()) annotation.dep = "dep";
            annotation.tag = "_SP";
            annotation.pos = "SPACE";
            annotation.morph.clear(); // spaCy normalizes MORPH="_" to empty.
        }
        if (annotation.is_sent_start) {
            if (!result.sentences.empty()) result.sentences.back().end_token = i;
            result.sentences.push_back({ i, n });
        }
        result.tokens.push_back(std::move(annotation));
    }
    check_cancel(cancelled);
    return result;
}

} // namespace spacy
} // namespace crispembed
