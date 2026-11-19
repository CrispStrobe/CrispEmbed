#include "spacy_transformer.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/env_gate.h"
#include "ggml.h"
#ifdef CRISPEMBED_SPACY_BLAS
#include "ggml-blas.h"
#endif

#include <sentencepiece_processor.h>

#include <algorithm>
#include <array>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace crispembed {
namespace spacy {
namespace {

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

void require(bool condition, const std::string & message) {
    if (!condition) throw std::invalid_argument(message);
}

// spacy-curated-transformers camembert_adapter: fairseq ids.
int camembert_id(int piece) {
    return piece == 0 ? 3 : piece + 4;
}

} // namespace

struct TransformerPipeline::Impl {
    TransformerConfig config;
    std::map<std::string, ggml_tensor *> weights;
    ggml_backend_t backend = nullptr;
    ggml_gallocr * allocator = nullptr;
    sentencepiece::SentencePieceProcessor spp;
    // RoBERTa byte-level BPE (curated-tokenizers ByteBPEProcessor semantics).
    bool byte_bpe = false;
    // BERT WordPiece (curated-tokenizers WordPieceProcessor + _bert_preprocess).
    bool wordpiece = false;
    std::unordered_map<std::string, int> wp_initial, wp_continuation;
    std::unique_ptr<PythonRegex> bert_punctuation;

    static std::vector<std::size_t> codepoint_offsets(const std::string & text) {
        std::vector<std::size_t> offsets;
        for (std::size_t i = 0; i < text.size(); ++i)
            if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) offsets.push_back(i);
        offsets.push_back(text.size());
        return offsets;
    }

    // WordPieceProcessor.encode: greedy longest prefix by code points
    // (including the empty prefix); an unmatched remainder appends one
    // unknown piece after the pieces found so far.
    void wordpiece_encode(const std::string & token, std::vector<int> & ids) const {
        const auto offsets = codepoint_offsets(token);
        const std::size_t count = offsets.size() - 1;
        std::size_t start = 0;
        bool initial = true;
        while (start < count) {
            const auto & table = initial ? wp_initial : wp_continuation;
            bool found = false;
            std::size_t length = count - start;
            for (;; --length) {
                const auto hit = table.find(token.substr(offsets[start], offsets[start + length] - offsets[start]));
                if (hit != table.end()) {
                    ids.push_back(hit->second);
                    found = true;
                    break;
                }
                if (length == 0) break;
            }
            if (!found) {
                ids.push_back(config.unk_id);
                return;
            }
            require(length > 0 || initial, "WordPiece empty continuation match would not terminate");
            start += length;
            initial = false;
        }
    }

    // _bert_preprocess: every punctuation character (ASCII punctuation or a
    // Unicode P* category) is its own part; other runs stay together.
    std::vector<int> encode_wordpiece(const std::string & text) const {
        std::vector<int> ids;
        std::size_t position = 0;
        for (const auto & span : bert_punctuation->find_all(text)) {
            if (span.first > position) wordpiece_encode(text.substr(position, span.first - position), ids);
            wordpiece_encode(text.substr(span.first, span.second - span.first), ids);
            position = span.second;
        }
        if (position < text.size()) wordpiece_encode(text.substr(position), ids);
        return ids;
    }
    std::unordered_map<std::string, int> piece_to_id;
    std::unordered_map<std::string, std::size_t> merge_rank; // key: first + '\0' + second
    std::array<std::string, 256> byte_encoder;
    std::unique_ptr<PythonRegex> split;

    // GPT-2 bytes_to_unicode().
    void build_byte_encoder() {
        std::vector<int> bs, cs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        for (std::size_t i = 0; i < bs.size(); ++i) {
            const auto cp = static_cast<std::uint32_t>(cs[i]);
            std::string utf8;
            if (cp < 0x80)
                utf8 += static_cast<char>(cp);
            else {
                utf8 += static_cast<char>(0xC0 | (cp >> 6));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            }
            byte_encoder[static_cast<std::size_t>(bs[i])] = utf8;
        }
    }

    // curated-tokenizers Merges::apply_merges: repeatedly merge every
    // left-to-right occurrence of the lowest-ranked adjacent pair.
    std::vector<std::string> apply_merges(std::vector<std::string> pieces) const {
        while (pieces.size() > 1) {
            std::size_t best_rank = merge_rank.size(), best = pieces.size();
            for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                const auto found = merge_rank.find(pieces[i] + '\0' + pieces[i + 1]);
                if (found != merge_rank.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best = i;
                }
            }
            if (best == pieces.size()) break;
            const std::string first = pieces[best], second = pieces[best + 1];
            std::vector<std::string> merged;
            for (std::size_t i = 0; i < pieces.size();) {
                if (i + 1 < pieces.size() && pieces[i] == first && pieces[i + 1] == second) {
                    merged.push_back(pieces[i] + pieces[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(pieces[i]);
                    ++i;
                }
            }
            pieces = std::move(merged);
        }
        return pieces;
    }

    std::vector<int> encode_byte_bpe(const std::string & text) const {
        std::vector<int> ids;
        for (const auto & span : split->find_all(text)) {
            std::vector<std::string> pieces;
            for (std::size_t i = span.first; i < span.second; ++i)
                pieces.push_back(byte_encoder[static_cast<unsigned char>(text[i])]);
            for (const auto & piece : apply_merges(std::move(pieces))) {
                const auto found = piece_to_id.find(piece);
                require(found != piece_to_id.end(), "Byte-BPE piece is not in the vocabulary");
                ids.push_back(found->second);
            }
        }
        return ids;
    }
    // Opt-in CRISPEMBED_SPACY_TRF_BLAS: encoder matrix products on ggml's BLAS
    // backend (Accelerate/OpenBLAS) through a scheduler; CPU for the rest.
    ggml_backend_t blas = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_tensor * weight(const std::string & name) const {
        const auto found = weights.find(name);
        require(found != weights.end() && found->second, "Missing spaCy transformer tensor: " + name);
        return found->second;
    }
    bool has(const std::string & name) const { return weights.find(name) != weights.end(); }

    void compute(ggml_cgraph * graph, const CancelCheck & cancelled) const {
        AbortState abort_state{ &cancelled };
        ggml_backend_cpu_set_abort_callback(backend, cancelled ? abort_requested : nullptr, &abort_state);
        const auto status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
        if (abort_state.fired || status == GGML_STATUS_ABORTED) throw NumericCancelled();
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("spaCy transformer graph computation failed");
    }

    // One span through the RoBERTa-style post-LN encoder; returns T x width.
    std::vector<float> encode_span(const std::vector<int> & ids, const CancelCheck & cancelled) const {
        const int64_t T = static_cast<int64_t>(ids.size());
        const int64_t W = config.width, H = config.heads, D = W / H;
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(4096, false);
        params.no_alloc = true;
        std::unique_ptr<ggml_context, ContextDeleter> owned(ggml_init(params));
        if (!owned) throw std::bad_alloc();
        ggml_context * ctx = owned.get();
        ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(tokens);
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(positions);
        auto layer_norm = [&](ggml_tensor * x, const std::string & prefix) {
            x = ggml_norm(ctx, x, config.layer_norm_eps);
            return ggml_add(ctx, ggml_mul(ctx, x, weight(prefix + ".weight")), weight(prefix + ".bias"));
        };
        auto linear = [&](ggml_tensor * x, const std::string & prefix) {
            return ggml_add(ctx, ggml_mul_mat(ctx, weight(prefix + ".weight"), x), weight(prefix + ".bias"));
        };
        // RobertaEmbeddings: (word + token_type) + position, then LayerNorm.
        ggml_tensor * x = ggml_get_rows(ctx, weight("trf.embed.word"), tokens);
        x = ggml_add(ctx, x, ggml_view_1d(ctx, weight("trf.embed.type"), W, 0));
        x = ggml_add(ctx, x, ggml_get_rows(ctx, weight("trf.embed.position"), positions));
        x = layer_norm(x, "trf.embed.norm");
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (int layer = 0; layer < config.layers; ++layer) {
            const auto p = "trf.layer." + std::to_string(layer) + ".";
            ggml_tensor * qkv = linear(x, p + "qkv"); // [3W, T]; q, k, v are contiguous row blocks
            auto split = [&](int64_t index) {
                ggml_tensor * view = ggml_view_2d(ctx, qkv, W, T, qkv->nb[1], index * W * ggml_element_size(qkv));
                return ggml_reshape_3d(ctx, ggml_cont(ctx, view), D, H, T);
            };
            ggml_tensor * q = ggml_permute(ctx, split(0), 0, 2, 1, 3); // [D, T, H]
            ggml_tensor * k = ggml_permute(ctx, split(1), 0, 2, 1, 3);
            ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, split(2), 1, 2, 0, 3)); // [T, D, H]
            ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                            // [T_k, T_q, H]
            scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
            ggml_tensor * attended = ggml_mul_mat(ctx, v, scores); // [D, T_q, H]
            attended = ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3));
            attended = ggml_reshape_2d(ctx, attended, W, T);
            ggml_tensor * attention = layer_norm(ggml_add(ctx, x, linear(attended, p + "attn_out")), p + "attn_norm");
            ggml_tensor * ffn = ggml_gelu_erf(ctx, linear(attention, p + "ffn_in"));
            x = layer_norm(ggml_add(ctx, attention, linear(ffn, p + "ffn_out")), p + "ffn_norm");
        }
        ggml_set_output(x);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, x);
        if (sched) {
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::bad_alloc();
        } else if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::bad_alloc();
        }
        std::vector<int32_t> position_ids(static_cast<size_t>(T));
        for (int64_t t = 0; t < T; ++t)
            position_ids[static_cast<size_t>(t)] = static_cast<int32_t>(t + config.position_offset);
        std::vector<int32_t> token_ids(ids.begin(), ids.end());
        ggml_backend_tensor_set(tokens, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions, position_ids.data(), 0, position_ids.size() * sizeof(int32_t));
        if (sched) {
            check_cancelled(cancelled);
            if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("spaCy transformer graph computation failed");
            check_cancelled(cancelled);
        } else {
            compute(graph, cancelled);
        }
        std::vector<float> output(static_cast<size_t>(T * W));
        ggml_backend_tensor_get(x, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

    // Opt-in CRISPEMBED_SPACY_TRF_FUSED: the same encoder as one batched graph
    // for B equal-length spans, Q/K/V from row views of the fused weight (no
    // activation copies) and ggml_flash_attn_ext (CrispEmbed's encoder pattern).
    std::vector<std::vector<float>> encode_batch(const std::vector<std::vector<int>> & batch,
                                                 const CancelCheck & cancelled) const {
        const int64_t B = static_cast<int64_t>(batch.size());
        const int64_t T = static_cast<int64_t>(batch.front().size());
        for (const auto & span : batch) require(static_cast<int64_t>(span.size()) == T, "Unequal span batch");
        const int64_t W = config.width, H = config.heads, D = W / H, TB = T * B;
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false);
        params.no_alloc = true;
        std::unique_ptr<ggml_context, ContextDeleter> owned(ggml_init(params));
        if (!owned) throw std::bad_alloc();
        ggml_context * ctx = owned.get();
        ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, TB);
        ggml_set_input(tokens);
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, TB);
        ggml_set_input(positions);
        auto layer_norm = [&](ggml_tensor * x, const std::string & prefix) {
            x = ggml_norm(ctx, x, config.layer_norm_eps);
            return ggml_add(ctx, ggml_mul(ctx, x, weight(prefix + ".weight")), weight(prefix + ".bias"));
        };
        auto linear = [&](ggml_tensor * x, const std::string & prefix) {
            return ggml_add(ctx, ggml_mul_mat(ctx, weight(prefix + ".weight"), x), weight(prefix + ".bias"));
        };
        ggml_tensor * x = ggml_get_rows(ctx, weight("trf.embed.word"), tokens);
        x = ggml_add(ctx, x, ggml_view_1d(ctx, weight("trf.embed.type"), W, 0));
        x = ggml_add(ctx, x, ggml_get_rows(ctx, weight("trf.embed.position"), positions));
        x = layer_norm(x, "trf.embed.norm");
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (int layer = 0; layer < config.layers; ++layer) {
            const auto p = "trf.layer." + std::to_string(layer) + ".";
            ggml_tensor * w = weight(p + "qkv.weight");
            ggml_tensor * b = weight(p + "qkv.bias");
            auto project = [&](int64_t index) {
                // Rows [index*W, (index+1)*W) of the fused [3W, W] weight are contiguous.
                ggml_tensor * wv = ggml_view_2d(ctx, w, W, W, w->nb[1], index * W * w->nb[1]);
                ggml_tensor * bv = ggml_view_1d(ctx, b, W, index * W * ggml_element_size(b));
                ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, wv, x), bv);             // [W, TB]
                return ggml_permute(ctx, ggml_reshape_4d(ctx, out, D, H, T, B), 0, 2, 1, 3); // [D, T, H, B]
            };
            ggml_tensor * q = project(0);
            ggml_tensor * k = project(1);
            ggml_tensor * v = project(2);
            ggml_tensor * attended = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
            attended = ggml_reshape_2d(ctx, attended, W, TB); // [D, H, T, B] -> [W, TB]
            ggml_tensor * attention = layer_norm(ggml_add(ctx, x, linear(attended, p + "attn_out")), p + "attn_norm");
            ggml_tensor * ffn = ggml_gelu_erf(ctx, linear(attention, p + "ffn_in"));
            x = layer_norm(ggml_add(ctx, attention, linear(ffn, p + "ffn_out")), p + "ffn_norm");
        }
        ggml_set_output(x);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, x);
        if (sched) {
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::bad_alloc();
        } else if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::bad_alloc();
        }
        std::vector<int32_t> token_ids, position_ids;
        token_ids.reserve(static_cast<size_t>(TB));
        position_ids.reserve(static_cast<size_t>(TB));
        for (const auto & span : batch)
            for (int64_t t = 0; t < T; ++t) {
                token_ids.push_back(span[static_cast<size_t>(t)]);
                position_ids.push_back(static_cast<int32_t>(t + config.position_offset));
            }
        ggml_backend_tensor_set(tokens, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions, position_ids.data(), 0, position_ids.size() * sizeof(int32_t));
        if (sched) {
            check_cancelled(cancelled);
            if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("spaCy transformer graph computation failed");
            check_cancelled(cancelled);
        } else {
            compute(graph, cancelled);
        }
        std::vector<float> all(static_cast<size_t>(TB * W));
        ggml_backend_tensor_get(x, all.data(), 0, all.size() * sizeof(float));
        std::vector<std::vector<float>> outputs;
        for (int64_t i = 0; i < B; ++i) outputs.emplace_back(all.begin() + i * T * W, all.begin() + (i + 1) * T * W);
        return outputs;
    }
    bool fused = false;
};

TransformerPipeline::TransformerPipeline(TransformerConfig config, std::map<std::string, ggml_tensor *> weights,
                                         ggml_backend_t backend)
    : impl_(std::make_unique<Impl>()) {
    auto & m = *impl_;
    m.config = std::move(config);
    m.weights = std::move(weights);
    m.backend = backend;
    require(backend && ggml_backend_is_cpu(backend), "spaCy transformer requires a CPU backend");
    require(m.config.width > 0 && m.config.heads > 0 && m.config.width % m.config.heads == 0 && m.config.layers > 0 &&
                m.config.layers <= 64,
            "Invalid spaCy transformer geometry");
    if (m.config.position_offset < 0) m.config.position_offset = m.config.padding_idx + 1;
    require(m.config.window > 0 && m.config.window / 2 <= m.config.stride && m.config.stride <= m.config.window &&
                m.config.window + 1 + m.config.position_offset < m.config.max_positions &&
                m.config.window <= m.config.model_max_length,
            "Invalid spaCy transformer span settings");
    for (const auto & item : m.weights)
        require(item.second && item.second->type == GGML_TYPE_F32,
                "spaCy transformer tensors must be F32: " + item.first);
    require(m.weight("trf.embed.word")->ne[0] == m.config.width, "Transformer width mismatch");
    if (m.config.piece_encoder == "roberta_byte_bpe") {
        m.byte_bpe = true;
        const auto & pieces = m.config.bpe_pieces;
        require(!pieces.empty() && pieces.size() <= 1000000, "Invalid byte-BPE vocabulary");
        for (std::size_t i = 0; i < pieces.size(); ++i)
            require(m.piece_to_id.emplace(pieces[i], static_cast<int>(i)).second, "Duplicate byte-BPE piece");
        for (std::size_t i = 0; i < m.config.bpe_merges.size(); ++i) {
            const auto & merge = m.config.bpe_merges[i];
            m.merge_rank[merge.first + '\0' + merge.second] = i; // later duplicates win, as in Merges()
        }
        m.build_byte_encoder();
        m.split = std::make_unique<PythonRegex>(m.config.bpe_split_pattern);
        for (const int id : { m.config.bos_id, m.config.eos_id, m.config.unk_id })
            require(id >= 0 && static_cast<std::size_t>(id) < pieces.size(), "Invalid byte-BPE special id");
    } else if (m.config.piece_encoder == "bert_wordpiece") {
        m.wordpiece = true;
        const auto & pieces = m.config.wordpiece_pieces;
        require(!pieces.empty() && pieces.size() <= 1000000, "Invalid WordPiece vocabulary");
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            const auto & piece = pieces[i];
            if (piece.rfind("##", 0) == 0) {
                require(piece.size() > 2, "Empty continuation WordPiece piece");
                m.wp_continuation.emplace(piece.substr(2), static_cast<int>(i)); // first wins
            } else {
                m.wp_initial.emplace(piece, static_cast<int>(i));
            }
        }
        // _is_bert_punctuation: HF's ASCII punctuation ranges or category P*.
        m.bert_punctuation = std::make_unique<PythonRegex>("[!-/:-@\\[-`{-~]|\\p{P}");
        for (const int id : { m.config.bos_id, m.config.eos_id, m.config.unk_id })
            require(id >= 0 && static_cast<std::size_t>(id) < pieces.size(), "Invalid WordPiece special id");
    } else {
        require(m.config.piece_encoder == "camembert_sentencepiece", "Unsupported spaCy piece encoder");
        const auto status = m.spp.LoadFromSerializedProto(m.config.sentencepiece_model);
        require(status.ok(), "Invalid SentencePiece model: " + status.ToString());
    }
    m.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!m.allocator) throw std::bad_alloc();
    // Default since the hosted A/B (all 48 configurations exact on Linux x86-64,
    // Linux arm64 and macOS arm64); CRISPEMBED_SPACY_TRF_FUSED=0 keeps the
    // reference per-span graph.
    m.fused = !core_env::explicitly_off("CRISPEMBED_SPACY_TRF_FUSED");
#ifdef CRISPEMBED_SPACY_BLAS
    if (core_env::on("CRISPEMBED_SPACY_TRF_BLAS")) {
        m.blas = ggml_backend_blas_init();
        require(m.blas != nullptr, "Cannot initialise the ggml BLAS backend");
        ggml_backend_t backends[2] = { m.blas, backend }; // CPU last, as the scheduler requires
        m.sched = ggml_backend_sched_new(backends, nullptr, 2, 8192, false, false);
        require(m.sched != nullptr, "Cannot create the spaCy transformer scheduler");
    }
#endif
}

TransformerPipeline::~TransformerPipeline() {
    if (!impl_) return;
    if (impl_->sched) ggml_backend_sched_free(impl_->sched);
    if (impl_->blas) ggml_backend_free(impl_->blas);
    if (impl_->allocator) ggml_gallocr_free(impl_->allocator);
}

std::vector<std::vector<int>> TransformerPipeline::pieces(const std::vector<LexicalToken> & tokens) const {
    std::vector<std::vector<int>> result;
    result.reserve(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto & token = tokens[index];
        std::vector<int> ids;
        if (!token.is_space) {
            if (impl_->wordpiece) {
                ids = impl_->encode_wordpiece(token.text);
            } else if (impl_->byte_bpe) {
                // GPT-2/RoBERTa keeps the preceding token's whitespace_ (" ").
                const bool space = index > 0 && tokens[index - 1].space_after;
                ids = impl_->encode_byte_bpe(space ? " " + token.text : token.text);
            } else {
                const auto status = impl_->spp.Encode(token.text, &ids);
                require(status.ok(), "SentencePiece encoding failed: " + status.ToString());
                for (auto & id : ids) id = camembert_id(id);
            }
        }
        result.push_back(std::move(ids));
    }
    return result;
}

NumericOutput TransformerPipeline::run(const std::vector<LexicalToken> & tokens, const TraceSink & trace,
                                       const CancelCheck & cancelled) const {
    const auto & m = *impl_;
    check_cancelled(cancelled);
    NumericOutput output;
    const auto n = tokens.size();
    if (n == 0) return output;
    const auto per_token = pieces(tokens);
    std::vector<int> doc = { m.byte_bpe || m.wordpiece ? m.config.bos_id : camembert_id(m.spp.bos_id()) };
    std::vector<size_t> lengths;
    for (const auto & ids : per_token) {
        doc.insert(doc.end(), ids.begin(), ids.end());
        lengths.push_back(ids.size());
    }
    doc.push_back(m.byte_bpe || m.wordpiece ? m.config.eos_id : camembert_id(m.spp.eos_id()));
    const size_t L = doc.size(), W = static_cast<size_t>(m.config.width);
    const size_t window = static_cast<size_t>(m.config.window), stride = static_cast<size_t>(m.config.stride);

    // with_strided_spans: spans data[i*stride : i*stride+window] while data remains.
    std::vector<std::vector<float>> spans;
    std::vector<size_t> span_lengths;
    for (size_t offset = 0; offset < L; offset += stride) span_lengths.push_back(std::min(window, L - offset));
    if (m.fused) {
        // Batch consecutive spans of equal length (all full windows, then the tail).
        for (size_t first = 0; first < span_lengths.size();) {
            size_t last = first;
            while (last + 1 < span_lengths.size() && span_lengths[last + 1] == span_lengths[first] &&
                   last + 1 - first < 32)
                ++last;
            std::vector<std::vector<int>> batch;
            for (size_t i = first; i <= last; ++i)
                batch.emplace_back(doc.begin() + static_cast<std::ptrdiff_t>(i * stride),
                                   doc.begin() + static_cast<std::ptrdiff_t>(i * stride + span_lengths[i]));
            for (auto & out : m.encode_batch(batch, cancelled)) spans.push_back(std::move(out));
            first = last + 1;
        }
    } else {
        for (size_t i = 0; i < span_lengths.size(); ++i) {
            check_cancelled(cancelled);
            spans.push_back(
                m.encode_span(std::vector<int>(doc.begin() + static_cast<std::ptrdiff_t>(i * stride),
                                               doc.begin() + static_cast<std::ptrdiff_t>(i * stride + span_lengths[i])),
                              cancelled));
        }
    }
    // _apply_to_overlaps(_average_arrays): pairwise in-place averages.
    {
        size_t remaining = L, index = 0;
        float * previous = nullptr;
        size_t previous_rows = 0;
        while (remaining != 0) {
            const size_t overlap = std::min(window - stride, remaining);
            if (previous) {
                float * prev = previous + (previous_rows - overlap) * W; // prev_overlap[-overlap:]
                float * cur = spans[index].data();                       // input[0][:overlap]
                for (size_t i = 0; i < overlap * W; ++i) {
                    float value = prev[i] + cur[i];
                    value /= 2.0f;
                    prev[i] = value;
                    cur[i] = value;
                }
            }
            previous = spans[index].data() + (span_lengths[index] - overlap) * W; // input[0][-overlap:]
            previous_rows = overlap;
            remaining -= std::min(stride, remaining);
            ++index;
        }
    }
    // _strided_arrays_to_ragged: first min(stride, remaining) rows of each span.
    std::vector<float> stitched;
    stitched.reserve(L * W);
    {
        size_t remaining = L;
        for (size_t index = 0; remaining != 0; ++index) {
            const size_t take = std::min(stride, remaining);
            stitched.insert(stitched.end(), spans[index].begin(), spans[index].begin() + take * W);
            remaining -= take;
        }
    }
    // remove_bos_eos + LastTransformerLayerListener(reduce_mean).
    Matrix vectors;
    vectors.rows = n;
    vectors.cols = W;
    vectors.data.assign(n * W, 0.0f);
    size_t row = 1;
    for (size_t t = 0; t < n; ++t) {
        float * target = vectors.data.data() + t * W;
        for (size_t p = 0; p < lengths[t]; ++p, ++row)
            for (size_t c = 0; c < W; ++c) target[c] += stitched[row * W + c];
        if (lengths[t])
            for (size_t c = 0; c < W; ++c) target[c] /= static_cast<float>(lengths[t]);
    }
    for (const float value : vectors.data)
        require(std::isfinite(value), "spaCy transformer produced a nonfinite activation");
    if (trace) trace("tok2vec.output", vectors.view());

    // Heads on the pooled vectors: tagger/morphologizer scores, parser lower cache.
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    params.no_alloc = true;
    std::unique_ptr<ggml_context, ContextDeleter> owned(ggml_init(params));
    if (!owned) throw std::bad_alloc();
    ggml_context * ctx = owned.get();
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int64_t>(W), static_cast<int64_t>(n));
    ggml_set_input(input);
    auto head = [&](const char * name) -> ggml_tensor * {
        if (!m.has(std::string(name) + ".W")) return nullptr;
        ggml_tensor * scores = ggml_add(ctx, ggml_mul_mat(ctx, m.weight(std::string(name) + ".W"), input),
                                        m.weight(std::string(name) + ".b"));
        ggml_set_output(scores);
        return scores;
    };
    ggml_tensor * tagger = head("tagger");
    ggml_tensor * morphologizer = head("morphologizer");
    ggml_tensor * lemmatizer = head("lemmatizer");
    ggml_tensor * reduced = head("parser.reduce");
    require(reduced != nullptr, "Missing spaCy parser reduce layer");
    ggml_tensor * lower_w = m.weight("parser.lower.W");
    const int64_t columns = lower_w->ne[1] * lower_w->ne[2] * lower_w->ne[3];
    ggml_tensor * lower = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, lower_w, lower_w->ne[0], columns), reduced);
    ggml_set_output(lower);
    // Transformer-listener NER: reduce + lower layer on the same vectors.
    ggml_tensor * ner_lower = nullptr;
    if (m.has("ner.reduce.W")) {
        ggml_tensor * ner_reduced = head("ner.reduce");
        ggml_tensor * ner_w = m.weight("ner.lower.W");
        ner_lower = ggml_mul_mat(
            ctx, ggml_reshape_2d(ctx, ner_w, ner_w->ne[0], ner_w->ne[1] * ner_w->ne[2] * ner_w->ne[3]), ner_reduced);
        ggml_set_output(ner_lower);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (ggml_tensor * out : { tagger, morphologizer, lemmatizer, lower, ner_lower })
        if (out) ggml_build_forward_expand(graph, out);
    if (!ggml_gallocr_alloc_graph(m.allocator, graph)) throw std::bad_alloc();
    ggml_backend_tensor_set(input, vectors.data.data(), 0, vectors.data.size() * sizeof(float));
    m.compute(graph, cancelled);
    auto read = [&](ggml_tensor * tensor) {
        Matrix result;
        if (!tensor) return result;
        result.rows = n;
        result.cols = static_cast<size_t>(tensor->ne[0]);
        result.data.resize(result.rows * result.cols);
        ggml_backend_tensor_get(tensor, result.data.data(), 0, result.data.size() * sizeof(float));
        for (const float value : result.data) require(std::isfinite(value), "Nonfinite spaCy head score");
        return result;
    };
    output.tagger = read(tagger);
    output.ner_lower = read(ner_lower);
    output.lemmatizer = read(lemmatizer);
    output.morphologizer = read(morphologizer);
    const auto lower_values = read(lower);
    output.parser_cache.rows = n + 1;
    output.parser_cache.cols = static_cast<size_t>(columns);
    output.parser_cache.data.resize(output.parser_cache.rows * output.parser_cache.cols);
    ggml_backend_tensor_get(m.weight("parser.lower.pad"), output.parser_cache.data.data(), 0,
                            static_cast<size_t>(columns) * sizeof(float));
    std::copy(lower_values.data.begin(), lower_values.data.end(),
              output.parser_cache.data.begin() + static_cast<std::ptrdiff_t>(columns));
    if (trace) {
        if (tagger) trace("tagger.scores", output.tagger.view());
        if (morphologizer) trace("morphologizer.scores", output.morphologizer.view());
    }
    output.tok2vec = std::move(vectors);
    check_cancelled(cancelled);
    return output;
}

} // namespace spacy
} // namespace crispembed
