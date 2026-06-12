// model_mgr.cpp — Auto-download model manager for CrispEmbed.

#include "model_mgr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cctype>
#include <string>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define mkdir(p, m) _mkdir(p)
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace crispembed_mgr {

namespace {

bool download_supported() {
#if defined(__EMSCRIPTEN__)
    return false;
#elif defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    return false;
#else
    return true;
#endif
}

}  // namespace

struct ModelEntry {
    const char * name;
    const char * filename;
    const char * url;
    const char * desc;
    const char * approx_size;
    const char * license;          // SPDX-style tag from the upstream model
                                    // card (NOT from the cstr/* re-host).
                                    // Verified by tests/check_registry_licenses.py.
    const char * model_card_url;   // upstream HuggingFace model card
};

// Prompt prefixes for models that need them for optimal retrieval.
// query_prefix() returns the prefix to prepend to queries.
// passage_prefix() returns the prefix to prepend to passages/documents.
static const char * query_prefix(const char * model) {
    if (!model) return nullptr;
    // BGE models
    if (strstr(model, "bge-") && !strstr(model, "reranker") && !strstr(model, "m3"))
        return "Represent this sentence for searching relevant passages: ";
    // E5 models
    if (strstr(model, "-e5-"))
        return "query: ";
    // Nomic
    if (strstr(model, "nomic-embed"))
        return "search_query: ";
    // Jina v5
    if (strstr(model, "jina-v5"))
        return "Query: ";
    return nullptr;
}

static const char * passage_prefix(const char * model) {
    if (!model) return nullptr;
    // E5 models
    if (strstr(model, "-e5-"))
        return "passage: ";
    // Nomic
    if (strstr(model, "nomic-embed"))
        return "search_document: ";
    // Jina v5
    if (strstr(model, "jina-v5"))
        return "Passage: ";
    return nullptr;
}

static const ModelEntry k_registry[] = {
    {"all-MiniLM-L6-v2",
     "all-MiniLM-L6-v2.gguf",
     "https://huggingface.co/cstr/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.gguf",
     "BERT 384d English (22M)", "87 MB", "apache-2.0",
     "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"},

    {"gte-small",
     "gte-small.gguf",
     "https://huggingface.co/cstr/gte-small-GGUF/resolve/main/gte-small.gguf",
     "BERT 384d English (33M)", "128 MB", "mit",
     "https://huggingface.co/thenlper/gte-small"},

    {"arctic-embed-xs",
     "arctic-embed-xs.gguf",
     "https://huggingface.co/cstr/arctic-embed-xs-GGUF/resolve/main/arctic-embed-xs.gguf",
     "BERT 384d CLS English (22M)", "87 MB", "apache-2.0",
     "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs"},

    {"multilingual-e5-small",
     "multilingual-e5-small.gguf",
     "https://huggingface.co/cstr/multilingual-e5-small-GGUF/resolve/main/multilingual-e5-small.gguf",
     "XLM-R 384d multilingual (118M)", "454 MB", "mit",
     "https://huggingface.co/intfloat/multilingual-e5-small"},

    {"pixie-rune-v1",
     "pixie-rune-v1.gguf",
     "https://huggingface.co/cstr/pixie-rune-v1-GGUF/resolve/main/pixie-rune-v1.gguf",
     "XLM-R 1024d 74-lang CLS (560M)", "2.2 GB", "apache-2.0",
     "https://huggingface.co/telepix/PIXIE-Rune-v1.0"},

    {"arctic-embed-l-v2",
     "arctic-embed-l-v2.gguf",
     "https://huggingface.co/cstr/arctic-embed-l-v2-GGUF/resolve/main/arctic-embed-l-v2.gguf",
     "XLM-R 1024d CLS English (560M)", "2.2 GB", "apache-2.0",
     "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0"},

    {"octen-0.6b",
     "octen-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/octen-0.6b-GGUF/resolve/main/octen-0.6b-q8_0.gguf",
     "Qwen3 1024d multilingual (600M)", "609 MB", "apache-2.0",
     "https://huggingface.co/Octen/Octen-Embedding-0.6B"},

    {"octen-4b",
     "octen-4b-q4_k.gguf",
     "https://huggingface.co/cstr/octen-4b-GGUF/resolve/main/octen-4b-q4_k.gguf",
     "Qwen3 2560d multilingual (4B)", "2.3 GB", "apache-2.0",
     "https://huggingface.co/Octen/Octen-Embedding-4B"},

    {"octen-8b",
     "octen-8b-q4_k.gguf",
     "https://huggingface.co/cstr/octen-8b-GGUF/resolve/main/octen-8b-q4_k.gguf",
     "Qwen3 4096d multilingual (8B)", "4.4 GB", "apache-2.0",
     "https://huggingface.co/Octen/Octen-Embedding-8B"},

    {"f2llm-v2-0.6b",
     "f2llm-v2-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/f2llm-v2-0.6b-GGUF/resolve/main/f2llm-v2-0.6b-q8_0.gguf",
     "Qwen3 1024d multilingual (600M)", "609 MB", "apache-2.0",
     "https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B"},

    {"jina-v5-nano",
     "jina-v5-nano-q8_0.gguf",
     "https://huggingface.co/cstr/jina-v5-nano-GGUF/resolve/main/jina-v5-nano-q8_0.gguf",
     "Qwen3 1024d compact (210M)", "222 MB", "cc-by-nc-4.0",
     "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano"},

    {"jina-v5-small",
     "jina-v5-small-q8_0.gguf",
     "https://huggingface.co/cstr/jina-v5-small-GGUF/resolve/main/jina-v5-small-q8_0.gguf",
     "Qwen3 1024d multilingual (600M)", "609 MB", "cc-by-nc-4.0",
     "https://huggingface.co/jinaai/jina-embeddings-v5-text-small"},

    {"harrier-0.6b",
     "harrier-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/harrier-0.6b-GGUF/resolve/main/harrier-0.6b-q8_0.gguf",
     "Qwen3 1024d SOTA (600M)", "609 MB", "mit",
     "https://huggingface.co/microsoft/harrier-oss-v1-0.6b"},

    {"harrier-270m",
     "harrier-270m-q8_0.gguf",
     "https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q8_0.gguf",
     "Gemma3 640d compact (270M)", "755 MB", "mit",
     "https://huggingface.co/microsoft/harrier-oss-v1-270m"},

    {"qwen3-embed-0.6b",
     "qwen3-embed-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-embed-0.6b-GGUF/resolve/main/qwen3-embed-0.6b-q8_0.gguf",
     "Qwen3 1024d official (600M)", "1.0 GB", "apache-2.0",
     "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"},

    {"qwen3-embed-4b",
     "qwen3-embed-4b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-embed-4b-GGUF/resolve/main/qwen3-embed-4b-q4_k.gguf",
     "Qwen3 2560d official (4B)", "2.3 GB", "apache-2.0",
     "https://huggingface.co/Qwen/Qwen3-Embedding-4B"},

    {"qwen3-embed-8b",
     "qwen3-embed-8b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-embed-8b-GGUF/resolve/main/qwen3-embed-8b-q4_k.gguf",
     "Qwen3 4096d official (8B)", "4.4 GB", "apache-2.0",
     "https://huggingface.co/Qwen/Qwen3-Embedding-8B"},

    // BidirLM-Omni — bidirectional Qwen3 (text) + Whisper-shape audio tower (cross-modal).
    // Two repos: -textonly is the smaller text-only variant; without suffix includes audio.
    {"bidirlm-omni-2.5b",
     "bidirlm-omni-2.5b-q8_0.gguf",
     "https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q8_0.gguf",
     "Qwen3-Bidirectional 2048d 90+langs text+audio (2.5B)", "3.1 GB", "apache-2.0",
     "https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding"},
    {"bidirlm-omni-2.5b-textonly",
     "bidirlm-omni-2.5b-textonly-q8_0.gguf",
     "https://huggingface.co/cstr/bidirlm-omni-2.5b-textonly-GGUF/resolve/main/bidirlm-omni-2.5b-textonly-q8_0.gguf",
     "Qwen3-Bidirectional 2048d text-only (2.5B)", "2.6 GB", "apache-2.0",
     "https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding"},

    // --- RAG-critical models (Phase 3) ---

    {"bge-small-en-v1.5",
     "bge-small-en-v1.5.gguf",
     "https://huggingface.co/cstr/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5.gguf",
     "BERT 384d English (33M)", "128 MB", "mit",
     "https://huggingface.co/BAAI/bge-small-en-v1.5"},

    {"bge-base-en-v1.5",
     "bge-base-en-v1.5.gguf",
     "https://huggingface.co/cstr/bge-base-en-v1.5-GGUF/resolve/main/bge-base-en-v1.5.gguf",
     "BERT 768d English (109M)", "418 MB", "mit",
     "https://huggingface.co/BAAI/bge-base-en-v1.5"},

    {"bge-large-en-v1.5",
     "bge-large-en-v1.5.gguf",
     "https://huggingface.co/cstr/bge-large-en-v1.5-GGUF/resolve/main/bge-large-en-v1.5.gguf",
     "BERT 1024d English (335M)", "1.3 GB", "mit",
     "https://huggingface.co/BAAI/bge-large-en-v1.5"},

    {"nomic-embed-text-v1.5",
     "nomic-embed-text-v1.5.gguf",
     "https://huggingface.co/cstr/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.gguf",
     "BERT 768d 8K context Matryoshka (137M)", "523 MB", "apache-2.0",
     "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5"},

    {"nomic-embed-text-v2-moe",
     "nomic-v2-moe-q8_0.gguf",
     "https://huggingface.co/cstr/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-v2-moe-q8_0.gguf",
     "NomicBERT MoE 768d 8-expert top-2 (475M)", "487 MB", "apache-2.0",
     "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe"},

    {"all-MiniLM-L12-v2",
     "all-MiniLM-L12-v2.gguf",
     "https://huggingface.co/cstr/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.gguf",
     "BERT 384d English (33M)", "128 MB", "apache-2.0",
     "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2"},

    {"paraphrase-multilingual-MiniLM-L12-v2",
     "paraphrase-multilingual-MiniLM-L12-v2.gguf",
     "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-GGUF/resolve/main/paraphrase-multilingual-MiniLM-L12-v2.gguf",
     "BERT 384d 50+ langs (118M, SentencePiece, mean-pool)", "454 MB", "apache-2.0",
     "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},

    {"all-mpnet-base-v2",
     "all-mpnet-base-v2.gguf",
     "https://huggingface.co/cstr/all-mpnet-base-v2-GGUF/resolve/main/all-mpnet-base-v2.gguf",
     "BERT 768d English (109M)", "418 MB", "apache-2.0",
     "https://huggingface.co/sentence-transformers/all-mpnet-base-v2"},

    {"mxbai-embed-large-v1",
     "mxbai-embed-large-v1.gguf",
     "https://huggingface.co/cstr/mxbai-embed-large-v1-GGUF/resolve/main/mxbai-embed-large-v1.gguf",
     "BERT 1024d English (335M)", "1.3 GB", "apache-2.0",
     "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1"},

    {"snowflake-arctic-embed-m",
     "snowflake-arctic-embed-m.gguf",
     "https://huggingface.co/cstr/snowflake-arctic-embed-m-GGUF/resolve/main/snowflake-arctic-embed-m.gguf",
     "BERT 768d CLS English (109M)", "418 MB", "apache-2.0",
     "https://huggingface.co/Snowflake/snowflake-arctic-embed-m"},

    {"snowflake-arctic-embed-l",
     "snowflake-arctic-embed-l.gguf",
     "https://huggingface.co/cstr/snowflake-arctic-embed-l-GGUF/resolve/main/snowflake-arctic-embed-l.gguf",
     "XLM-R 1024d CLS English (335M)", "1.3 GB", "apache-2.0",
     "https://huggingface.co/Snowflake/snowflake-arctic-embed-l"},

    {"bge-m3",
     "bge-m3-q4_k.gguf",
     "https://huggingface.co/cstr/bge-m3-GGUF/resolve/main/bge-m3-q4_k.gguf",
     "XLM-R 1024d dense+sparse+ColBERT multilingual (568M)", "438 MB", "mit",
     "https://huggingface.co/BAAI/bge-m3"},

    // --- Reranker models (Phase 4) ---

    {"bge-reranker-v2-m3",
     "bge-reranker-v2-m3.gguf",
     "https://huggingface.co/cstr/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3.gguf",
     "XLM-R reranker multilingual (568M)", "2.2 GB", "apache-2.0",
     "https://huggingface.co/BAAI/bge-reranker-v2-m3"},

    {"bge-reranker-base",
     "bge-reranker-base.gguf",
     "https://huggingface.co/cstr/bge-reranker-base-GGUF/resolve/main/bge-reranker-base.gguf",
     "BERT reranker EN+ZH (278M)", "1.1 GB", "mit",
     "https://huggingface.co/BAAI/bge-reranker-base"},

    {"ms-marco-MiniLM-L-6-v2",
     "ms-marco-MiniLM-L-6-v2.gguf",
     "https://huggingface.co/cstr/ms-marco-MiniLM-L-6-v2-GGUF/resolve/main/ms-marco-MiniLM-L-6-v2.gguf",
     "BERT reranker English fast (22M)", "87 MB", "apache-2.0",
     "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2"},

    {"ms-marco-MiniLM-L-12-v2",
     "ms-marco-MiniLM-L-12-v2.gguf",
     "https://huggingface.co/cstr/ms-marco-MiniLM-L-12-v2-GGUF/resolve/main/ms-marco-MiniLM-L-12-v2.gguf",
     "BERT reranker English (33M)", "128 MB", "apache-2.0",
     "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2"},

    {"jina-reranker-v2-base-multilingual",
     "jina-reranker-v2-base-multilingual.gguf",
     "https://huggingface.co/cstr/jina-reranker-v2-base-multilingual-GGUF/resolve/main/jina-reranker-v2-base-multilingual.gguf",
     "XLM-R reranker multilingual (278M)", "1.1 GB", "cc-by-nc-4.0",
     "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual"},

    {"mxbai-rerank-xsmall-v1",
     "mxbai-rerank-xsmall-v1.gguf",
     "https://huggingface.co/cstr/mxbai-rerank-xsmall-v1-GGUF/resolve/main/mxbai-rerank-xsmall-v1.gguf",
     "BERT reranker English fast (33M)", "128 MB", "apache-2.0",
     "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1"},

    {"mxbai-rerank-base-v1",
     "mxbai-rerank-base-v1.gguf",
     "https://huggingface.co/cstr/mxbai-rerank-base-v1-GGUF/resolve/main/mxbai-rerank-base-v1.gguf",
     "BERT reranker English (86M)", "330 MB", "apache-2.0",
     "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1"},

    // --- MTEB top multilingual models ---

    {"multilingual-e5-base",
     "multilingual-e5-base.gguf",
     "https://huggingface.co/cstr/multilingual-e5-base-GGUF/resolve/main/multilingual-e5-base.gguf",
     "XLM-R 768d 100+ languages (278M)", "1.1 GB", "mit",
     "https://huggingface.co/intfloat/multilingual-e5-base"},

    {"multilingual-e5-large",
     "multilingual-e5-large.gguf",
     "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large.gguf",
     "XLM-R 1024d 100+ languages (560M)", "2.2 GB", "mit",
     "https://huggingface.co/intfloat/multilingual-e5-large"},

    {"granite-embedding-278m",
     "granite-embedding-278m-multilingual.gguf",
     "https://huggingface.co/cstr/granite-embedding-278m-multilingual-GGUF/resolve/main/granite-embedding-278m-multilingual.gguf",
     "XLM-R 768d IBM multilingual (278M)", "1.1 GB", "apache-2.0",
     "https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual"},

    {"granite-embedding-107m",
     "granite-embedding-107m-multilingual.gguf",
     "https://huggingface.co/cstr/granite-embedding-107m-multilingual-GGUF/resolve/main/granite-embedding-107m-multilingual.gguf",
     "XLM-R 384d IBM multilingual (107M)", "418 MB", "apache-2.0",
     "https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual"},

    // --- Sparse models ---

    {"splade-pp-en-v1",
     "splade-pp-en-v1.gguf",
     "https://huggingface.co/cstr/splade-pp-en-v1-GGUF/resolve/main/splade-pp-en-v1.gguf",
     "BERT sparse (SPLADE) English (109M)", "418 MB", "apache-2.0",
     "https://huggingface.co/prithivida/Splade_PP_en_v1"},

    // --- GTE v1.5 (new BERT) ---

    {"gte-base-en-v1.5",
     "gte-base-en-v1.5.gguf",
     "https://huggingface.co/cstr/gte-base-en-v1.5-GGUF/resolve/main/gte-base-en-v1.5.gguf",
     "GTE 768d English pre-LN+RoPE+GeGLU (109M)", "522 MB", "apache-2.0",
     "https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5"},

    {"gte-large-en-v1.5",
     "gte-large-en-v1.5.gguf",
     "https://huggingface.co/cstr/gte-large-en-v1.5-GGUF/resolve/main/gte-large-en-v1.5.gguf",
     "GTE 1024d English pre-LN+RoPE+GeGLU (335M)", "1.7 GB", "apache-2.0",
     "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5"},

    {"embeddinggemma-300m",
     "embeddinggemma-300m.gguf",
     "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m.gguf",
     "Gemma3 768d 24-layer last-token (300M)", "741 MB", "gemma",
     "https://huggingface.co/google/embeddinggemma-300m"},

    {"yunet",
     "yunet.gguf",
     "https://huggingface.co/cstr/yunet-GGUF/resolve/main/yunet.gguf",
     "YuNet face detection (ShuffleNetV2 640x640, 75K)", "0.2 MB", "apache-2.0",
     "https://huggingface.co/cstr/yunet-GGUF"},

    {"clip-vit-base-patch16",
     "clip-vit-base-patch16.gguf",
     "https://huggingface.co/cstr/clip-vit-base-patch16-GGUF/resolve/main/clip-vit-base-patch16.gguf",
     "CLIP ViT-B/16 vision encoder (86M)", "329 MB", "mit",
     "https://huggingface.co/openai/clip-vit-base-patch16"},

    {"clip-vit-large-patch14",
     "clip-vit-large-patch14.gguf",
     "https://huggingface.co/cstr/clip-vit-large-patch14-GGUF/resolve/main/clip-vit-large-patch14.gguf",
     "CLIP ViT-L/14 vision encoder (304M)", "1.2 GB", "mit",
     "https://huggingface.co/openai/clip-vit-large-patch14"},

    {"clip-text-base",
     "clip-text-base.gguf",
     "https://huggingface.co/cstr/clip-text-base-GGUF/resolve/main/clip-text-base.gguf",
     "CLIP text encoder base (63M, 512d)", "244 MB", "mit",
     "https://huggingface.co/openai/clip-vit-base-patch16"},

    {"clip-text-large",
     "clip-text-large.gguf",
     "https://huggingface.co/cstr/clip-text-large-GGUF/resolve/main/clip-text-large.gguf",
     "CLIP text encoder large (124M, 768d)", "480 MB", "mit",
     "https://huggingface.co/openai/clip-vit-large-patch14"},

    {"siglip-large-256",
     "siglip-large-256.gguf",
     "https://huggingface.co/cstr/siglip-large-256-GGUF/resolve/main/siglip-large-256.gguf",
     "SigLIP ViT-L/16 vision encoder 256x256 (304M)", "1.2 GB", "apache-2.0",
     "https://huggingface.co/google/siglip-large-patch16-256"},

    {"siglip-so400m-patch14-384",
     "siglip-so400m-patch14-384.gguf",
     "https://huggingface.co/cstr/siglip-so400m-patch14-384-GGUF/resolve/main/siglip-so400m-patch14-384.gguf",
     "SigLIP SoViT-400M/14 vision encoder 384x384 (428M)", "1.6 GB", "apache-2.0",
     "https://huggingface.co/google/siglip-so400m-patch14-384"},

    {"clip-vit-large-patch14-336",
     "clip-vit-large-patch14-336.gguf",
     "https://huggingface.co/cstr/clip-vit-large-patch14-336-GGUF/resolve/main/clip-vit-large-patch14-336.gguf",
     "CLIP ViT-L/14@336px vision encoder (304M)", "1.2 GB", "mit",
     "https://huggingface.co/openai/clip-vit-large-patch14-336"},

    {"siglip-base",
     "siglip-base.gguf",
     "https://huggingface.co/cstr/siglip-base-GGUF/resolve/main/siglip-base.gguf",
     "SigLIP ViT-B/16 vision encoder 384x384 (93M)", "354 MB", "apache-2.0",
     "https://huggingface.co/google/siglip-base-patch16-384"},

    {"siglip-text-base",
     "siglip-text-base.gguf",
     "https://huggingface.co/cstr/siglip-text-base-GGUF/resolve/main/siglip-text-base.gguf",
     "SigLIP text encoder base (93M, 768d)", "421 MB", "apache-2.0",
     "https://huggingface.co/google/siglip-base-patch16-224"},

    {"scrfd-det-10g",
     "scrfd-det-10g.gguf",
     "https://huggingface.co/cstr/scrfd-det-10g-GGUF/resolve/main/scrfd-det-10g.gguf",
     "SCRFD face detection (ResNet-50 FPN, 640x640)", "16 MB", "apache-2.0",
     "https://huggingface.co/cstr/scrfd-det-10g-GGUF"},

    {"auraface-v1",
     "auraface-v1.gguf",
     "https://huggingface.co/cstr/auraface-v1-GGUF/resolve/main/auraface-v1.gguf",
     "AuraFace face recognition (ResNet-100, 512d)", "249 MB", "apache-2.0",
     "https://huggingface.co/cstr/auraface-v1-GGUF"},

    {"sface",
     "sface.gguf",
     "https://huggingface.co/cstr/sface-GGUF/resolve/main/sface.gguf",
     "SFace face recognition (MobileFaceNet, 128d)", "37 MB", "apache-2.0",
     "https://huggingface.co/cstr/sface-GGUF"},

    {"hmer-hw",
     "hmer-hw-q4_k.gguf",
     "https://huggingface.co/cstr/hmer-handwritten-math-gguf/resolve/main/hmer-hw-q4_k.gguf",
     "HMER handwritten math OCR (DenseNet-121+GRU, 112 tokens)", "5 MB", "mit",
     "https://huggingface.co/cstr/hmer-handwritten-math-gguf"},

    {"bttr-hw",
     "bttr-hw-q4_k.gguf",
     "https://huggingface.co/cstr/bttr-handwritten-math-gguf/resolve/main/bttr-hw-q4_k.gguf",
     "BTTR handwritten math OCR (DenseNet+Transformer, 113 tokens)", "5 MB", "mit",
     "https://huggingface.co/cstr/bttr-handwritten-math-gguf"},

    {"ppformulanet-l",
     "ppformulanet-l-q8_0.gguf",
     "https://huggingface.co/cstr/ppformulanet-l-gguf/resolve/main/ppformulanet-l-q8_0.gguf",
     "PP-FormulaNet-L printed math OCR (SAM-ViT+MBart, 181M)", "180 MB", "apache-2.0",
     "https://huggingface.co/cstr/ppformulanet-l-gguf"},

    {"posformer-crohme",
     "posformer-crohme-q8_0.gguf",
     "https://huggingface.co/cstr/posformer-crohme-GGUF/resolve/main/posformer-crohme-q8_0.gguf",
     "PosFormer handwritten math OCR (DenseNet+Transformer+ARM, 57% CROHME)", "12 MB", "cc-by-nc-sa-3.0",
     "https://huggingface.co/cstr/posformer-crohme-GGUF"},

    {"pix2tex-mfr",
     "pix2tex-mfr-q4_k.gguf",
     "https://huggingface.co/cstr/pix2tex-mfr-gguf/resolve/main/pix2tex-mfr-q4_k.gguf",
     "pix2tex printed math OCR (DeiT+TrOCR, 28M)", "17 MB", "mit",
     "https://huggingface.co/cstr/pix2tex-mfr-gguf"},

    {"texo-distill",
     "texo-distill-q8_0.gguf",
     "https://huggingface.co/cstr/texo-distill-gguf/resolve/main/texo-distill-q8_0.gguf",
     "Texo-Distill printed math OCR (HGNetv2+MBart, 20M, BLEU 0.90)", "22 MB", "agpl-3.0",
     "https://huggingface.co/cstr/texo-distill-gguf"},

    {"dbnet-det",
     "dbnet-ic15-q4_k.gguf",
     "https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q4_k.gguf",
     "DBNet text detection (ResNet-18+FPNC, PP-OCRv4)", "7 MB", "apache-2.0",
     "https://huggingface.co/cstr/dbnet-ic15-GGUF"},

    {"surya-det",
     "surya-det-f16.gguf",
     "https://huggingface.co/cstr/surya-det-GGUF/resolve/main/surya-det-f16.gguf",
     "surya-ocr-2 text detection (EfficientViT segformer, 38M, 91 langs)", "73 MB", "openrail-m",
     "https://huggingface.co/cstr/surya-det-GGUF"},

    {"mixtex-zhen",
     "mixtex-zhen-f16.gguf",
     "",
     "MixTex Chinese+English LaTeX OCR (Swin-Tiny+RoBERTa, 86M)", "165 MB", "apache-2.0",
     ""},

    {"trocr-printed",
     "trocr-small-printed-q8_0.gguf",
     "https://huggingface.co/cstr/trocr-small-printed-GGUF/resolve/main/trocr-small-printed-q8_0.gguf",
     "TrOCR text recognition (DeiT+Transformer, printed)", "63 MB", "mit",
     "https://huggingface.co/cstr/trocr-small-printed-GGUF"},

    {"layout-heron",
     "layout-heron-f32.gguf",
     "https://huggingface.co/cstr/layout-heron-gguf/resolve/main/layout-heron-f32.gguf",
     "RT-DETRv2 document layout detection (ResNet-50+Transformer, 17 classes)", "161 MB", "apache-2.0",
     "https://huggingface.co/cstr/layout-heron-gguf"},

    {"qwen2vl-3b",
     "qwen2.5-vl-3b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen2.5-vl-3b-crispembed-GGUF/resolve/main/qwen2.5-vl-3b-q4_k.gguf",
     "Qwen2.5-VL-3B VLM OCR (32-layer ViT + 36-layer Qwen2.5, German docs)", "2610 MB", "apache-2.0",
     "https://huggingface.co/cstr/qwen2.5-vl-3b-crispembed-GGUF"},

    {"internvl2-2b",
     "internvl2.5-2b-q4_k.gguf",
     "https://huggingface.co/cstr/internvl2.5-2b-crispembed-GGUF/resolve/main/internvl2.5-2b-q4_k.gguf",
     "InternVL2.5-2B VLM OCR (InternViT-300M + InternLM2.5-1.8B, EN+DE, OCRBench ~830)", "1400 MB", "mit",
     "https://huggingface.co/cstr/internvl2.5-2b-crispembed-GGUF"},

    {"gliner-lfm",
     "gliner-lfm-q8_0.gguf",
     "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF/resolve/main/gliner-lfm-q8_0.gguf",
     "GLiNER zero-shot NER (LFM2.5-350M bidirectional, 5 languages)", "419 MB", "lfm1.0*",
     "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF"},

    {"gliner-lfm-q4k",
     "gliner-lfm-q4_k.gguf",
     "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF/resolve/main/gliner-lfm-q4_k.gguf",
     "GLiNER zero-shot NER (LFM2.5-350M, Q4_K compact)", "254 MB", "lfm1.0*",
     "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF"},

    {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}
};

std::string cache_dir() {
    // Check env override
    const char * env = std::getenv("CRISPEMBED_CACHE_DIR");
    if (env && env[0]) {
        std::string value = env;
        size_t start = 0;
        while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
            start++;
        }
        size_t end = value.size();
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
            end--;
        }
        value = value.substr(start, end - start);
        if (!value.empty()) return value;
    }

    // Default: per-user cache dir unless CRISPEMBED_CACHE_DIR is set.
    std::string home;
#ifdef _WIN32
    const char * h = std::getenv("USERPROFILE");
    if (!h) h = std::getenv("HOME");
    if (h) home = h;
#else
    const char * h = std::getenv("HOME");
    if (h) home = h;
#endif
    if (home.empty()) home = "/tmp";
    return (std::filesystem::path(home) / ".cache" / "crispembed").string();
}

static bool file_exists(const std::string & path) {
    // Use _stat64 on Windows: regular stat() has 32-bit st_size which overflows for files > 2 GB
#ifdef _WIN32
    struct __stat64 st;
    return _stat64(path.c_str(), &st) == 0 && st.st_size > 0;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && st.st_size > 0;
#endif
}

static void mkdirs(const std::string & path) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path), ec);
}

static long long file_size(const std::string & path) {
#ifdef _WIN32
    struct __stat64 st;
    if (_stat64(path.c_str(), &st) != 0) return -1;
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return -1;
#endif
    return static_cast<long long>(st.st_size);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
static bool download_file(const std::string & source_url, const std::string & dest_path) {
#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    (void)source_url;
    (void)dest_path;
    return false;
#else
    std::string tmp = dest_path + ".tmp";

    // Resume-aware: if a previous attempt left a partial .tmp, log the
    // recovery and pass `-C -` / `-c` to curl / wget so we resume from
    // there instead of starting over. Hours of bandwidth saved on flaky
    // links. On success the .tmp is renamed; on failure we deliberately
    // KEEP the .tmp so the next attempt can resume rather than redo.
    long long resume_from = file_size(tmp);
    if (resume_from > 0) {
        fprintf(stderr, "Resuming download from %.1f MB...\n",
                resume_from / (1024.0 * 1024.0));
    }

    // `-C -` (curl) and `-c` (wget) tell the client to ask for HTTP Range
    // bytes=N- where N is the existing local size. Both handle the
    // already-complete case (HTTP 416) gracefully.
#ifdef _WIN32
    // Windows 10+ bundles curl.exe — supports -C -.
    std::string cmd = "curl.exe -fL -C - --progress-bar -o \"" + tmp +
                      "\" \"" + source_url + "\"";
#else
    std::string cmd = "curl -fL -C - --progress-bar -o \"" + tmp +
                      "\" \"" + source_url + "\"";
#endif
    // NOLINTNEXTLINE(bugprone-command-processor)
    int ret = system(cmd.c_str());
    if (ret == 0 && file_exists(tmp)) {
        rename(tmp.c_str(), dest_path.c_str());
        return true;
    }

#ifndef _WIN32
    // wget fallback (Linux/macOS only). `-c` resumes the partial .tmp.
    cmd = "wget -c -q --show-progress -O \"" + tmp + "\" \"" + source_url + "\"";
    // NOLINTNEXTLINE(bugprone-command-processor)
    ret = system(cmd.c_str());
    if (ret == 0 && file_exists(tmp)) {
        rename(tmp.c_str(), dest_path.c_str());
        return true;
    }
#endif

    // Both attempts failed. Keep the .tmp so the next invocation can
    // resume from wherever curl/wget reached. If the partial is corrupt
    // (server changed, mid-byte error) the caller can `rm <cache>/*.tmp`
    // manually — bandwidth-cheap to lose, expensive to redo from zero.
    long long partial = file_size(tmp);
    if (partial > 0) {
        fprintf(stderr,
                "Download failed; partial %.1f MB kept at %s — re-run to resume.\n",
                partial / (1024.0 * 1024.0), tmp.c_str());
    }
    return false;
#endif
}

bool license_requires_acceptance(const char * spdx) {
    if (!spdx || !*spdx) return false;
    // cc-by-nc-* (and friends)
    if (strncmp(spdx, "cc-by-nc", 8) == 0) return true;
    // Vendor licenses with use-restriction policies the user must accept.
    static const char * restricted[] = {
        "gemma", "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3",
        "llama4", "qwen-research", "mistral-ai-research", "other", nullptr,
    };
    for (const char ** p = restricted; *p; ++p) {
        if (strcmp(spdx, *p) == 0) return true;
    }
    return false;
}

static bool license_accepted(const char * spdx,
                              const std::string & accepted_arg) {
    auto matches = [&](const std::string & accepted) {
        if (accepted.empty()) return false;
        if (accepted == "all" || accepted == "*") return true;
        return accepted == spdx;
    };
    if (matches(accepted_arg)) return true;
    const char * env = std::getenv("CRISPEMBED_ACCEPT_LICENSE");
    if (env && matches(env)) return true;
    return false;
}

std::string resolve_model(const std::string & arg, bool auto_download,
                          const std::string & accepted_license) {
    // If it's already a file path, use it directly
    if (file_exists(arg)) return arg;

    // Look up in registry
    const ModelEntry * entry = nullptr;
    for (const ModelEntry * e = k_registry; e->name; e++) {
        if (arg == e->name || arg == e->filename) {
            entry = e;
            break;
        }
    }

    // Fuzzy match: check if arg is a substring of any model name
    if (!entry) {
        for (const ModelEntry * e = k_registry; e->name; e++) {
            if (strstr(e->name, arg.c_str()) || strstr(e->filename, arg.c_str())) {
                entry = e;
                break;
            }
        }
    }

    if (!entry) {
        fprintf(stderr, "Unknown model: '%s'\n", arg.c_str());
        fprintf(stderr, "Use --list-models to see available models.\n");
        return "";
    }

    // Check cache
    std::string dir = cache_dir();
    std::string cached = dir + "/" + entry->filename;

    if (file_exists(cached)) {
        return cached;
    }

    const bool restricted = license_requires_acceptance(entry->license);
    const bool accepted   = license_accepted(entry->license, accepted_license);
    const bool is_tty     = isatty(fileno(stdin));

    // Download flow:
    //   - For permissive licenses: existing behaviour (auto_download or
    //     interactive [y/N]).
    //   - For restricted licenses (cc-by-nc-*, gemma, other): require
    //     explicit acceptance via --accept-license <spdx>,
    //     CRISPEMBED_ACCEPT_LICENSE=<spdx>, or an interactive prompt that
    //     shows the license + model card URL. `auto_download` alone is NOT
    //     sufficient.
    if (restricted) {
        if (!accepted) {
            if (is_tty) {
                fprintf(stderr,
                        "Model '%s' is released under a restricted license:\n",
                        entry->name);
                fprintf(stderr, "  License:    %s\n", entry->license);
                fprintf(stderr, "  Model card: %s\n",
                        entry->model_card_url ? entry->model_card_url : "(unknown)");
                if (strncmp(entry->license, "cc-by-nc", 8) == 0) {
                    fprintf(stderr, "  Notice:     non-commercial use only — see upstream model card for terms.\n");
                } else if (strcmp(entry->license, "gemma") == 0) {
                    fprintf(stderr, "  Notice:     governed by Google's Gemma Terms of Use & Prohibited Use Policy.\n");
                } else {
                    fprintf(stderr, "  Notice:     review the upstream model card for the full license terms.\n");
                }
                fprintf(stderr, "Download %s (%s) and accept this license? [y/N] ",
                        entry->filename, entry->approx_size);
                char c = 0;
                if (scanf(" %c", &c) != 1 || (c != 'y' && c != 'Y')) {
                    return "";
                }
            } else {
                fprintf(stderr,
                        "error: model '%s' is released under '%s' (restricted).\n",
                        entry->name, entry->license);
                fprintf(stderr,
                        "       Pass --accept-license %s (or set "
                        "CRISPEMBED_ACCEPT_LICENSE=%s) to acknowledge.\n",
                        entry->license, entry->license);
                if (entry->model_card_url) {
                    fprintf(stderr, "       Model card: %s\n", entry->model_card_url);
                }
                return "";
            }
        }
    } else if (!auto_download) {
        if (is_tty) {
            fprintf(stderr, "Model '%s' not found locally.\n", entry->name);
            fprintf(stderr, "  License: %s   (%s)\n",
                    entry->license ? entry->license : "?",
                    entry->model_card_url ? entry->model_card_url : "");
            fprintf(stderr, "Download %s (%s) from HuggingFace? [y/N] ",
                    entry->filename, entry->approx_size);
            char c = 0;
            if (scanf(" %c", &c) != 1 || (c != 'y' && c != 'Y')) {
                return "";
            }
        } else {
            fprintf(stderr, "Model '%s' not found. Use --auto-download to download automatically.\n",
                    entry->name);
            return "";
        }
    }

    if (!download_supported()) {
        fprintf(stderr,
                "Model '%s' is not cached, and auto-download is unavailable on iOS builds.\n",
                entry->name);
        return "";
    }

    mkdirs(dir);
    fprintf(stderr, "Downloading %s (%s, license: %s)...\n",
            entry->filename, entry->approx_size,
            entry->license ? entry->license : "?");
    if (download_file(entry->url, cached)) {
        fprintf(stderr, "Downloaded to %s\n", cached.c_str());
        return cached;
    } else {
        fprintf(stderr, "Download failed.\n");
        return "";
    }
}

void list_models() {
    fprintf(stderr, "Available models:\n");
    fprintf(stderr, "  %-40s %-14s %-9s %s\n",
            "Name", "License", "Size", "Description");
    fprintf(stderr, "  %-40s %-14s %-9s %s\n",
            "----", "-------", "----", "-----------");
    for (const ModelEntry * e = k_registry; e->name; e++) {
        std::string cached = cache_dir() + "/" + e->filename;
        const char * status = file_exists(cached) ? " [cached]" : "";
        const char * license = e->license ? e->license : "?";
        const char * marker  = license_requires_acceptance(e->license) ? "*" : " ";
        fprintf(stderr, " %s%-40s %-14s %-9s %s%s\n",
                marker, e->name, license, e->approx_size, e->desc, status);
    }
    fprintf(stderr,
            "\n  * = restricted license (non-commercial or vendor terms); "
            "requires --accept-license <spdx> or interactive consent.\n");
}

int n_models() {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++) n++;
    return n;
}

const char * model_name(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->name;
    return nullptr;
}

const char * model_desc(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->desc;
    return nullptr;
}

const char * model_filename(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->filename;
    return nullptr;
}

const char * model_size(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->approx_size;
    return nullptr;
}

const char * model_license(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->license;
    return nullptr;
}

const char * model_card_url(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->model_card_url;
    return nullptr;
}

const char * get_query_prefix(const char * model) { return query_prefix(model); }
const char * get_passage_prefix(const char * model) { return passage_prefix(model); }

}  // namespace crispembed_mgr
