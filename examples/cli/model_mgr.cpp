// model_mgr.cpp — Auto-download model manager for CrispEmbed.

#include "model_mgr.h"

#include "crispembed.h"   // crispembed_accept_biometric_use
#include "model_hashes.h" // model_pinned_sha256 (generated)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cctype>
#include <string>
#include <vector>
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

// --- SHA-256 -------------------------------------------------------------
//
// Self-contained rather than shelling out to shasum/sha256sum/certutil: the
// three disagree on flags and output format across the platforms we ship, and
// an integrity check that silently degrades to "tool not found" is worse than
// none. FIPS 180-4, streaming so a multi-GB GGUF never lands in memory.

struct sha256_state {
    uint32_t h[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    uint64_t total_bits = 0;
    uint8_t buf[64] = {};
    size_t buf_len = 0;
};

inline uint32_t sha_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

void sha256_compress(sha256_state & s, const uint8_t * block) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };

    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) | ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = sha_rotr(w[i - 15], 7) ^ sha_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = sha_rotr(w[i - 2], 17) ^ sha_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3];
    uint32_t e = s.h[4], f = s.h[5], g = s.h[6], hh = s.h[7];

    for (int i = 0; i < 64; i++) {
        const uint32_t S1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
        const uint32_t S0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;

        hh = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    s.h[0] += a;
    s.h[1] += b;
    s.h[2] += c;
    s.h[3] += d;
    s.h[4] += e;
    s.h[5] += f;
    s.h[6] += g;
    s.h[7] += hh;
}

void sha256_update(sha256_state & s, const uint8_t * data, size_t len) {
    s.total_bits += (uint64_t)len * 8;
    while (len > 0) {
        const size_t take = std::min(len, sizeof(s.buf) - s.buf_len);
        memcpy(s.buf + s.buf_len, data, take);
        s.buf_len += take;
        data += take;
        len -= take;
        if (s.buf_len == sizeof(s.buf)) {
            sha256_compress(s, s.buf);
            s.buf_len = 0;
        }
    }
}

std::string sha256_finish(sha256_state & s) {
    // Pad in place rather than via sha256_update(), which would fold the
    // padding into total_bits and corrupt the length field.
    const uint64_t bits = s.total_bits;

    s.buf[s.buf_len++] = 0x80;
    if (s.buf_len > 56) {
        memset(s.buf + s.buf_len, 0, 64 - s.buf_len);
        sha256_compress(s, s.buf);
        s.buf_len = 0;
    }
    memset(s.buf + s.buf_len, 0, 56 - s.buf_len);
    for (int i = 0; i < 8; i++) s.buf[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    sha256_compress(s, s.buf);

    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + i * 8, 9, "%08x", s.h[i]);
    return std::string(out, 64);
}

// Lowercase hex digest of a file, or "" when it cannot be read.
std::string sha256_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return "";

    sha256_state s;
    std::vector<uint8_t> chunk(1 << 20);
    for (;;) {
        const size_t n = fread(chunk.data(), 1, chunk.size(), f);
        if (n > 0) sha256_update(s, chunk.data(), n);
        if (n < chunk.size()) break;
    }
    const bool ok = ferror(f) == 0;
    fclose(f);
    if (!ok) return "";
    return sha256_finish(s);
}

bool url_is_https(const std::string & url) {
    return url.rfind("https://", 0) == 0;
}

bool unpinned_downloads_allowed() {
    const char * env = std::getenv("CRISPEMBED_ALLOW_UNPINNED_MODEL");
    return env && *env && strcmp(env, "0") != 0;
}

} // namespace

struct ModelEntry {
    const char * name;
    const char * filename;
    const char * url;
    const char * desc;
    const char * approx_size;
    const char * license;        // SPDX-style tag from the upstream model
                                 // card (NOT from the cstr/* re-host).
                                 // Verified by tests/check_registry_licenses.py.
    const char * model_card_url; // upstream HuggingFace model card
    // Scripts the model's recognition dictionary can actually emit, scanned
    // from the shipped GGUF by tools/scan_model_languages.py. Empty means
    // "not scanned", never "no coverage" — only OCR recognizers carry it.
    //
    // Coverage is NECESSARY BUT NOT SUFFICIENT for quality: kana in the dict
    // says the model CAN emit kana, not that it reads Japanese well. Zero
    // coverage is the sufficient direction, and that is what this field is
    // for: ppocrv6-tiny-rec has no kana at all and used to fail silently on
    // Japanese (issue #44). Evidence tiers live in docs/LANGUAGES.md.
    const char * languages;

    // Optional second file installed into the SAME cache directory as
    // `filename`. Vision-language backends keep the vision tower in an
    // `mmproj-*.gguf` sibling and find it by scanning the model's own
    // directory, so registering the LLM alone would install something that
    // cannot load. Both files are pinned: tools/fetch_model_hashes.py scans
    // every resolve-URL in this array, so a companion URL is covered without
    // any change to the generator.
    //
    // Names and order mirror CrispASR's `src/crispasr_model_registry.cpp`,
    // which has carried the same three fields for moonshine's tokenizer.bin,
    // kokoro's voice GGUF and the TTS codec companions — same concept, so the
    // two registries should read the same.
    //
    // Trailing fields, so the ~600 existing entries — which stop at `license`
    // or `model_card_url` — value-initialise these to nullptr and need no edit.
    const char * companion_file;
    const char * companion_url;
    const char * companion_size; // NULL falls back to approx_size
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
    if (strstr(model, "-e5-")) return "query: ";
    // Snowflake Arctic Embed. v2.0 (l-v2 / m-v2) ships prompts.query = "query: ";
    // v1 (xs / m / l) ships the BGE-style instruction. Documents take no prefix
    // in either generation, so passage_prefix() stays silent for arctic.
    if (strstr(model, "arctic-embed")) {
        if (strstr(model, "-v2")) return "query: ";
        return "Represent this sentence for searching relevant passages: ";
    }
    // Nomic
    if (strstr(model, "nomic-embed")) return "search_query: ";
    // Jina v5
    if (strstr(model, "jina-v5")) return "Query: ";
    // MOST Embed DE (SentenceTransformers prompts.query)
    if (strstr(model, "most-embed-de")) return "query: ";
    // F2LLM-v2 — instruction-style prompt, verbatim from the family's
    // config_sentence_transformers.json "query" prompt (the trailing newline
    // before "Query: " is load-bearing: it is a distinct token, and dropping
    // it measurably moves the embedding). Documents get NO prefix.
    if (strstr(model, "f2llm-v2"))
        return "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: ";
    // LFM2.5 Embedding / ColBERT
    if (strstr(model, "lfm2-embed") || strstr(model, "lfm2.5-embed") || strstr(model, "lfm2-colbert") ||
        strstr(model, "lfm2.5-colbert"))
        return "query: ";
    return nullptr;
}

static const char * passage_prefix(const char * model) {
    if (!model) return nullptr;
    // E5 models
    if (strstr(model, "-e5-")) return "passage: ";
    // MOST Embed DE (SentenceTransformers prompts.document)
    if (strstr(model, "most-embed-de")) return "passage: ";
    // Nomic
    if (strstr(model, "nomic-embed")) return "search_document: ";
    // Jina v5
    if (strstr(model, "jina-v5")) return "Passage: ";
    // LFM2.5 Embedding / ColBERT
    if (strstr(model, "lfm2-embed") || strstr(model, "lfm2.5-embed") || strstr(model, "lfm2-colbert") ||
        strstr(model, "lfm2.5-colbert"))
        return "document: ";
    return nullptr;
}

static const ModelEntry k_registry[] = {
    { "all-MiniLM-L6-v2", "all-MiniLM-L6-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "19 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" },
    { "all-MiniLM-L6-v2-q4k", "all-MiniLM-L6-v2-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2-q4_k-imatrix.gguf",
      "BERT 384d English (Q4_K+imatrix)", "19 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" },
    { "all-MiniLM-L6-v2-iq4xs", "all-MiniLM-L6-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "19 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" },
    { "all-MiniLM-L6-v2-q8", "all-MiniLM-L6-v2-q8_0.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2-q8_0.gguf",
      "BERT 384d English (Q8_0)", "25 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" },

    { "gte-small", "gte-small-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-small-GGUF/resolve/main/gte-small-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "25 MB", "mit", "https://huggingface.co/thenlper/gte-small" },
    { "gte-small-q4k", "gte-small-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/gte-small-GGUF/resolve/main/gte-small-q4_k-imatrix.gguf",
      "BERT 384d English (Q4_K+imatrix)", "25 MB", "mit", "https://huggingface.co/thenlper/gte-small" },
    { "gte-small-iq4xs", "gte-small-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-small-GGUF/resolve/main/gte-small-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "25 MB", "mit", "https://huggingface.co/thenlper/gte-small" },
    { "gte-small-q8", "gte-small-q8_0.gguf",
      "https://huggingface.co/cstr/gte-small-GGUF/resolve/main/gte-small-q8_0.gguf", "BERT 384d English (Q8_0)",
      "36 MB", "mit", "https://huggingface.co/thenlper/gte-small" },

    { "arctic-embed-xs", "arctic-embed-xs-iq4_xs.gguf",
      "https://huggingface.co/cstr/arctic-embed-xs-GGUF/resolve/main/arctic-embed-xs-iq4_xs.gguf",
      "BERT 384d CLS English (IQ4_XS+imatrix)", "19 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs" },
    { "arctic-embed-xs-q4k", "arctic-embed-xs-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/arctic-embed-xs-GGUF/resolve/main/arctic-embed-xs-q4_k-imatrix.gguf",
      "BERT 384d CLS English (Q4_K+imatrix)", "19 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs" },
    { "arctic-embed-xs-iq4xs", "arctic-embed-xs-iq4_xs.gguf",
      "https://huggingface.co/cstr/arctic-embed-xs-GGUF/resolve/main/arctic-embed-xs-iq4_xs.gguf",
      "BERT 384d CLS English (IQ4_XS+imatrix)", "19 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs" },
    { "arctic-embed-xs-q8", "arctic-embed-xs-q8_0.gguf",
      "https://huggingface.co/cstr/arctic-embed-xs-GGUF/resolve/main/arctic-embed-xs-q8_0.gguf",
      "BERT 384d CLS English (Q8_0)", "25 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs" },

    { "multilingual-e5-small", "multilingual-e5-small-iq4_xs.gguf",
      "https://huggingface.co/cstr/multilingual-e5-small-GGUF/resolve/main/multilingual-e5-small-iq4_xs.gguf",
      "XLM-R 384d multilingual (IQ4_XS+imatrix)", "121 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-small" },
    { "multilingual-e5-small-q4k", "multilingual-e5-small-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/multilingual-e5-small-GGUF/resolve/main/multilingual-e5-small-q4_k-imatrix.gguf",
      "XLM-R 384d multilingual (Q4_K+imatrix)", "121 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-small" },
    { "multilingual-e5-small-iq4xs", "multilingual-e5-small-iq4_xs.gguf",
      "https://huggingface.co/cstr/multilingual-e5-small-GGUF/resolve/main/multilingual-e5-small-iq4_xs.gguf",
      "XLM-R 384d multilingual (IQ4_XS+imatrix)", "121 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-small" },
    { "multilingual-e5-small-q8", "multilingual-e5-small-q8_0.gguf",
      "https://huggingface.co/cstr/multilingual-e5-small-GGUF/resolve/main/multilingual-e5-small-q8_0.gguf",
      "XLM-R 384d multilingual (Q8_0)", "132 MB", "mit", "https://huggingface.co/intfloat/multilingual-e5-small" },

    { "pixie-rune-v1", "pixie-rune-v1-iq4_xs.gguf",
      "https://huggingface.co/cstr/pixie-rune-v1-GGUF/resolve/main/pixie-rune-v1-iq4_xs.gguf",
      "XLM-R 1024d 74-lang CLS (IQ4_XS+imatrix)", "449 MB", "apache-2.0",
      "https://huggingface.co/telepix/PIXIE-Rune-v1.0" },
    { "pixie-rune-v1-q4k", "pixie-rune-v1-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/pixie-rune-v1-GGUF/resolve/main/pixie-rune-v1-q4_k-imatrix.gguf",
      "XLM-R 1024d 74-lang CLS (Q4_K+imatrix)", "459 MB", "apache-2.0",
      "https://huggingface.co/telepix/PIXIE-Rune-v1.0" },
    { "pixie-rune-v1-iq4xs", "pixie-rune-v1-iq4_xs.gguf",
      "https://huggingface.co/cstr/pixie-rune-v1-GGUF/resolve/main/pixie-rune-v1-iq4_xs.gguf",
      "XLM-R 1024d 74-lang CLS (IQ4_XS+imatrix)", "449 MB", "apache-2.0",
      "https://huggingface.co/telepix/PIXIE-Rune-v1.0" },
    { "pixie-rune-v1-q8", "pixie-rune-v1-q8_0.gguf",
      "https://huggingface.co/cstr/pixie-rune-v1-GGUF/resolve/main/pixie-rune-v1-q8_0.gguf",
      "XLM-R 1024d 74-lang CLS (Q8_0)", "610 MB", "apache-2.0", "https://huggingface.co/telepix/PIXIE-Rune-v1.0" },

    { "arctic-embed-l-v2", "arctic-embed-l-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/arctic-embed-l-v2-GGUF/resolve/main/arctic-embed-l-v2-iq4_xs.gguf",
      "XLM-R 1024d CLS English (IQ4_XS+imatrix)", "449 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0" },
    { "arctic-embed-l-v2-q4k", "arctic-embed-l-v2-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/arctic-embed-l-v2-GGUF/resolve/main/arctic-embed-l-v2-q4_k-imatrix.gguf",
      "XLM-R 1024d CLS English (Q4_K+imatrix)", "459 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0" },
    { "arctic-embed-l-v2-iq4xs", "arctic-embed-l-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/arctic-embed-l-v2-GGUF/resolve/main/arctic-embed-l-v2-iq4_xs.gguf",
      "XLM-R 1024d CLS English (IQ4_XS+imatrix)", "449 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0" },
    { "arctic-embed-l-v2-q8", "arctic-embed-l-v2-q8_0.gguf",
      "https://huggingface.co/cstr/arctic-embed-l-v2-GGUF/resolve/main/arctic-embed-l-v2-q8_0.gguf",
      "XLM-R 1024d CLS English (Q8_0)", "610 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0" },

    // Arctic Embed M v2.0 is GTE v1.5 (RoPE + GeGLU + post-LN), NOT the XLM-R
    // backbone its l-v2 sibling uses; only the SentencePiece vocab is shared.
    // Default stays Q8_0. An imatrix HAS now been calibrated (PLAN.md T19-E3,
    // 134 German/English/query-prompt/newline texts). Over 65 held-out texts vs
    // the F32 base: Q8_0 min 0.9994 / mean 0.9996; Q4_K 0.9466/0.9584, and
    // +imatrix only 0.9480/0.9614. **IQ4_XS+imatrix is the one to reach for
    // below Q8_0** — 0.9667/0.9757 AND smaller (270 vs 274 MB); -iq4xs serves
    // it. Q4_K's weak showing has a known cause: the imatrix reaches only 36 of
    // the 73 quantized tensors here, because the runtime pre-merges q/k/v into
    // one UNNAMED tensor (crispembed.cpp:799-832), so the collector files its
    // statistics under ggml's auto name and every attn.{q,k,v}.weight is
    // quantized without importance. Fixing that naming is the lever.
    { "arctic-embed-m-v2", "arctic-embed-m-v2-q8_0.gguf",
      "https://huggingface.co/cstr/arctic-embed-m-v2-GGUF/resolve/main/arctic-embed-m-v2-q8_0.gguf",
      "GTE-v1.5 768d CLS multilingual (Q8_0)", "315 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0" },
    { "arctic-embed-m-v2-q8", "arctic-embed-m-v2-q8_0.gguf",
      "https://huggingface.co/cstr/arctic-embed-m-v2-GGUF/resolve/main/arctic-embed-m-v2-q8_0.gguf",
      "GTE-v1.5 768d CLS multilingual (Q8_0)", "315 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0" },
    // G3 (post-F7b): both sub-Q8 aliases re-pinned to the -f7 re-quants built
    // with real q/k/v importance. Kaggle x86 A/B + local Metal/CPU cross-check
    // (2026-08-05): q4_k+imatrix mean .9614→.9937, iq4_xs .9757→.9867 vs
    // full-precision gold; backend delta ≤0.002. Old artifacts remain on HF.
    { "arctic-embed-m-v2-q4k", "arctic-embed-m-v2-q4_k-imatrix-f7.gguf",
      "https://huggingface.co/cstr/arctic-embed-m-v2-GGUF/resolve/main/arctic-embed-m-v2-q4_k-imatrix-f7.gguf",
      "GTE-v1.5 768d CLS multilingual (Q4_K+imatrix)", "261 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0" },
    { "arctic-embed-m-v2-iq4xs", "arctic-embed-m-v2-iq4_xs-f7.gguf",
      "https://huggingface.co/cstr/arctic-embed-m-v2-GGUF/resolve/main/arctic-embed-m-v2-iq4_xs-f7.gguf",
      "GTE-v1.5 768d CLS multilingual (IQ4_XS+imatrix)", "258 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0" },

    { "octen-0.6b", "octen-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/octen-0.6b-GGUF/resolve/main/octen-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d multilingual (Q4_K+imatrix)", "419 MB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-0.6B" },
    { "octen-0.6b-q4k", "octen-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/octen-0.6b-GGUF/resolve/main/octen-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d multilingual (Q4_K+imatrix)", "419 MB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-0.6B" },
    { "octen-0.6b-iq4xs", "octen-0.6b-iq4_xs.gguf",
      "https://huggingface.co/cstr/octen-0.6b-GGUF/resolve/main/octen-0.6b-iq4_xs.gguf",
      "Qwen3 1024d multilingual (IQ4_XS+imatrix)", "405 MB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-0.6B" },
    { "octen-0.6b-q8", "octen-0.6b-q8_0.gguf",
      "https://huggingface.co/cstr/octen-0.6b-GGUF/resolve/main/octen-0.6b-q8_0.gguf",
      "Qwen3 1024d multilingual (Q8_0)", "639 MB", "apache-2.0", "https://huggingface.co/Octen/Octen-Embedding-0.6B" },

    { "octen-4b", "octen-4b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/octen-4b-GGUF/resolve/main/octen-4b-q4_k-imatrix.gguf",
      "Qwen3 2560d multilingual (Q4_K+imatrix)", "2.5 GB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-4B" },
    { "octen-4b-q4k", "octen-4b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/octen-4b-GGUF/resolve/main/octen-4b-q4_k-imatrix.gguf",
      "Qwen3 2560d multilingual (Q4_K+imatrix)", "2.5 GB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-4B" },
    { "octen-4b-iq4xs", "octen-4b-iq4_xs.gguf",
      "https://huggingface.co/cstr/octen-4b-GGUF/resolve/main/octen-4b-iq4_xs.gguf",
      "Qwen3 2560d multilingual (IQ4_XS+imatrix)", "2.3 GB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-4B" },
    { "octen-4b-q8", "octen-4b-q8_0.gguf", "https://huggingface.co/cstr/octen-4b-GGUF/resolve/main/octen-4b-q8_0.gguf",
      "Qwen3 2560d multilingual (Q8_0)", "4.3 GB", "apache-2.0", "https://huggingface.co/Octen/Octen-Embedding-4B" },

    { "octen-8b", "octen-8b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/octen-8b-GGUF/resolve/main/octen-8b-q4_k-imatrix.gguf",
      "Qwen3 4096d multilingual (Q4_K+imatrix)", "4.6 GB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-8B" },
    { "octen-8b-q4k", "octen-8b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/octen-8b-GGUF/resolve/main/octen-8b-q4_k-imatrix.gguf",
      "Qwen3 4096d multilingual (Q4_K+imatrix)", "4.6 GB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-8B" },
    { "octen-8b-iq4xs", "octen-8b-iq4_xs.gguf",
      "https://huggingface.co/cstr/octen-8b-GGUF/resolve/main/octen-8b-iq4_xs.gguf",
      "Qwen3 4096d multilingual (IQ4_XS+imatrix)", "4.4 GB", "apache-2.0",
      "https://huggingface.co/Octen/Octen-Embedding-8B" },
    { "octen-8b-q8", "octen-8b-q8_0.gguf", "https://huggingface.co/cstr/octen-8b-GGUF/resolve/main/octen-8b-q8_0.gguf",
      "Qwen3 4096d multilingual (Q8_0)", "8.0 GB", "apache-2.0", "https://huggingface.co/Octen/Octen-Embedding-8B" },

    // The 80M/160M/330M are all pruned from the 0.6B base. All three survive
    // Q8_0 far better than the 0.6B itself does (worst cosine vs the f32 HF
    // reference over 14 mixed German/English/code texts: 0.9996 / 0.9994 /
    // 0.9989, against 0.9909 for the 0.6B), so Q8_0 is the default for each.
    { "f2llm-v2-80m", "f2llm-v2-80m-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-80m-GGUF/resolve/main/f2llm-v2-80m-q8_0.gguf",
      "Qwen3 320d multilingual (Q8_0)", "86 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-80M" },
    { "f2llm-v2-80m-q8", "f2llm-v2-80m-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-80m-GGUF/resolve/main/f2llm-v2-80m-q8_0.gguf",
      "Qwen3 320d multilingual (Q8_0)", "86 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-80M" },
    { "f2llm-v2-80m-f16", "f2llm-v2-80m.gguf",
      "https://huggingface.co/cstr/f2llm-v2-80m-GGUF/resolve/main/f2llm-v2-80m.gguf", "Qwen3 320d multilingual (F16)",
      "250 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-80M" },
    // Sub-Q8 flavors, imatrix-calibrated (PLAN.md T19-E3). Over 65 held-out
    // texts vs the F16 base: Q4_K 0.9727 mean / 0.9499 min, +imatrix
    // 0.9767/0.9455 (mean up, min DOWN), IQ4_XS+imatrix 0.9812/0.9601 — better
    // on both and marginally smaller, so -iq4xs is the one to prefer here.
    { "f2llm-v2-80m-q4k", "f2llm-v2-80m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/f2llm-v2-80m-GGUF/resolve/main/f2llm-v2-80m-q4_k-imatrix.gguf",
      "Qwen3 320d multilingual (Q4_K+imatrix)", "71 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-80M" },
    { "f2llm-v2-80m-iq4xs", "f2llm-v2-80m-iq4_xs.gguf",
      "https://huggingface.co/cstr/f2llm-v2-80m-GGUF/resolve/main/f2llm-v2-80m-iq4_xs.gguf",
      "Qwen3 320d multilingual (IQ4_XS+imatrix)", "71 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-80M" },

    // Default = Q8_0. The 160M is pruned from the 0.6B base and is the best
    // sub-200M German embedder on MTEB(deu, v1); unlike the 0.6B it survives
    // Q8_0 essentially intact (worst cosine 0.9994 vs the f32 HF reference
    // over 14 mixed German/English/code texts, against 0.9909 for the 0.6B).
    { "f2llm-v2-160m", "f2llm-v2-160m-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-160m-GGUF/resolve/main/f2llm-v2-160m-q8_0.gguf",
      "Qwen3 640d multilingual (Q8_0)", "166 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-160M" },
    { "f2llm-v2-160m-q8", "f2llm-v2-160m-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-160m-GGUF/resolve/main/f2llm-v2-160m-q8_0.gguf",
      "Qwen3 640d multilingual (Q8_0)", "166 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-160M" },
    { "f2llm-v2-160m-f16", "f2llm-v2-160m.gguf",
      "https://huggingface.co/cstr/f2llm-v2-160m-GGUF/resolve/main/f2llm-v2-160m.gguf", "Qwen3 640d multilingual (F16)",
      "494 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-160M" },
    // imatrix-calibrated sub-Q8 flavors (PLAN.md T19-E3). Over 65 held-out
    // texts vs the F16 base, imatrix lifts Q4_K on BOTH tails here: mean
    // 0.9652 -> 0.9719, min 0.9331 -> 0.9495. IQ4_XS+imatrix is better again
    // (0.9766/0.9645) and slightly smaller. Q8_0's 0.9996 stays the default.
    { "f2llm-v2-160m-q4k", "f2llm-v2-160m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/f2llm-v2-160m-GGUF/resolve/main/f2llm-v2-160m-q4_k-imatrix.gguf",
      "Qwen3 640d multilingual (Q4_K+imatrix)", "137 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-160M" },
    { "f2llm-v2-160m-iq4xs", "f2llm-v2-160m-iq4_xs.gguf",
      "https://huggingface.co/cstr/f2llm-v2-160m-GGUF/resolve/main/f2llm-v2-160m-iq4_xs.gguf",
      "Qwen3 640d multilingual (IQ4_XS+imatrix)", "136 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-160M" },

    { "f2llm-v2-330m", "f2llm-v2-330m-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-330m-GGUF/resolve/main/f2llm-v2-330m-q8_0.gguf",
      "Qwen3 896d multilingual (Q8_0)", "344 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-330M" },
    { "f2llm-v2-330m-q8", "f2llm-v2-330m-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-330m-GGUF/resolve/main/f2llm-v2-330m-q8_0.gguf",
      "Qwen3 896d multilingual (Q8_0)", "344 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-330M" },
    { "f2llm-v2-330m-f16", "f2llm-v2-330m.gguf",
      "https://huggingface.co/cstr/f2llm-v2-330m-GGUF/resolve/main/f2llm-v2-330m.gguf", "Qwen3 896d multilingual (F16)",
      "903 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-330M" },
    // imatrix-calibrated sub-Q8 flavors (PLAN.md T19-E3). This model degrades
    // fastest of the three pruned siblings at 4 bits and gains the most from
    // calibration: over 65 held-out texts vs the F16 base, Q4_K goes mean
    // 0.9230 -> 0.9501 and min 0.8840 -> 0.9179. IQ4_XS+imatrix is better still
    // (0.9619/0.9443) and smaller. Both stay well under Q8_0's 0.9992, default.
    { "f2llm-v2-330m-q4k", "f2llm-v2-330m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/f2llm-v2-330m-GGUF/resolve/main/f2llm-v2-330m-q4_k-imatrix.gguf",
      "Qwen3 896d multilingual (Q4_K+imatrix)", "250 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-330M" },
    { "f2llm-v2-330m-iq4xs", "f2llm-v2-330m-iq4_xs.gguf",
      "https://huggingface.co/cstr/f2llm-v2-330m-GGUF/resolve/main/f2llm-v2-330m-iq4_xs.gguf",
      "Qwen3 896d multilingual (IQ4_XS+imatrix)", "248 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-330M" },

    { "f2llm-v2-0.6b", "f2llm-v2-0.6b-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-0.6b-GGUF/resolve/main/f2llm-v2-0.6b-q8_0.gguf",
      "Qwen3 1024d multilingual (Q8_0)", "639 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B" },
    { "f2llm-v2-0.6b-q4k", "f2llm-v2-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/f2llm-v2-0.6b-GGUF/resolve/main/f2llm-v2-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d multilingual (Q4_K+imatrix)", "419 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B" },
    { "f2llm-v2-0.6b-iq4xs", "f2llm-v2-0.6b-iq4_xs.gguf",
      "https://huggingface.co/cstr/f2llm-v2-0.6b-GGUF/resolve/main/f2llm-v2-0.6b-iq4_xs.gguf",
      "Qwen3 1024d multilingual (IQ4_XS+imatrix)", "405 MB", "apache-2.0",
      "https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B" },
    { "f2llm-v2-0.6b-q8", "f2llm-v2-0.6b-q8_0.gguf",
      "https://huggingface.co/cstr/f2llm-v2-0.6b-GGUF/resolve/main/f2llm-v2-0.6b-q8_0.gguf",
      "Qwen3 1024d multilingual (Q8_0)", "639 MB", "apache-2.0", "https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B" },

    // Ministral3 attention is especially quantization-sensitive here. The
    // compact default keeps q/k/v/o projections at Q8_0 and quantizes the
    // remaining eligible matrices to Q4_K; Q8_0 is available for maximum
    // parity. Both variants inherit the fine-tune's non-commercial license.
    { "most-embed-de", "most-embed-de-q4_k-attn-q8.gguf",
      "https://huggingface.co/cstr/most-embed-de-GGUF/resolve/main/most-embed-de-q4_k-attn-q8.gguf",
      "Ministral3 2048d German retrieval (Q4_K + Q8 attention)", "918 MB", "cc-by-nc-4.0",
      "https://huggingface.co/malteos/most-embed-de" },
    { "most-embed-de-q4k", "most-embed-de-q4_k-attn-q8.gguf",
      "https://huggingface.co/cstr/most-embed-de-GGUF/resolve/main/most-embed-de-q4_k-attn-q8.gguf",
      "Ministral3 2048d German retrieval (Q4_K + Q8 attention)", "918 MB", "cc-by-nc-4.0",
      "https://huggingface.co/malteos/most-embed-de" },
    { "most-embed-de-q8", "most-embed-de-q8_0.gguf",
      "https://huggingface.co/cstr/most-embed-de-GGUF/resolve/main/most-embed-de-q8_0.gguf",
      "Ministral3 2048d German retrieval (Q8_0)", "1.22 GB", "cc-by-nc-4.0",
      "https://huggingface.co/malteos/most-embed-de" },

    // Default = best flavor (Q4_K+imatrix, A/B winner). -q4k serves the imatrix
    // build (same size, strictly better); -iq4xs and -q8 select other flavors.
    { "jina-v5-nano", "jina-v5-nano-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/jina-v5-nano-GGUF/resolve/main/jina-v5-nano-q4_k-imatrix.gguf",
      "Qwen3 1024d compact (210M, Q4_K+imatrix)", "176 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano" },
    { "jina-v5-nano-q4k", "jina-v5-nano-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/jina-v5-nano-GGUF/resolve/main/jina-v5-nano-q4_k-imatrix.gguf",
      "Qwen3 1024d compact (210M, Q4_K+imatrix)", "176 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano" },
    { "jina-v5-nano-iq4xs", "jina-v5-nano-iq4_xs.gguf",
      "https://huggingface.co/cstr/jina-v5-nano-GGUF/resolve/main/jina-v5-nano-iq4_xs.gguf",
      "Qwen3 1024d compact (210M, IQ4_XS+imatrix)", "173 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano" },
    { "jina-v5-nano-q8", "jina-v5-nano-q8_0.gguf",
      "https://huggingface.co/cstr/jina-v5-nano-GGUF/resolve/main/jina-v5-nano-q8_0.gguf",
      "Qwen3 1024d compact (210M, Q8_0)", "233 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano" },

    { "jina-v5-small", "jina-v5-small-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/jina-v5-small-GGUF/resolve/main/jina-v5-small-q4_k-imatrix.gguf",
      "Qwen3 1024d multilingual (600M, Q4_K+imatrix)", "419 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-small" },
    { "jina-v5-small-q4k", "jina-v5-small-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/jina-v5-small-GGUF/resolve/main/jina-v5-small-q4_k-imatrix.gguf",
      "Qwen3 1024d multilingual (600M, Q4_K+imatrix)", "419 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-small" },
    { "jina-v5-small-iq4xs", "jina-v5-small-iq4_xs.gguf",
      "https://huggingface.co/cstr/jina-v5-small-GGUF/resolve/main/jina-v5-small-iq4_xs.gguf",
      "Qwen3 1024d multilingual (600M, IQ4_XS+imatrix)", "406 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-small" },
    { "jina-v5-small-q8", "jina-v5-small-q8_0.gguf",
      "https://huggingface.co/cstr/jina-v5-small-GGUF/resolve/main/jina-v5-small-q8_0.gguf",
      "Qwen3 1024d multilingual (600M, Q8_0)", "639 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-embeddings-v5-text-small" },

    { "harrier-0.6b", "harrier-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/harrier-0.6b-GGUF/resolve/main/harrier-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d SOTA (Q4_K+imatrix)", "419 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-0.6b" },
    { "harrier-0.6b-q4k", "harrier-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/harrier-0.6b-GGUF/resolve/main/harrier-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d SOTA (Q4_K+imatrix)", "419 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-0.6b" },
    { "harrier-0.6b-iq4xs", "harrier-0.6b-iq4_xs.gguf",
      "https://huggingface.co/cstr/harrier-0.6b-GGUF/resolve/main/harrier-0.6b-iq4_xs.gguf",
      "Qwen3 1024d SOTA (IQ4_XS+imatrix)", "405 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-0.6b" },
    { "harrier-0.6b-q8", "harrier-0.6b-q8_0.gguf",
      "https://huggingface.co/cstr/harrier-0.6b-GGUF/resolve/main/harrier-0.6b-q8_0.gguf", "Qwen3 1024d SOTA (Q8_0)",
      "639 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-0.6b" },

    { "harrier-270m", "harrier-270m-iq4_xs.gguf",
      "https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-iq4_xs.gguf",
      "Gemma3 640d compact (IQ4_XS+imatrix)", "250 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-270m" },
    { "harrier-270m-q4k", "harrier-270m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q4_k-imatrix.gguf",
      "Gemma3 640d compact (Q4_K+imatrix)", "251 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-270m" },
    { "harrier-270m-iq4xs", "harrier-270m-iq4_xs.gguf",
      "https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-iq4_xs.gguf",
      "Gemma3 640d compact (IQ4_XS+imatrix)", "250 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-270m" },
    { "harrier-270m-q8", "harrier-270m-q8_0.gguf",
      "https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q8_0.gguf", "Gemma3 640d compact (Q8_0)",
      "301 MB", "mit", "https://huggingface.co/microsoft/harrier-oss-v1-270m" },

    { "qwen3-embed-0.6b", "qwen3-embed-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/qwen3-embed-0.6b-GGUF/resolve/main/qwen3-embed-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d official (Q4_K+imatrix)", "419 MB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B" },
    { "qwen3-embed-0.6b-q4k", "qwen3-embed-0.6b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/qwen3-embed-0.6b-GGUF/resolve/main/qwen3-embed-0.6b-q4_k-imatrix.gguf",
      "Qwen3 1024d official (Q4_K+imatrix)", "419 MB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B" },
    { "qwen3-embed-0.6b-iq4xs", "qwen3-embed-0.6b-iq4_xs.gguf",
      "https://huggingface.co/cstr/qwen3-embed-0.6b-GGUF/resolve/main/qwen3-embed-0.6b-iq4_xs.gguf",
      "Qwen3 1024d official (IQ4_XS+imatrix)", "405 MB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B" },
    { "qwen3-embed-0.6b-q8", "qwen3-embed-0.6b-q8_0.gguf",
      "https://huggingface.co/cstr/qwen3-embed-0.6b-GGUF/resolve/main/qwen3-embed-0.6b-q8_0.gguf",
      "Qwen3 1024d official (Q8_0)", "639 MB", "apache-2.0", "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B" },

    { "qwen3-embed-4b", "qwen3-embed-4b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/qwen3-embed-4b-GGUF/resolve/main/qwen3-embed-4b-q4_k-imatrix.gguf",
      "Qwen3 2560d official (Q4_K+imatrix)", "2.5 GB", "apache-2.0", "https://huggingface.co/Qwen/Qwen3-Embedding-4B" },
    { "qwen3-embed-4b-q4k", "qwen3-embed-4b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/qwen3-embed-4b-GGUF/resolve/main/qwen3-embed-4b-q4_k-imatrix.gguf",
      "Qwen3 2560d official (Q4_K+imatrix)", "2.5 GB", "apache-2.0", "https://huggingface.co/Qwen/Qwen3-Embedding-4B" },
    { "qwen3-embed-4b-iq4xs", "qwen3-embed-4b-iq4_xs.gguf",
      "https://huggingface.co/cstr/qwen3-embed-4b-GGUF/resolve/main/qwen3-embed-4b-iq4_xs.gguf",
      "Qwen3 2560d official (IQ4_XS+imatrix)", "2.3 GB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-4B" },
    { "qwen3-embed-4b-q8", "qwen3-embed-4b-q8_0.gguf",
      "https://huggingface.co/cstr/qwen3-embed-4b-GGUF/resolve/main/qwen3-embed-4b-q8_0.gguf",
      "Qwen3 2560d official (Q8_0)", "4.3 GB", "apache-2.0", "https://huggingface.co/Qwen/Qwen3-Embedding-4B" },

    { "qwen3-embed-8b", "qwen3-embed-8b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/qwen3-embed-8b-GGUF/resolve/main/qwen3-embed-8b-q4_k-imatrix.gguf",
      "Qwen3 4096d official 8B (Q4_K+imatrix)", "4.6 GB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-8B" },
    { "qwen3-embed-8b-q4k", "qwen3-embed-8b-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/qwen3-embed-8b-GGUF/resolve/main/qwen3-embed-8b-q4_k-imatrix.gguf",
      "Qwen3 4096d official 8B (Q4_K+imatrix)", "4.6 GB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-8B" },
    { "qwen3-embed-8b-iq4xs", "qwen3-embed-8b-iq4_xs.gguf",
      "https://huggingface.co/cstr/qwen3-embed-8b-GGUF/resolve/main/qwen3-embed-8b-iq4_xs.gguf",
      "Qwen3 4096d official 8B (IQ4_XS+imatrix)", "4.4 GB", "apache-2.0",
      "https://huggingface.co/Qwen/Qwen3-Embedding-8B" },
    { "qwen3-embed-8b-q8", "qwen3-embed-8b-q8_0.gguf",
      "https://huggingface.co/cstr/qwen3-embed-8b-GGUF/resolve/main/qwen3-embed-8b-q8_0.gguf",
      "Qwen3 4096d official 8B (Q8_0)", "8.0 GB", "apache-2.0", "https://huggingface.co/Qwen/Qwen3-Embedding-8B" },

    // BidirLM-Omni — bidirectional Qwen3 (text) + Whisper-shape audio tower (cross-modal).
    // Two repos: -textonly is the smaller text-only variant; without suffix includes audio.
    { "bidirlm-omni-2.5b", "bidirlm-omni-2.5b-q8_0.gguf",
      "https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q8_0.gguf",
      "Qwen3-Bidirectional 2048d 90+langs text+audio (2.5B)", "3.1 GB", "apache-2.0",
      "https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding" },
    { "bidirlm-omni-2.5b-mm", "bidirlm-omni-2.5b-q4_k-imatrix-multimodal.gguf",
      "https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q4_k-imatrix-multimodal.gguf",
      "BidirLM-Omni 2.5B (Q4_K, MULTIMODAL imatrix: text +0.036 / image +0.007 / audio ~0 cos)", "1.6 GB", "apache-2.0",
      "https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding" },
    { "bidirlm-omni-2.5b-textonly", "bidirlm-omni-2.5b-textonly-q8_0.gguf",
      "https://huggingface.co/cstr/bidirlm-omni-2.5b-textonly-GGUF/resolve/main/bidirlm-omni-2.5b-textonly-q8_0.gguf",
      "Qwen3-Bidirectional 2048d text-only (2.5B)", "1834 MB", "apache-2.0",
      "https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding" },
    { "bidirlm-omni-2.5b-textonly-q4k", "bidirlm-omni-2.5b-textonly-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/bidirlm-omni-2.5b-textonly-GGUF/resolve/main/"
      "bidirlm-omni-2.5b-textonly-q4_k-imatrix.gguf",
      "BidirLM-Omni 2.5B text-only (Q4_K+imatrix — cos 0.948, size option; q8 for quality)", "1.1 GB", "apache-2.0",
      "https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding" },

    // --- RAG-critical models (Phase 3) ---

    { "bge-small-en-v1.5", "bge-small-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "25 MB", "mit", "https://huggingface.co/BAAI/bge-small-en-v1.5" },
    { "bge-small-en-v1.5-q4k", "bge-small-en-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5-q4_k-imatrix.gguf",
      "BERT 384d English (Q4_K+imatrix)", "25 MB", "mit", "https://huggingface.co/BAAI/bge-small-en-v1.5" },
    { "bge-small-en-v1.5-iq4xs", "bge-small-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "25 MB", "mit", "https://huggingface.co/BAAI/bge-small-en-v1.5" },
    { "bge-small-en-v1.5-q8", "bge-small-en-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf",
      "BERT 384d English (Q8_0)", "36 MB", "mit", "https://huggingface.co/BAAI/bge-small-en-v1.5" },

    { "bge-base-en-v1.5", "bge-base-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/bge-base-en-v1.5-GGUF/resolve/main/bge-base-en-v1.5-iq4_xs.gguf",
      "BERT 768d English (IQ4_XS+imatrix)", "72 MB", "mit", "https://huggingface.co/BAAI/bge-base-en-v1.5" },
    { "bge-base-en-v1.5-q4k", "bge-base-en-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/bge-base-en-v1.5-GGUF/resolve/main/bge-base-en-v1.5-q4_k-imatrix.gguf",
      "BERT 768d English (Q4_K+imatrix)", "74 MB", "mit", "https://huggingface.co/BAAI/bge-base-en-v1.5" },
    { "bge-base-en-v1.5-iq4xs", "bge-base-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/bge-base-en-v1.5-GGUF/resolve/main/bge-base-en-v1.5-iq4_xs.gguf",
      "BERT 768d English (IQ4_XS+imatrix)", "72 MB", "mit", "https://huggingface.co/BAAI/bge-base-en-v1.5" },
    { "bge-base-en-v1.5-q8", "bge-base-en-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/bge-base-en-v1.5-GGUF/resolve/main/bge-base-en-v1.5-q8_0.gguf",
      "BERT 768d English (Q8_0)", "117 MB", "mit", "https://huggingface.co/BAAI/bge-base-en-v1.5" },

    { "bge-large-en-v1.5", "bge-large-en-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/bge-large-en-v1.5-GGUF/resolve/main/bge-large-en-v1.5-q4_k-imatrix.gguf",
      "BERT 1024d English (Q4_K+imatrix)", "206 MB", "mit", "https://huggingface.co/BAAI/bge-large-en-v1.5" },
    { "bge-large-en-v1.5-q4k", "bge-large-en-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/bge-large-en-v1.5-GGUF/resolve/main/bge-large-en-v1.5-q4_k-imatrix.gguf",
      "BERT 1024d English (Q4_K+imatrix)", "206 MB", "mit", "https://huggingface.co/BAAI/bge-large-en-v1.5" },
    { "bge-large-en-v1.5-iq4xs", "bge-large-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/bge-large-en-v1.5-GGUF/resolve/main/bge-large-en-v1.5-iq4_xs.gguf",
      "BERT 1024d English (IQ4_XS+imatrix)", "196 MB", "mit", "https://huggingface.co/BAAI/bge-large-en-v1.5" },
    { "bge-large-en-v1.5-q8", "bge-large-en-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/bge-large-en-v1.5-GGUF/resolve/main/bge-large-en-v1.5-q8_0.gguf",
      "BERT 1024d English (Q8_0)", "358 MB", "mit", "https://huggingface.co/BAAI/bge-large-en-v1.5" },

    { "nomic-embed-text-v1.5", "nomic-embed-text-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5-q8_0.gguf",
      "BERT 768d 8K context Matryoshka (Q8_0)", "146 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5" },
    { "nomic-embed-text-v1.5-q4k", "nomic-embed-text-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5-q4_k-imatrix.gguf",
      "BERT 768d 8K context Matryoshka (Q4_K+imatrix)", "89 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5" },
    { "nomic-embed-text-v1.5-iq4xs", "nomic-embed-text-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5-iq4_xs.gguf",
      "BERT 768d 8K context Matryoshka (IQ4_XS+imatrix)", "86 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5" },
    { "nomic-embed-text-v1.5-q8", "nomic-embed-text-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5-q8_0.gguf",
      "BERT 768d 8K context Matryoshka (Q8_0)", "146 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5" },

    { "nomic-embed-text-v2-moe", "nomic-v2-moe-iq4_xs.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-v2-moe-iq4_xs.gguf",
      "NomicBERT MoE 768d 8-expert top-2 (IQ4_XS+imatrix)", "360 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe" },
    { "nomic-embed-text-v2-moe-q4k", "nomic-v2-moe-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-v2-moe-q4_k-imatrix.gguf",
      "NomicBERT MoE 768d 8-expert top-2 (Q4_K+imatrix)", "369 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe" },
    { "nomic-embed-text-v2-moe-iq4xs", "nomic-v2-moe-iq4_xs.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-v2-moe-iq4_xs.gguf",
      "NomicBERT MoE 768d 8-expert top-2 (IQ4_XS+imatrix)", "360 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe" },
    { "nomic-embed-text-v2-moe-q8", "nomic-v2-moe-q8_0.gguf",
      "https://huggingface.co/cstr/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-v2-moe-q8_0.gguf",
      "NomicBERT MoE 768d 8-expert top-2 (Q8_0)", "511 MB", "apache-2.0",
      "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe" },

    { "all-MiniLM-L12-v2", "all-MiniLM-L12-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "25 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2" },
    { "all-MiniLM-L12-v2-q4k", "all-MiniLM-L12-v2-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2-q4_k-imatrix.gguf",
      "BERT 384d English (Q4_K+imatrix)", "25 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2" },
    { "all-MiniLM-L12-v2-iq4xs", "all-MiniLM-L12-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2-iq4_xs.gguf",
      "BERT 384d English (IQ4_XS+imatrix)", "25 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2" },
    { "all-MiniLM-L12-v2-q8", "all-MiniLM-L12-v2-q8_0.gguf",
      "https://huggingface.co/cstr/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2-q8_0.gguf",
      "BERT 384d English (Q8_0)", "36 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2" },

    { "paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-multilingual-MiniLM-L12-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-GGUF/resolve/main/"
      "paraphrase-multilingual-MiniLM-L12-v2-iq4_xs.gguf",
      "BERT 384d 50+ langs (118M, SentencePiece, mean-pool) (IQ4_XS+imatrix)", "120 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" },
    { "paraphrase-multilingual-MiniLM-L12-v2-iq4xs", "paraphrase-multilingual-MiniLM-L12-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-GGUF/resolve/main/"
      "paraphrase-multilingual-MiniLM-L12-v2-iq4_xs.gguf",
      "BERT 384d 50+ langs (118M, SentencePiece, mean-pool) (IQ4_XS+imatrix)", "120 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" },
    { "paraphrase-multilingual-MiniLM-L12-v2-q4k", "paraphrase-multilingual-MiniLM-L12-v2-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-GGUF/resolve/main/"
      "paraphrase-multilingual-MiniLM-L12-v2-q4_k-imatrix.gguf",
      "BERT 384d 50+ langs (118M, SentencePiece, mean-pool) (Q4_K+imatrix)", "120 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" },
    { "paraphrase-multilingual-MiniLM-L12-v2-q8", "paraphrase-multilingual-MiniLM-L12-v2-q8_0.gguf",
      "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-GGUF/resolve/main/"
      "paraphrase-multilingual-MiniLM-L12-v2-q8_0.gguf",
      "BERT 384d 50+ langs (118M, SentencePiece, mean-pool) (Q8_0)", "131 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" },

    { "all-mpnet-base-v2", "all-mpnet-base-v2-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/all-mpnet-base-v2-GGUF/resolve/main/all-mpnet-base-v2-q4_k-imatrix.gguf",
      "BERT 768d English (Q4_K+imatrix)", "74 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-mpnet-base-v2" },
    { "all-mpnet-base-v2-q4k", "all-mpnet-base-v2-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/all-mpnet-base-v2-GGUF/resolve/main/all-mpnet-base-v2-q4_k-imatrix.gguf",
      "BERT 768d English (Q4_K+imatrix)", "74 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-mpnet-base-v2" },
    { "all-mpnet-base-v2-iq4xs", "all-mpnet-base-v2-iq4_xs.gguf",
      "https://huggingface.co/cstr/all-mpnet-base-v2-GGUF/resolve/main/all-mpnet-base-v2-iq4_xs.gguf",
      "BERT 768d English (IQ4_XS+imatrix)", "72 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-mpnet-base-v2" },
    { "all-mpnet-base-v2-q8", "all-mpnet-base-v2-q8_0.gguf",
      "https://huggingface.co/cstr/all-mpnet-base-v2-GGUF/resolve/main/all-mpnet-base-v2-q8_0.gguf",
      "BERT 768d English (Q8_0)", "117 MB", "apache-2.0",
      "https://huggingface.co/sentence-transformers/all-mpnet-base-v2" },

    { "mxbai-embed-large-v1", "mxbai-embed-large-v1-iq4_xs.gguf",
      "https://huggingface.co/cstr/mxbai-embed-large-v1-GGUF/resolve/main/mxbai-embed-large-v1-iq4_xs.gguf",
      "BERT 1024d English (IQ4_XS+imatrix)", "196 MB", "apache-2.0",
      "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1" },
    { "mxbai-embed-large-v1-q4k", "mxbai-embed-large-v1-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/mxbai-embed-large-v1-GGUF/resolve/main/mxbai-embed-large-v1-q4_k-imatrix.gguf",
      "BERT 1024d English (Q4_K+imatrix)", "206 MB", "apache-2.0",
      "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1" },
    { "mxbai-embed-large-v1-iq4xs", "mxbai-embed-large-v1-iq4_xs.gguf",
      "https://huggingface.co/cstr/mxbai-embed-large-v1-GGUF/resolve/main/mxbai-embed-large-v1-iq4_xs.gguf",
      "BERT 1024d English (IQ4_XS+imatrix)", "196 MB", "apache-2.0",
      "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1" },
    { "mxbai-embed-large-v1-q8", "mxbai-embed-large-v1-q8_0.gguf",
      "https://huggingface.co/cstr/mxbai-embed-large-v1-GGUF/resolve/main/mxbai-embed-large-v1-q8_0.gguf",
      "BERT 1024d English (Q8_0)", "357 MB", "apache-2.0",
      "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1" },

    { "snowflake-arctic-embed-m", "snowflake-arctic-embed-m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-m-GGUF/resolve/main/"
      "snowflake-arctic-embed-m-q4_k-imatrix.gguf",
      "BERT 768d CLS English (Q4_K+imatrix)", "74 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m" },
    { "snowflake-arctic-embed-m-q4k", "snowflake-arctic-embed-m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-m-GGUF/resolve/main/"
      "snowflake-arctic-embed-m-q4_k-imatrix.gguf",
      "BERT 768d CLS English (Q4_K+imatrix)", "74 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m" },
    { "snowflake-arctic-embed-m-iq4xs", "snowflake-arctic-embed-m-iq4_xs.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-m-GGUF/resolve/main/snowflake-arctic-embed-m-iq4_xs.gguf",
      "BERT 768d CLS English (IQ4_XS+imatrix)", "72 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m" },
    { "snowflake-arctic-embed-m-q8", "snowflake-arctic-embed-m-q8_0.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-m-GGUF/resolve/main/snowflake-arctic-embed-m-q8_0.gguf",
      "BERT 768d CLS English (Q8_0)", "117 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-m" },

    { "snowflake-arctic-embed-l", "snowflake-arctic-embed-l-iq4_xs.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-l-GGUF/resolve/main/snowflake-arctic-embed-l-iq4_xs.gguf",
      "XLM-R 1024d CLS English (IQ4_XS+imatrix)", "196 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l" },
    { "snowflake-arctic-embed-l-q4k", "snowflake-arctic-embed-l-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-l-GGUF/resolve/main/"
      "snowflake-arctic-embed-l-q4_k-imatrix.gguf",
      "XLM-R 1024d CLS English (Q4_K+imatrix)", "206 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l" },
    { "snowflake-arctic-embed-l-iq4xs", "snowflake-arctic-embed-l-iq4_xs.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-l-GGUF/resolve/main/snowflake-arctic-embed-l-iq4_xs.gguf",
      "XLM-R 1024d CLS English (IQ4_XS+imatrix)", "196 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l" },
    { "snowflake-arctic-embed-l-q8", "snowflake-arctic-embed-l-q8_0.gguf",
      "https://huggingface.co/cstr/snowflake-arctic-embed-l-GGUF/resolve/main/snowflake-arctic-embed-l-q8_0.gguf",
      "XLM-R 1024d CLS English (Q8_0)", "357 MB", "apache-2.0",
      "https://huggingface.co/Snowflake/snowflake-arctic-embed-l" },

    // Default = best flavor (IQ4_XS+imatrix, A/B winner: smaller AND higher cos
    // than Q4_K here). -q4k serves the Q4_K+imatrix build; -iq4xs/-q8 explicit.
    { "bge-m3", "bge-m3-iq4_xs.gguf", "https://huggingface.co/cstr/bge-m3-GGUF/resolve/main/bge-m3-iq4_xs.gguf",
      "XLM-R 1024d dense+sparse+ColBERT multilingual (568M, IQ4_XS+imatrix)", "449 MB", "mit",
      "https://huggingface.co/BAAI/bge-m3" },
    { "bge-m3-iq4xs", "bge-m3-iq4_xs.gguf", "https://huggingface.co/cstr/bge-m3-GGUF/resolve/main/bge-m3-iq4_xs.gguf",
      "XLM-R 1024d dense+sparse+ColBERT multilingual (568M, IQ4_XS+imatrix)", "449 MB", "mit",
      "https://huggingface.co/BAAI/bge-m3" },
    { "bge-m3-q4k", "bge-m3-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/bge-m3-GGUF/resolve/main/bge-m3-q4_k-imatrix.gguf",
      "XLM-R 1024d dense+sparse+ColBERT multilingual (568M, Q4_K+imatrix)", "459 MB", "mit",
      "https://huggingface.co/BAAI/bge-m3" },
    { "bge-m3-q8", "bge-m3-q8_0.gguf", "https://huggingface.co/cstr/bge-m3-GGUF/resolve/main/bge-m3-q8_0.gguf",
      "XLM-R 1024d dense+sparse+ColBERT multilingual (568M, Q8_0)", "610 MB", "mit",
      "https://huggingface.co/BAAI/bge-m3" },

    // --- Reranker models (Phase 4) ---

    // Reranker defaults: 4-bit imatrix only where Kendall-tau vs full precision
    // stays 1.0 (jina, ms-marco); the others keep q8_0 (tau<1.0 at 4-bit, and q8_0
    // is still smaller than the full-precision base) — see LEARNINGS.md → reranker A/B.
    { "bge-reranker-v2-m3", "bge-reranker-v2-m3-q8_0.gguf",
      "https://huggingface.co/cstr/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-q8_0.gguf",
      "XLM-R reranker multilingual 568M (Q8_0)", "613 MB", "apache-2.0",
      "https://huggingface.co/BAAI/bge-reranker-v2-m3" },
    // First sub-Q8 alias for this family, added with the -f7 imatrix re-pin
    // (F7b, 87e11a4e): tau vs f16 .920→.942 (CPU) / .920→.947 (Metal) and
    // |Δscore| −29/−33% over the leaf_N-defect imatrix quant; q8_0 stays default.
    { "bge-reranker-v2-m3-q4k", "bge-reranker-v2-m3-q4_k-imatrix-f7.gguf",
      "https://huggingface.co/cstr/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-q4_k-imatrix-f7.gguf",
      "XLM-R reranker multilingual 568M (Q4_K+imatrix, smaller)", "462 MB", "apache-2.0",
      "https://huggingface.co/BAAI/bge-reranker-v2-m3" },

    { "bge-reranker-base", "bge-reranker-base-q8_0.gguf",
      "https://huggingface.co/cstr/bge-reranker-base-GGUF/resolve/main/bge-reranker-base-q8_0.gguf",
      "BERT reranker EN+ZH 278M (Q8_0)", "304 MB", "mit", "https://huggingface.co/BAAI/bge-reranker-base" },

    // Reranker defaults are Q8_0: 4-bit keeps top-1 but reorders the tail (Kendall-τ
    // 0.92–0.96 vs f16 on a 16×6 corpus; imatrix doesn't help — the score head is
    // argmax-sensitive). The old iq4_xs/q4_k "τ=1.0" was a coarse-corpus artifact.
    { "ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-6-v2-q8_0-g7c.gguf",
      "https://huggingface.co/cstr/ms-marco-MiniLM-L-6-v2-GGUF/resolve/main/ms-marco-MiniLM-L-6-v2-q8_0-g7c.gguf",
      "BERT reranker English fast 22M (Q8_0; g7c: HF-exact tanh-pooler head)", "24 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2" },
    { "ms-marco-MiniLM-L-6-v2-iq4xs", "ms-marco-MiniLM-L-6-v2-iq4_xs-g7c.gguf",
      "https://huggingface.co/cstr/ms-marco-MiniLM-L-6-v2-GGUF/resolve/main/ms-marco-MiniLM-L-6-v2-iq4_xs-g7c.gguf",
      "BERT reranker English fast 22M (IQ4_XS+imatrix, smaller)", "19 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2" },

    { "ms-marco-MiniLM-L-12-v2", "ms-marco-MiniLM-L-12-v2-q8_0-g7c.gguf",
      "https://huggingface.co/cstr/ms-marco-MiniLM-L-12-v2-GGUF/resolve/main/ms-marco-MiniLM-L-12-v2-q8_0-g7c.gguf",
      "BERT reranker English 33M (Q8_0; g7c: HF-exact tanh-pooler head)", "36 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2" },
    { "ms-marco-MiniLM-L-12-v2-iq4xs", "ms-marco-MiniLM-L-12-v2-iq4_xs-g7c.gguf",
      "https://huggingface.co/cstr/ms-marco-MiniLM-L-12-v2-GGUF/resolve/main/ms-marco-MiniLM-L-12-v2-iq4_xs-g7c.gguf",
      "BERT reranker English 33M (IQ4_XS+imatrix, smaller)", "25 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2" },

    { "jina-reranker-v2-base-multilingual", "jina-reranker-v2-base-multilingual-q8_0.gguf",
      "https://huggingface.co/cstr/jina-reranker-v2-base-multilingual-GGUF/resolve/main/"
      "jina-reranker-v2-base-multilingual-q8_0.gguf",
      "XLM-R reranker multilingual 278M (Q8_0 — exact ranking; q4_k+im τ≈0.93)", "302 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual" },
    // -f7 imatrix (F7b re-collection, 87e11a4e): attn q/k/v now covered; local
    // Metal+CPU cross-check reproduced the Kaggle dscore to 4dp — |Δscore| vs
    // f16 drops ~25% on both backends, tau within the backend near-tie band.
    { "jina-reranker-v2-base-multilingual-q4k", "jina-reranker-v2-base-multilingual-q4_k-imatrix-f7.gguf",
      "https://huggingface.co/cstr/jina-reranker-v2-base-multilingual-GGUF/resolve/main/"
      "jina-reranker-v2-base-multilingual-q4_k-imatrix-f7.gguf",
      "XLM-R reranker multilingual 278M (Q4_K+imatrix, smaller)", "261 MB", "cc-by-nc-4.0",
      "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual" },

    { "mxbai-rerank-xsmall-v1", "mxbai-rerank-xsmall-v1-q8_0-g7c.gguf",
      "https://huggingface.co/cstr/mxbai-rerank-xsmall-v1-GGUF/resolve/main/mxbai-rerank-xsmall-v1-q8_0-g7c.gguf",
      "DeBERTa-v2 reranker English fast 33M (Q8_0)", "78 MB", "apache-2.0",
      "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1" },

    { "mxbai-rerank-base-v1", "mxbai-rerank-base-v1-q8_0-g7c.gguf",
      "https://huggingface.co/cstr/mxbai-rerank-base-v1-GGUF/resolve/main/mxbai-rerank-base-v1-q8_0-g7c.gguf",
      "DeBERTa-v2 reranker English 86M (Q8_0)", "199 MB", "apache-2.0",
      "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1" },

    { "ettin-reranker-17m", "ettin-reranker-17m-v1-q8_0.gguf",
      "https://huggingface.co/cstr/ettin-reranker-17m-v1-GGUF/resolve/main/ettin-reranker-17m-v1-q8_0.gguf",
      "ModernBERT reranker 17M (Q8_0, fastest)", "19 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-17m-v1" },
    { "ettin-reranker-32m", "ettin-reranker-32m-v1-q8_0.gguf",
      "https://huggingface.co/cstr/ettin-reranker-32m-v1-GGUF/resolve/main/ettin-reranker-32m-v1-q8_0.gguf",
      "ModernBERT reranker 32M (Q8_0)", "34 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-32m-v1" },
    { "ettin-reranker-68m", "ettin-reranker-68m-v1-q8_0.gguf",
      "https://huggingface.co/cstr/ettin-reranker-68m-v1-GGUF/resolve/main/ettin-reranker-68m-v1-q8_0.gguf",
      "ModernBERT reranker 68M (Q8_0)", "71 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-68m-v1" },
    { "ettin-reranker-150m", "ettin-reranker-150m-v1-q8_0.gguf",
      "https://huggingface.co/cstr/ettin-reranker-150m-v1-GGUF/resolve/main/ettin-reranker-150m-v1-q8_0.gguf",
      "ModernBERT reranker 150M (Q8_0)", "155 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-150m-v1" },
    { "ettin-reranker-150m-q6k", "ettin-reranker-150m-v1-q6_k.gguf",
      "https://huggingface.co/cstr/ettin-reranker-150m-v1-GGUF/resolve/main/ettin-reranker-150m-v1-q6_k.gguf",
      "ModernBERT reranker 150M (Q6_K)", "133 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-150m-v1" },
    { "ettin-reranker-150m-q4k", "ettin-reranker-150m-v1-q4_k.gguf",
      "https://huggingface.co/cstr/ettin-reranker-150m-v1-GGUF/resolve/main/ettin-reranker-150m-v1-q4_k.gguf",
      "ModernBERT reranker 150M (Q4_K, smaller)", "102 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-150m-v1" },
    { "ettin-reranker-400m", "ettin-reranker-400m-v1-q8_0.gguf",
      "https://huggingface.co/cstr/ettin-reranker-400m-v1-GGUF/resolve/main/ettin-reranker-400m-v1-q8_0.gguf",
      "ModernBERT reranker 400M (Q8_0)", "405 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-400m-v1" },
    { "ettin-reranker-400m-q6k", "ettin-reranker-400m-v1-q6_k.gguf",
      "https://huggingface.co/cstr/ettin-reranker-400m-v1-GGUF/resolve/main/ettin-reranker-400m-v1-q6_k.gguf",
      "ModernBERT reranker 400M (Q6_K)", "343 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-400m-v1" },
    { "ettin-reranker-400m-q4k", "ettin-reranker-400m-v1-q4_k.gguf",
      "https://huggingface.co/cstr/ettin-reranker-400m-v1-GGUF/resolve/main/ettin-reranker-400m-v1-q4_k.gguf",
      "ModernBERT reranker 400M (Q4_K, smaller)", "241 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-400m-v1" },
    { "ettin-reranker-1b", "ettin-reranker-1b-v1-q8_0.gguf",
      "https://huggingface.co/cstr/ettin-reranker-1b-v1-GGUF/resolve/main/ettin-reranker-1b-v1-q8_0.gguf",
      "ModernBERT reranker 1B (Q8_0)", "1055 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-1b-v1" },
    { "ettin-reranker-1b-q6k", "ettin-reranker-1b-v1-q6_k.gguf",
      "https://huggingface.co/cstr/ettin-reranker-1b-v1-GGUF/resolve/main/ettin-reranker-1b-v1-q6_k.gguf",
      "ModernBERT reranker 1B (Q6_K)", "838 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-1b-v1" },
    { "ettin-reranker-1b-q4k", "ettin-reranker-1b-v1-q4_k.gguf",
      "https://huggingface.co/cstr/ettin-reranker-1b-v1-GGUF/resolve/main/ettin-reranker-1b-v1-q4_k.gguf",
      "ModernBERT reranker 1B (Q4_K, smaller)", "608 MB", "apache-2.0",
      "https://huggingface.co/cross-encoder/ettin-reranker-1b-v1" },

    // --- MTEB top multilingual models ---

    { "multilingual-e5-base", "multilingual-e5-base-iq4_xs.gguf",
      "https://huggingface.co/cstr/multilingual-e5-base-GGUF/resolve/main/multilingual-e5-base-iq4_xs.gguf",
      "XLM-R 768d 100+ languages (IQ4_XS+imatrix)", "257 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-base" },
    { "multilingual-e5-base-q4k", "multilingual-e5-base-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/multilingual-e5-base-GGUF/resolve/main/multilingual-e5-base-q4_k-imatrix.gguf",
      "XLM-R 768d 100+ languages (Q4_K+imatrix)", "259 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-base" },
    { "multilingual-e5-base-iq4xs", "multilingual-e5-base-iq4_xs.gguf",
      "https://huggingface.co/cstr/multilingual-e5-base-GGUF/resolve/main/multilingual-e5-base-iq4_xs.gguf",
      "XLM-R 768d 100+ languages (IQ4_XS+imatrix)", "257 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-base" },
    { "multilingual-e5-base-q8", "multilingual-e5-base-q8_0.gguf",
      "https://huggingface.co/cstr/multilingual-e5-base-GGUF/resolve/main/multilingual-e5-base-q8_0.gguf",
      "XLM-R 768d 100+ languages (Q8_0)", "302 MB", "mit", "https://huggingface.co/intfloat/multilingual-e5-base" },

    // Default = best flavor (IQ4_XS+imatrix, A/B winner) — was defaulting to the
    // 2.2 GB F32 (now the -f32 variant). -q4k/-iq4xs/-q8 select flavors.
    { "multilingual-e5-large", "multilingual-e5-large-iq4_xs.gguf",
      "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large-iq4_xs.gguf",
      "XLM-R 1024d 100+ languages (560M, IQ4_XS+imatrix)", "441 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-large" },
    { "multilingual-e5-large-iq4xs", "multilingual-e5-large-iq4_xs.gguf",
      "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large-iq4_xs.gguf",
      "XLM-R 1024d 100+ languages (560M, IQ4_XS+imatrix)", "441 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-large" },
    { "multilingual-e5-large-q4k", "multilingual-e5-large-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large-q4_k-imatrix.gguf",
      "XLM-R 1024d 100+ languages (560M, Q4_K+imatrix)", "450 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-large" },
    { "multilingual-e5-large-q8", "multilingual-e5-large-q8_0.gguf",
      "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large-q8_0.gguf",
      "XLM-R 1024d 100+ languages (560M, Q8_0)", "601 MB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-large" },
    { "multilingual-e5-large-f32", "multilingual-e5-large.gguf",
      "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large.gguf",
      "XLM-R 1024d 100+ languages (560M, F32)", "2.2 GB", "mit",
      "https://huggingface.co/intfloat/multilingual-e5-large" },

    { "granite-embedding-278m", "granite-embedding-278m-multilingual-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/granite-embedding-278m-multilingual-GGUF/resolve/main/"
      "granite-embedding-278m-multilingual-q4_k-imatrix.gguf",
      "XLM-R 768d IBM multilingual 278M (Q4_K+imatrix)", "259 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual" },

    { "granite-embedding-107m", "granite-embedding-107m-multilingual-iq4_xs.gguf",
      "https://huggingface.co/cstr/granite-embedding-107m-multilingual-GGUF/resolve/main/"
      "granite-embedding-107m-multilingual-iq4_xs.gguf",
      "XLM-R 384d IBM multilingual 107M (IQ4_XS+imatrix)", "115 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual" },

    // granite-embedding-r2: ModernBERT backbone (RoPE, alternating local/global
    // attention, 8192 ctx), CLS pooling, no query/document prefix. The 97m uses
    // the o200k BPE split, the 311m a SentencePiece BPE — see
    // tokenizer.ggml.pre / tokenizer.ggml.is_spm_bpe. Q8_0 is the default
    // because no imatrix has been calibrated for either yet.
    { "granite-embedding-97m-r2", "granite-embedding-97m-multilingual-r2-q8_0.gguf",
      "https://huggingface.co/cstr/granite-embedding-97m-multilingual-r2-GGUF/resolve/main/"
      "granite-embedding-97m-multilingual-r2-q8_0.gguf",
      "ModernBERT 384d IBM multilingual r2 97M, 8k ctx (Q8_0, cos 0.9996)", "106 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2" },

    { "granite-embedding-97m-r2-q8", "granite-embedding-97m-multilingual-r2-q8_0.gguf",
      "https://huggingface.co/cstr/granite-embedding-97m-multilingual-r2-GGUF/resolve/main/"
      "granite-embedding-97m-multilingual-r2-q8_0.gguf",
      "ModernBERT 384d IBM multilingual r2 97M, 8k ctx (Q8_0)", "106 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2" },

    { "granite-embedding-97m-r2-f16", "granite-embedding-97m-multilingual-r2-f16.gguf",
      "https://huggingface.co/cstr/granite-embedding-97m-multilingual-r2-GGUF/resolve/main/"
      "granite-embedding-97m-multilingual-r2-f16.gguf",
      "ModernBERT 384d IBM multilingual r2 97M, 8k ctx (F16, cos 1.0000)", "362 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2" },

    { "granite-embedding-311m-r2", "granite-embedding-311m-multilingual-r2-q8_0.gguf",
      "https://huggingface.co/cstr/granite-embedding-311m-multilingual-r2-GGUF/resolve/main/"
      "granite-embedding-311m-multilingual-r2-q8_0.gguf",
      "ModernBERT 768d IBM multilingual r2 311M, 8k ctx (Q8_0, cos 0.9998)", "331 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2" },

    { "granite-embedding-311m-r2-q8", "granite-embedding-311m-multilingual-r2-q8_0.gguf",
      "https://huggingface.co/cstr/granite-embedding-311m-multilingual-r2-GGUF/resolve/main/"
      "granite-embedding-311m-multilingual-r2-q8_0.gguf",
      "ModernBERT 768d IBM multilingual r2 311M, 8k ctx (Q8_0)", "331 MB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2" },

    { "granite-embedding-311m-r2-f16", "granite-embedding-311m-multilingual-r2-f16.gguf",
      "https://huggingface.co/cstr/granite-embedding-311m-multilingual-r2-GGUF/resolve/main/"
      "granite-embedding-311m-multilingual-r2-f16.gguf",
      "ModernBERT 768d IBM multilingual r2 311M, 8k ctx (F16, cos 1.0000)", "1.1 GB", "apache-2.0",
      "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2" },

    // --- Sparse models ---

    { "splade-pp-en-v1", "splade-pp-en-v1-iq4_xs.gguf",
      "https://huggingface.co/cstr/splade-pp-en-v1-GGUF/resolve/main/splade-pp-en-v1-iq4_xs.gguf",
      "BERT sparse SPLADE English 109M (IQ4_XS+imatrix, sparse-cos 0.996)", "72 MB", "apache-2.0",
      "https://huggingface.co/prithivida/Splade_PP_en_v1" },
    { "splade-v3", "splade-v3-iq4_xs.gguf",
      "https://huggingface.co/cstr/splade-v3-GGUF/resolve/main/splade-v3-iq4_xs.gguf",
      "BERT sparse SPLADE v3 English 110M (IQ4_XS+imatrix, sparse-cos 0.997)", "68 MB", "cc-by-nc-sa-4.0",
      "https://huggingface.co/naver/splade-v3" },
    { "splade-v3-q8", "splade-v3-q8_0.gguf",
      "https://huggingface.co/cstr/splade-v3-GGUF/resolve/main/splade-v3-q8_0.gguf",
      "BERT sparse SPLADE v3 English 110M (Q8_0, sparse-cos 1.000)", "111 MB", "cc-by-nc-sa-4.0",
      "https://huggingface.co/naver/splade-v3" },

    // --- GTE v1.5 (new BERT) ---

    { "gte-base-en-v1.5", "gte-base-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-base-en-v1.5-GGUF/resolve/main/gte-base-en-v1.5-iq4_xs.gguf",
      "GTE 768d English pre-LN+RoPE+GeGLU (IQ4_XS+imatrix)", "86 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5" },
    { "gte-base-en-v1.5-q4k", "gte-base-en-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/gte-base-en-v1.5-GGUF/resolve/main/gte-base-en-v1.5-q4_k-imatrix.gguf",
      "GTE 768d English pre-LN+RoPE+GeGLU (Q4_K+imatrix)", "90 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5" },
    { "gte-base-en-v1.5-iq4xs", "gte-base-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-base-en-v1.5-GGUF/resolve/main/gte-base-en-v1.5-iq4_xs.gguf",
      "GTE 768d English pre-LN+RoPE+GeGLU (IQ4_XS+imatrix)", "86 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5" },
    { "gte-base-en-v1.5-q8", "gte-base-en-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/gte-base-en-v1.5-GGUF/resolve/main/gte-base-en-v1.5-q8_0.gguf",
      "GTE 768d English pre-LN+RoPE+GeGLU (Q8_0)", "146 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5" },

    { "gte-large-en-v1.5", "gte-large-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-large-en-v1.5-GGUF/resolve/main/gte-large-en-v1.5-iq4_xs.gguf",
      "GTE 1024d English pre-LN+RoPE+GeGLU (IQ4_XS+imatrix)", "249 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5" },
    { "gte-large-en-v1.5-q4k", "gte-large-en-v1.5-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/gte-large-en-v1.5-GGUF/resolve/main/gte-large-en-v1.5-q4_k-imatrix.gguf",
      "GTE 1024d English pre-LN+RoPE+GeGLU (Q4_K+imatrix)", "261 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5" },
    { "gte-large-en-v1.5-iq4xs", "gte-large-en-v1.5-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-large-en-v1.5-GGUF/resolve/main/gte-large-en-v1.5-iq4_xs.gguf",
      "GTE 1024d English pre-LN+RoPE+GeGLU (IQ4_XS+imatrix)", "249 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5" },
    { "gte-large-en-v1.5-q8", "gte-large-en-v1.5-q8_0.gguf",
      "https://huggingface.co/cstr/gte-large-en-v1.5-GGUF/resolve/main/gte-large-en-v1.5-q8_0.gguf",
      "GTE 1024d English pre-LN+RoPE+GeGLU (Q8_0)", "463 MB", "apache-2.0",
      "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5" },

    { "gte-modernbert-base", "gte-modernbert-base-iq4_xs.gguf",
      "https://huggingface.co/cstr/gte-modernbert-base-GGUF/resolve/main/gte-modernbert-base-iq4_xs.gguf",
      "ModernBERT 768d English CLS, alternating global/local SWA + per-layer RoPE 149M (IQ4_XS+imatrix)", "102 MB",
      "apache-2.0", "https://huggingface.co/Alibaba-NLP/gte-modernbert-base" },

    { "embeddinggemma-300m", "embeddinggemma-300m-iq4_xs.gguf",
      "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-iq4_xs.gguf",
      "Gemma3 768d 24-layer mean-pool (IQ4_XS+imatrix)", "303 MB", "gemma",
      "https://huggingface.co/google/embeddinggemma-300m" },
    { "embeddinggemma-300m-q4k", "embeddinggemma-300m-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-q4_k-imatrix.gguf",
      "Gemma3 768d 24-layer mean-pool (Q4_K+imatrix)", "306 MB", "gemma",
      "https://huggingface.co/google/embeddinggemma-300m" },
    { "embeddinggemma-300m-iq4xs", "embeddinggemma-300m-iq4_xs.gguf",
      "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-iq4_xs.gguf",
      "Gemma3 768d 24-layer mean-pool (IQ4_XS+imatrix)", "303 MB", "gemma",
      "https://huggingface.co/google/embeddinggemma-300m" },
    { "embeddinggemma-300m-q8", "embeddinggemma-300m-q8_0.gguf",
      "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-q8_0.gguf",
      "Gemma3 768d 24-layer mean-pool (Q8_0)", "357 MB", "gemma", "https://huggingface.co/google/embeddinggemma-300m" },
    { "embeddinggemma-300m-qat", "embeddinggemma-300m-qat-q8_0-dense.gguf",
      "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-qat-q8_0-dense.gguf",
      "Gemma3 768d 24-layer mean-pool, community gemma-embedding arch (QAT Q8_0 + baked Dense)", "347 MB", "gemma",
      "https://huggingface.co/google/embeddinggemma-300m" },

    { "yunet", "yunet.gguf", "https://huggingface.co/cstr/yunet-GGUF/resolve/main/yunet.gguf",
      "YuNet face detection (ShuffleNetV2 640x640, 75K)", "0.2 MB", "apache-2.0",
      "https://huggingface.co/cstr/yunet-GGUF" },

    { "clip-vit-base-patch16", "clip-vit-base-patch16.gguf",
      "https://huggingface.co/cstr/clip-vit-base-patch16-GGUF/resolve/main/clip-vit-base-patch16.gguf",
      "CLIP ViT-B/16 vision encoder (86M)", "329 MB", "mit", "https://huggingface.co/openai/clip-vit-base-patch16" },

    { "clip-vit-large-patch14", "clip-vit-large-patch14.gguf",
      "https://huggingface.co/cstr/clip-vit-large-patch14-GGUF/resolve/main/clip-vit-large-patch14.gguf",
      "CLIP ViT-L/14 vision encoder (304M)", "1.2 GB", "mit", "https://huggingface.co/openai/clip-vit-large-patch14" },

    { "clip-text-base", "clip-text-base-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/clip-text-base-GGUF/resolve/main/clip-text-base-q4_k-imatrix.gguf",
      "CLIP text encoder base (63M, 512d, Q4_K+imatrix)", "50 MB", "mit",
      "https://huggingface.co/openai/clip-vit-base-patch16" },
    { "clip-text-base-iq4xs", "clip-text-base-iq4_xs.gguf",
      "https://huggingface.co/cstr/clip-text-base-GGUF/resolve/main/clip-text-base-iq4_xs.gguf",
      "CLIP text encoder base (IQ4_XS+imatrix)", "49 MB", "mit",
      "https://huggingface.co/openai/clip-vit-base-patch16" },
    { "clip-text-base-q8", "clip-text-base-q8_0.gguf",
      "https://huggingface.co/cstr/clip-text-base-GGUF/resolve/main/clip-text-base-q8_0.gguf",
      "CLIP text encoder base (Q8_0)", "70 MB", "mit", "https://huggingface.co/openai/clip-vit-base-patch16" },
    { "clip-text-base-f32", "clip-text-base.gguf",
      "https://huggingface.co/cstr/clip-text-base-GGUF/resolve/main/clip-text-base.gguf",
      "CLIP text encoder base (F32)", "244 MB", "mit", "https://huggingface.co/openai/clip-vit-base-patch16" },

    { "clip-text-large", "clip-text-large-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/clip-text-large-GGUF/resolve/main/clip-text-large-q4_k-imatrix.gguf",
      "CLIP text encoder large (124M, 768d, Q4_K+imatrix)", "91 MB", "mit",
      "https://huggingface.co/openai/clip-vit-large-patch14" },
    { "clip-text-large-iq4xs", "clip-text-large-iq4_xs.gguf",
      "https://huggingface.co/cstr/clip-text-large-GGUF/resolve/main/clip-text-large-iq4_xs.gguf",
      "CLIP text encoder large (IQ4_XS+imatrix)", "88 MB", "mit",
      "https://huggingface.co/openai/clip-vit-large-patch14" },
    { "clip-text-large-q8", "clip-text-large-q8_0.gguf",
      "https://huggingface.co/cstr/clip-text-large-GGUF/resolve/main/clip-text-large-q8_0.gguf",
      "CLIP text encoder large (Q8_0)", "134 MB", "mit", "https://huggingface.co/openai/clip-vit-large-patch14" },
    { "clip-text-large-f32", "clip-text-large.gguf",
      "https://huggingface.co/cstr/clip-text-large-GGUF/resolve/main/clip-text-large.gguf",
      "CLIP text encoder large (F32)", "480 MB", "mit", "https://huggingface.co/openai/clip-vit-large-patch14" },

    { "siglip-large-256", "siglip-large-256.gguf",
      "https://huggingface.co/cstr/siglip-large-256-GGUF/resolve/main/siglip-large-256.gguf",
      "SigLIP ViT-L/16 vision encoder 256x256 (304M)", "1.2 GB", "apache-2.0",
      "https://huggingface.co/google/siglip-large-patch16-256" },

    { "siglip-so400m-patch14-384", "siglip-so400m-patch14-384.gguf",
      "https://huggingface.co/cstr/siglip-so400m-patch14-384-GGUF/resolve/main/siglip-so400m-patch14-384.gguf",
      "SigLIP SoViT-400M/14 vision encoder 384x384 (428M)", "1.6 GB", "apache-2.0",
      "https://huggingface.co/google/siglip-so400m-patch14-384" },

    { "clip-vit-large-patch14-336", "clip-vit-large-patch14-336.gguf",
      "https://huggingface.co/cstr/clip-vit-large-patch14-336-GGUF/resolve/main/clip-vit-large-patch14-336.gguf",
      "CLIP ViT-L/14@336px vision encoder (304M)", "1.2 GB", "mit",
      "https://huggingface.co/openai/clip-vit-large-patch14-336" },

    { "siglip-base", "siglip-base.gguf", "https://huggingface.co/cstr/siglip-base-GGUF/resolve/main/siglip-base.gguf",
      "SigLIP ViT-B/16 vision encoder 384x384 (93M)", "354 MB", "apache-2.0",
      "https://huggingface.co/google/siglip-base-patch16-384" },

    { "siglip-text-base", "siglip-text-base-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/siglip-text-base-GGUF/resolve/main/siglip-text-base-q4_k-imatrix.gguf",
      "SigLIP text encoder base (93M, 768d, Q4_K+imatrix; cos 0.962 — q8 for max fidelity)", "75 MB", "apache-2.0",
      "https://huggingface.co/google/siglip-base-patch16-224" },
    { "siglip-text-base-iq4xs", "siglip-text-base-iq4_xs.gguf",
      "https://huggingface.co/cstr/siglip-text-base-GGUF/resolve/main/siglip-text-base-iq4_xs.gguf",
      "SigLIP text encoder base (IQ4_XS+imatrix)", "73 MB", "apache-2.0",
      "https://huggingface.co/google/siglip-base-patch16-224" },
    { "siglip-text-base-q8", "siglip-text-base-q8_0.gguf",
      "https://huggingface.co/cstr/siglip-text-base-GGUF/resolve/main/siglip-text-base-q8_0.gguf",
      "SigLIP text encoder base (Q8_0, cos 0.9995)", "118 MB", "apache-2.0",
      "https://huggingface.co/google/siglip-base-patch16-224" },
    { "siglip-text-base-f32", "siglip-text-base.gguf",
      "https://huggingface.co/cstr/siglip-text-base-GGUF/resolve/main/siglip-text-base.gguf",
      "SigLIP text encoder base (F32)", "421 MB", "apache-2.0",
      "https://huggingface.co/google/siglip-base-patch16-224" },

    { "scrfd-det-10g", "scrfd-det-10g.gguf",
      "https://huggingface.co/cstr/scrfd-det-10g-GGUF/resolve/main/scrfd-det-10g.gguf",
      "SCRFD face detection (ResNet-50 FPN, 640x640)", "16 MB", "apache-2.0",
      "https://huggingface.co/cstr/scrfd-det-10g-GGUF" },

    { "auraface-v1", "auraface-v1.gguf", "https://huggingface.co/cstr/auraface-v1-GGUF/resolve/main/auraface-v1.gguf",
      "AuraFace face recognition (ResNet-100, 512d)", "249 MB", "apache-2.0",
      "https://huggingface.co/cstr/auraface-v1-GGUF" },

    { "sface", "sface.gguf", "https://huggingface.co/cstr/sface-GGUF/resolve/main/sface.gguf",
      "SFace face recognition (MobileFaceNet, 128d)", "37 MB", "apache-2.0", "https://huggingface.co/cstr/sface-GGUF" },

    { "hmer-hw", "hmer-hw-q4_k.gguf",
      "https://huggingface.co/cstr/hmer-handwritten-math-gguf/resolve/main/hmer-hw-q4_k.gguf",
      "HMER handwritten math OCR (DenseNet-121+GRU, 112 tokens)", "5 MB", "mit",
      "https://huggingface.co/cstr/hmer-handwritten-math-gguf" },

    { "bttr-hw", "bttr-hw-q4_k.gguf",
      "https://huggingface.co/cstr/bttr-handwritten-math-gguf/resolve/main/bttr-hw-q4_k.gguf",
      "BTTR handwritten math OCR (DenseNet+Transformer, 113 tokens)", "11 MB", "mit",
      "https://huggingface.co/cstr/bttr-handwritten-math-gguf" },

    { "ppformulanet-l", "ppformulanet-l-q8_0.gguf",
      "https://huggingface.co/cstr/ppformulanet-l-gguf/resolve/main/ppformulanet-l-q8_0.gguf",
      "PP-FormulaNet-L printed math OCR (SAM-ViT+MBart, 181M)", "252 MB", "apache-2.0",
      "https://huggingface.co/cstr/ppformulanet-l-gguf" },

    { "posformer-crohme", "posformer-crohme-q8_0.gguf",
      "https://huggingface.co/cstr/posformer-crohme-GGUF/resolve/main/posformer-crohme-q8_0.gguf",
      "PosFormer handwritten math OCR (DenseNet+Transformer+ARM, 57% CROHME)", "12 MB", "cc-by-nc-sa-3.0",
      "https://huggingface.co/cstr/posformer-crohme-GGUF" },

    { "pix2tex-mfr", "pix2tex-mfr-q4_k.gguf",
      "https://huggingface.co/cstr/pix2tex-mfr-gguf/resolve/main/pix2tex-mfr-q4_k.gguf",
      "pix2tex printed math OCR (DeiT+TrOCR, 28M)", "17 MB", "mit", "https://huggingface.co/cstr/pix2tex-mfr-gguf" },

    { "texo-distill", "texo-distill-q8_0.gguf",
      "https://huggingface.co/cstr/texo-distill-gguf/resolve/main/texo-distill-q8_0.gguf",
      "Texo-Distill printed math OCR (HGNetv2+MBart, 20M, BLEU 0.90)", "22 MB", "agpl-3.0",
      "https://huggingface.co/cstr/texo-distill-gguf" },

    { "parseq", "parseq-q8_0.gguf", "https://huggingface.co/cstr/parseq-GGUF/resolve/main/parseq-q8_0.gguf",
      "PARSeq scene text OCR (ViT+Transformer, 24M, ECCV 2022)", "24 MB", "apache-2.0",
      "https://huggingface.co/cstr/parseq-GGUF" },

    { "parseq-tiny", "parseq-tiny-q8_0.gguf",
      "https://huggingface.co/cstr/parseq-GGUF/resolve/main/parseq-tiny-q8_0.gguf",
      "PARSeq-tiny scene text OCR (ViT+Transformer, 6M, ECCV 2022)", "6 MB", "apache-2.0",
      "https://huggingface.co/cstr/parseq-GGUF" },

    { "dbnet-det", "dbnet-ic15-q8_0.gguf",
      "https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q8_0.gguf",
      "DBNet text detection (ResNet-18+FPNC, ICDAR2015)", "13 MB", "apache-2.0",
      "https://huggingface.co/cstr/dbnet-ic15-GGUF" },

    { "ppocrv6-tiny-det", "PP-OCRv6_tiny_det-f16.gguf",
      "https://huggingface.co/cstr/PP-OCRv6-tiny-det-GGUF/resolve/main/PP-OCRv6_tiny_det-f16.gguf",
      "PP-OCRv6 tiny text detector (DB++ neck)", "1 MB", "apache-2.0",
      "https://huggingface.co/cstr/PP-OCRv6-tiny-det-GGUF" },
    { "ppocrv6-small-det", "PP-OCRv6_small_det-f16.gguf",
      "https://huggingface.co/cstr/PP-OCRv6-small-det-GGUF/resolve/main/PP-OCRv6_small_det-f16.gguf",
      "PP-OCRv6 small text detector (DB++ neck)", "5 MB", "apache-2.0",
      "https://huggingface.co/cstr/PP-OCRv6-small-det-GGUF" },
    { "ppocrv6-medium-det", "PP-OCRv6_medium_det-f16.gguf",
      "https://huggingface.co/cstr/PP-OCRv6-medium-det-GGUF/resolve/main/PP-OCRv6_medium_det-f16.gguf",
      "PP-OCRv6 medium text detector (DB++ neck)", "42 MB", "apache-2.0",
      "https://huggingface.co/cstr/PP-OCRv6-medium-det-GGUF" },
    { "ppocrv6-tiny-rec", "PP-OCRv6_tiny_rec-q8-head.gguf",
      "https://huggingface.co/cstr/PP-OCRv6_tiny_rec-GGUF/resolve/main/PP-OCRv6_tiny_rec-q8-head.gguf",
      "PP-OCRv6 tiny CTC recognizer (F32 backbone, head-only Q8)", "5 MB", "apache-2.0",
      "https://huggingface.co/cstr/PP-OCRv6_tiny_rec-GGUF", "latin+cjk+greek" },
    { "ppocrv6-small-rec", "PP-OCRv6_small_rec-q8-head.gguf",
      "https://huggingface.co/cstr/PP-OCRv6_small_rec-GGUF/resolve/main/PP-OCRv6_small_rec-q8-head.gguf",
      "PP-OCRv6 small CTC recognizer (F32 backbone, head-only Q8)", "20 MB", "apache-2.0",
      "https://huggingface.co/cstr/PP-OCRv6_small_rec-GGUF", "latin+cjk+kana+greek" },
    { "ppocrv6-medium-rec", "PP-OCRv6_medium_rec-q8-head.gguf",
      "https://huggingface.co/cstr/PP-OCRv6_medium_rec-GGUF/resolve/main/PP-OCRv6_medium_rec-q8-head.gguf",
      "PP-OCRv6 medium CTC recognizer (F32 backbone, head-only Q8)", "63 MB", "apache-2.0",
      "https://huggingface.co/cstr/PP-OCRv6_medium_rec-GGUF", "latin+cjk+kana+greek" },

    // EasyOCR CRNN recognizers and the PP-LCNet line-orientation classifier are
    // produced locally by models/convert-easyocr-to-gguf.py and
    // models/convert-pplcnet-orientation-to-gguf.py; no GGUF is published yet,
    // so these entries name the artifact for --ocr-rec/--ocr-cls resolution
    // without promising a download (same pattern as mixtex-zhen).
    { "easyocr-english-g2", "easyocr-english-g2-f16.gguf", "",
      "EasyOCR English Gen2 CRNN recognizer (local conversion)", "16 MB", "apache-2.0",
      "https://github.com/JaidedAI/EasyOCR" },
    { "easyocr-latin-g2", "easyocr-latin-g2-f16.gguf", "", "EasyOCR Latin Gen2 CRNN recognizer (local conversion)",
      "16 MB", "apache-2.0", "https://github.com/JaidedAI/EasyOCR" },
    { "pplcnet-textline-ori", "PP-LCNet_x1_0_textline_ori-f16.gguf", "",
      "PP-LCNet x1.0 text-line 0/180 orientation classifier (local conversion)", "13 MB", "apache-2.0",
      "https://github.com/PaddlePaddle/PaddleOCR" },

    { "surya-det", "surya-det-f16.gguf", "https://huggingface.co/cstr/surya-det-GGUF/resolve/main/surya-det-f16.gguf",
      "surya-ocr-2 text detection (EfficientViT segformer, 38M, 91 langs)", "73 MB", "openrail-m",
      "https://huggingface.co/cstr/surya-det-GGUF" },

    { "mixtex-zhen", "mixtex-zhen-f16.gguf", "", "MixTex Chinese+English LaTeX OCR (Swin-Tiny+RoBERTa, 86M)", "165 MB",
      "apache-2.0", "" },

    { "trocr-printed", "trocr-small-printed-q8_0.gguf",
      "https://huggingface.co/cstr/trocr-small-printed-GGUF/resolve/main/trocr-small-printed-q8_0.gguf",
      "TrOCR text recognition (DeiT+Transformer, printed)", "63 MB", "mit",
      "https://huggingface.co/cstr/trocr-small-printed-GGUF" },

    { "layout-heron", "layout-heron-f32.gguf",
      "https://huggingface.co/cstr/layout-heron-gguf/resolve/main/layout-heron-f32.gguf",
      "RT-DETRv2 document layout detection (ResNet-50+Transformer, 17 classes)", "161 MB", "apache-2.0",
      "https://huggingface.co/cstr/layout-heron-gguf" },

    { "qari-ocr", "qari-ocr-2b-q4_k.gguf",
      "https://huggingface.co/cstr/qari-ocr-crispembed-GGUF/resolve/main/qari-ocr-2b-q4_k.gguf",
      "Qari Arabic OCR with diacritics (Qwen2-VL-2B fine-tune)", "1300 MB", "apache-2.0",
      "https://huggingface.co/cstr/qari-ocr-crispembed-GGUF" },

    { "granite-vision", "granite-vision-3.3-2b-q8_0.gguf",
      "https://huggingface.co/cstr/granite-vision-crispembed-GGUF/resolve/main/granite-vision-3.3-2b-q8_0.gguf",
      "Granite Vision 3.3-2B OCR (SigLIP+Granite LLM, OCRBench 852)", "3212 MB", "apache-2.0",
      "https://huggingface.co/cstr/granite-vision-crispembed-GGUF" },

    { "granite-vision-q4k", "granite-vision-3.3-2b-q4_k.gguf",
      "https://huggingface.co/cstr/granite-vision-crispembed-GGUF/resolve/main/granite-vision-3.3-2b-q4_k.gguf",
      "Granite Vision 3.3-2B OCR Q4_K (LLM Q4_K, vision F16)", "1913 MB", "apache-2.0",
      "https://huggingface.co/cstr/granite-vision-crispembed-GGUF" },

    // dots.ocr removed — license is NOT pure MIT (supplemental PRC agreement
    // with unilateral amendment clause, mandatory attribution, use restrictions).
    // Code kept in feat/dots-ocr branch only.

    { "firered-ocr", "firered-ocr-q8_0.gguf",
      "https://huggingface.co/cstr/firered-ocr-crispembed-GGUF/resolve/main/firered-ocr-q8_0.gguf",
      "FireRed-OCR (Qwen3-VL 2B, GRPO, tables+LaTeX)", "2249 MB", "apache-2.0",
      "https://huggingface.co/cstr/firered-ocr-crispembed-GGUF" },

    { "firered-ocr-q4k", "firered-ocr-q4_k.gguf",
      "https://huggingface.co/cstr/firered-ocr-crispembed-GGUF/resolve/main/firered-ocr-q4_k.gguf",
      "FireRed-OCR Q4_K (Qwen3-VL 2B)", "1577 MB", "apache-2.0",
      "https://huggingface.co/cstr/firered-ocr-crispembed-GGUF" },

    { "nafnet-denoise", "nafnet-sidd-w32-q8_0.gguf",
      "https://huggingface.co/cstr/nafnet-sidd-GGUF/resolve/main/nafnet-sidd-w32-q8_0.gguf",
      "NAFNet image denoising (U-Net, 29M params, SIDD-trained)", "30 MB", "mit",
      "https://huggingface.co/cstr/nafnet-sidd-GGUF" },

    { "safmn-x4", "safmn-x4-f32.gguf", "https://huggingface.co/cstr/safmn-sr-GGUF/resolve/main/safmn-x4-f32.gguf",
      "SAFMN 4x super-resolution (228K params, ICCV 2023)", "0.9 MB", "apache-2.0",
      "https://huggingface.co/cstr/safmn-sr-GGUF" },

    { "esrgan-x4", "esrgan-x4-f32.gguf", "https://huggingface.co/cstr/esrgan-sr-GGUF/resolve/main/esrgan-x4-f32.gguf",
      "Real-ESRGAN 4x SR (SRVGGNetCompact, 620K params, BSD-3)", "2.4 MB", "bsd-3-clause",
      "https://huggingface.co/cstr/esrgan-sr-GGUF" },

    { "tps-loc", "tps-loc-f32.gguf", "https://huggingface.co/cstr/tps-loc-GGUF/resolve/main/tps-loc-f32.gguf",
      "TPS localization CNN for document dewarping (108K params, 20 control points)", "0.4 MB", "apache-2.0",
      "https://huggingface.co/cstr/tps-loc-GGUF" },

    // --- Text super-resolution ---

    { "tbsrn-telescope", "tbsrn-telescope-f16.gguf",
      "https://huggingface.co/cstr/text-super-resolution-gguf/resolve/main/tbsrn-telescope-f16.gguf",
      "TBSRN text-line SR (1.1M params, PaddleOCR Telescope)", "2.2 MB", "Apache-2.0",
      "https://huggingface.co/cstr/text-super-resolution-gguf" },

    { "pan-x4", "pan-x4-f16.gguf",
      "https://huggingface.co/cstr/text-super-resolution-gguf/resolve/main/pan-x4-f16.gguf",
      "PAN 4x image SR (272K params, PaddleGAN)", "0.5 MB", "Apache-2.0",
      "https://huggingface.co/cstr/text-super-resolution-gguf" },

    { "hat-sr-x4", "hat-sr-x4-f16.gguf",
      "https://huggingface.co/cstr/text-super-resolution-gguf/resolve/main/hat-sr-x4-f16.gguf",
      "HAT 4x SR (21M params, CVPR 2023 SOTA)", "40 MB", "MIT",
      "https://huggingface.co/cstr/text-super-resolution-gguf" },

    { "swinir-sr-x4", "swinir-light-x4-f16.gguf",
      "https://huggingface.co/cstr/text-super-resolution-gguf/resolve/main/swinir-light-x4-f16.gguf",
      "SwinIR-light 4x SR (930K params)", "15 MB", "Apache-2.0",
      "https://huggingface.co/cstr/text-super-resolution-gguf" },

    { "dat-sr-x2", "dat-light-x2-f16.gguf",
      "https://huggingface.co/cstr/text-super-resolution-gguf/resolve/main/dat-light-x2-f16.gguf",
      "DAT-light 2x SR (830K params, ICCV 2023, dual attention)", "38 MB", "Apache-2.0",
      "https://huggingface.co/cstr/text-super-resolution-gguf" },

    { "restormer-denoise", "restormer-denoise-f16.gguf",
      "https://huggingface.co/cstr/text-super-resolution-gguf/resolve/main/restormer-denoise-f16.gguf",
      "Restormer image restoration (26M params, CVPR 2022)", "50 MB", "Apache-2.0",
      "https://huggingface.co/cstr/text-super-resolution-gguf" },

    { "scunet-color", "scunet-color-f32.gguf",
      "https://huggingface.co/cstr/scunet-GGUF/resolve/main/scunet-color-f32.gguf",
      "SCUNet color denoising (Swin-Conv-UNet, 18M params, SIDD)", "69 MB", "apache-2.0",
      "https://huggingface.co/cstr/scunet-GGUF" },

    // Canonical lowercase repo id; cstr/InstructIR-GGUF only worked via
    // HuggingFace's case redirect.
    { "instructir", "instructir-f16.gguf",
      "https://huggingface.co/cstr/instructir-GGUF/resolve/main/instructir-f16.gguf",
      "InstructIR all-in-one restoration (NAFNet+ICB, 16M params, 7 tasks)", "32 MB", "MIT",
      "https://huggingface.co/cstr/instructir-GGUF" },

    // Back on f16, halving the download. The F16 defect was a shape bug in this
    // repo — the quantizer flattens 4-D conv weights to 2-D and three hidden
    // widths were read off ne[3] — fixed in src/adair.cpp (67ec560c). The f16 is
    // now uploaded and pinned; against the f32 on the same input it is
    // cos 0.99999994, max 1 LSB, mean 115.369 vs 115.370.
    // Repo id is the canonical lowercase one — cstr/AdaIR-GGUF only worked via
    // HuggingFace's case redirect.
    { "adair-5d", "adair-5d-f16.gguf", "https://huggingface.co/cstr/adair-GGUF/resolve/main/adair-5d-f16.gguf",
      "AdaIR all-in-one restoration (Restormer+AFLB+FFT, 28.8M params, 5 tasks)", "59 MB", "MIT",
      "https://huggingface.co/cstr/adair-GGUF" },

    // text-sr: NAFNet-SR engine — no default model; supply a custom trained GGUF.
    { "text-sr", "text-sr-nafnet.gguf", "", "NAFNet-SR text super-resolution engine (custom trained model required)",
      "", "apache-2.0", "" },

    { "qwen2vl-3b", "qwen2.5-vl-3b-q4_k.gguf",
      "https://huggingface.co/cstr/qwen2.5-vl-3b-crispembed-GGUF/resolve/main/qwen2.5-vl-3b-q4_k.gguf",
      "Qwen2.5-VL-3B VLM OCR (32-layer ViT + 36-layer Qwen2.5, German docs)", "2610 MB", "apache-2.0",
      "https://huggingface.co/cstr/qwen2.5-vl-3b-crispembed-GGUF" },

    { "olmocr-2-7b", "olmocr-2-7b-q4_k.gguf",
      "https://huggingface.co/cstr/olmOCR-2-7B-1025-GGUF/resolve/main/olmocr-2-7b-q4_k.gguf",
      "olmOCR-2-7B document OCR (Qwen2.5-VL-7B fine-tune, markdown+front-matter output)", "5468 MB", "apache-2.0",
      "https://huggingface.co/allenai/olmOCR-2-7B-1025" },

    { "smoldocling", "smoldocling-q8_0.gguf",
      "https://huggingface.co/cstr/smoldocling-GGUF/resolve/main/smoldocling-q8_0.gguf",
      "SmolDocling-256M full-page DocTags OCR (SigLIP + SmolLM2, tiled 512px input)", "261 MB", "apache-2.0",
      "https://huggingface.co/ds4sd/SmolDocling-256M-preview" },

    { "qwen3vl-2b", "qwen3-vl-2b-q4_k.gguf",
      "https://huggingface.co/cstr/qwen3-vl-2b-crispembed-gguf/resolve/main/qwen3-vl-2b-q4_k.gguf",
      "Qwen3-VL-2B VLM OCR (24-layer ViT + 28-layer Qwen3, DeepStack, IMROPE)", "1590 MB", "apache-2.0",
      "https://huggingface.co/cstr/qwen3-vl-2b-crispembed-gguf" },

    { "german-ocr-3.1", "german-ocr-3.1-q4_k.gguf",
      "https://huggingface.co/cstr/german-ocr-3.1-crispembed-GGUF/resolve/main/german-ocr-3.1-q4_k.gguf",
      "German-OCR-3.1 VLM (Qwen2-VL-2B fine-tune, German invoices/forms/receipts)", "1684 MB", "apache-2.0",
      "https://huggingface.co/keyvan-ai/german-ocr-3.1" },

    { "nanonets-ocr-s", "nanonets-ocr-s-q4_k.gguf",
      "https://huggingface.co/cstr/nanonets-ocr-s-crispembed-GGUF/resolve/main/nanonets-ocr-s-q4_k.gguf",
      "Nanonets-OCR-s VLM OCR (Qwen2.5-VL-3B fine-tune, 12+ languages)", "2610 MB", "apache-2.0",
      "https://huggingface.co/cstr/nanonets-ocr-s-crispembed-GGUF" },

    { "nanonets-ocr2-1.5b", "nanonets-ocr2-1.5b-q4_k.gguf",
      "https://huggingface.co/cstr/nanonets-ocr2-1.5b-crispembed-GGUF/resolve/main/nanonets-ocr2-1.5b-q4_k.gguf",
      "Nanonets-OCR2-1.5B VLM OCR (Qwen2-VL pruned 16L, 12+ languages incl. German)", "1346 MB", "apache-2.0",
      "https://huggingface.co/cstr/nanonets-ocr2-1.5b-crispembed-GGUF" },

    // q8_0, NOT q4_k: q4_k was measured broken for this checkpoint and withdrawn
    // from the repo (llm_layer_0 cos 0.594995, anti-correlated by layer 2 — it
    // loads and emits confident text that is not on the page). q8_0 is
    // 0.998033/0.995498 against the blueprint reference and transcribes.
    // Needs MSAC two-scale tiling AND the h2ogpt2 prompt template with no BOS;
    // all three are runtime-side and were the reason this model looked broken.
    // See PERFORMANCE.md "h2ovl-2b — resolved, and the quant ladder".
    { "h2ovl-mississippi-2b", "h2ovl-mississippi-2b-q8_0.gguf",
      "https://huggingface.co/cstr/h2ovl-mississippi-2b-crispembed-GGUF/resolve/main/h2ovl-mississippi-2b-q8_0.gguf",
      "H2OVL-Mississippi-2B VLM OCR (InternViT-300M + Danube-2-1.8B, OCRBench 782)", "2592 MB", "apache-2.0",
      "https://huggingface.co/cstr/h2ovl-mississippi-2b-crispembed-GGUF" },

    { "h2ovl-mississippi-800m", "h2ovl-800m-q4_k.gguf",
      "https://huggingface.co/cstr/h2ovl-800m-crispembed-GGUF/resolve/main/h2ovl-800m-q4_k.gguf",
      "H2OVL-Mississippi-0.8B VLM OCR (InternViT-300M + Danube-3-0.5B, OCRBench 751, edge)", "644 MB", "apache-2.0",
      "https://huggingface.co/cstr/h2ovl-800m-crispembed-GGUF" },

    { "internvl2-2b", "internvl2.5-2b-q4_k.gguf",
      "https://huggingface.co/cstr/internvl2.5-2b-crispembed-GGUF/resolve/main/internvl2.5-2b-q4_k.gguf",
      "InternVL2.5-2B VLM OCR (InternViT-300M + InternLM2.5-1.8B, EN+DE, OCRBench ~830)", "1400 MB", "mit",
      "https://huggingface.co/cstr/internvl2.5-2b-crispembed-GGUF" },

    { "internvl2-1b", "internvl2-1b-q4_k.gguf",
      "https://huggingface.co/cstr/internvl2-1b-crispembed-GGUF/resolve/main/internvl2-1b-q4_k.gguf",
      "InternVL2-1B VLM OCR (InternViT-300M + Qwen2-0.5B, 0.9B, edge/WASM, OCRBench 779)", "600 MB", "mit",
      "https://huggingface.co/cstr/internvl2-1b-crispembed-GGUF" },

    { "glm-ocr", "glm-ocr-q8_0.gguf",
      "https://huggingface.co/cstr/glm-ocr-crispembed-GGUF/resolve/main/glm-ocr-q8_0.gguf",
      "GLM-OCR document OCR (CogViT + GLM-0.5B, 0.9B, 8 languages, OmniDocBench #1)", "950 MB", "mit",
      "https://huggingface.co/cstr/glm-ocr-crispembed-GGUF" },

    { "got-ocr2", "got-ocr2-q4_k.gguf",
      "https://huggingface.co/cstr/got-ocr2-crispembed-GGUF/resolve/main/got-ocr2-q4_k.gguf",
      "GOT-OCR2 document OCR (SAM-ViT-B + Qwen2-0.5B, 0.7B, text/LaTeX/tables)", "445 MB", "apache-2.0",
      "https://huggingface.co/cstr/got-ocr2-crispembed-GGUF" },

    // LFM2.5-VL-3B — LiquidAI's own GGUF export, used directly rather than
    // re-hosted: our loader reads it unchanged, so a cstr/* mirror would add a
    // copy to keep in sync for nothing. Two files — the LLM and the SigLIP2
    // vision tower — and lfm2_vl_ocr_init() finds the tower by scanning the
    // model's directory for mmproj-<base>-{F16,Q8_0,BF16}.gguf, hence the
    // companion fields. LFM-1.0 is revenue-capped, so this is a restricted
    // entry: --accept-license lfm1.0 (or CRISPEMBED_ACCEPT_LICENSE) gates the
    // download, and the engine separately requires CRISPEMBED_ACCEPT_LFM_LICENSE=1
    // to load.
    { "lfm2-vl", "LFM2.5-VL-3B-Q4_K_M.gguf",
      "https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF/resolve/main/LFM2.5-VL-3B-Q4_K_M.gguf",
      "LFM2.5-VL document OCR (SigLIP2 NaFlex + LFM2 hybrid conv/attn, 3.4B, Q4_K_M)", "1.67 GB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-VL-3B", nullptr, "mmproj-LFM2.5-VL-3B-F16.gguf",
      "https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF/resolve/main/mmproj-LFM2.5-VL-3B-F16.gguf", "854 MB" },

    { "pix2struct-base", "pix2struct-base-q8_0.gguf",
      "https://huggingface.co/cstr/pix2struct-GGUF/resolve/main/pix2struct-base-q8_0.gguf",
      "Pix2Struct document understanding (ViT + T5 decoder, 282M, image-to-text)", "467 MB", "apache-2.0",
      "https://huggingface.co/cstr/pix2struct-GGUF" },

    // The base checkpoint is pretraining-only (screenshot pseudo-HTML) and
    // babbles on natural documents (HF-verified); textcaps is the promptless
    // finetuned variant that produces real text (the docvqa variants need a
    // question rendered into the image, which the engine does not do yet).
    { "pix2struct-textcaps", "pix2struct-textcaps-q8_0.gguf",
      "https://huggingface.co/cstr/pix2struct-GGUF/resolve/main/pix2struct-textcaps-q8_0.gguf",
      "Pix2Struct TextCaps captioning (finetuned, promptless; ViT + T5 decoder, 282M)", "354 MB", "apache-2.0",
      "https://huggingface.co/cstr/pix2struct-GGUF" },

    // Stacked MoE experts (converter #4): ~1.3 GB lower resident footprint than the
    // per-expert layout; the loader falls back to per-expert for older GGUFs. Distinct
    // cache filename so an existing per-expert cache re-downloads the stacked file.
    { "deepseek-ocr2", "deepseek-ocr2-q4_k-stacked.gguf",
      "https://huggingface.co/cstr/deepseek-ocr2-crispembed-GGUF/resolve/main/deepseek-ocr2-q4_k-stacked.gguf",
      "DeepSeek-OCR-2 (SAM + Qwen2-enc + MoE decoder, 3.4B, grounding; stacked experts)", "2.3 GB", "apache-2.0",
      "https://huggingface.co/cstr/deepseek-ocr2-crispembed-GGUF" },

    // Stacked MoE experts: ~1.2 GB lower resident footprint than the per-expert
    // layout; the loader falls back to per-expert for older GGUFs. Distinct cache
    // filename so an existing per-expert cache re-downloads the stacked file.
    { "unlimited-ocr", "unlimited-ocr-q4_k-stacked.gguf",
      "https://huggingface.co/cstr/unlimited-ocr-crispembed-GGUF/resolve/main/unlimited-ocr-q4_k-stacked.gguf",
      "Unlimited-OCR (SAM + CLIP + MoE decoder, 3.3B, full-page OCR; stacked experts)", "2.1 GB", "mit",
      "https://huggingface.co/cstr/unlimited-ocr-crispembed-GGUF" },

    // PaddleOCR-VL-0.9B — NaViT ViT + ERNIE-4.5 LLM, 109 languages
    // License: Apache-2.0
    { "paddleocr-vl", "paddleocr-vl-0.9b-q8_0.gguf",
      "https://huggingface.co/cstr/paddleocr-vl-0.9b-GGUF/resolve/main/paddleocr-vl-0.9b-q8_0.gguf",
      "PaddleOCR-VL 0.9B NaViT+ERNIE-4.5 109-lang OCR", "1.4 GB", "apache-2.0",
      "https://huggingface.co/PaddlePaddle/PaddleOCR-VL" },

    { "paddleocr-vl-q4k", "paddleocr-vl-0.9b-q4_k.gguf",
      "https://huggingface.co/cstr/paddleocr-vl-0.9b-GGUF/resolve/main/paddleocr-vl-0.9b-q4_k.gguf",
      "PaddleOCR-VL 0.9B NaViT+ERNIE-4.5 Q4_K 109-lang OCR", "1.3 GB", "apache-2.0",
      "https://huggingface.co/PaddlePaddle/PaddleOCR-VL" },

    // PaddleOCR-VL-1.6 — same arch as 0.9B, improved training (96.3% OmniDocBench)
    { "paddleocr-vl-1.6", "paddleocr-vl-1.6-q8_0.gguf",
      "https://huggingface.co/cstr/paddleocr-vl-1.6-GGUF/resolve/main/paddleocr-vl-1.6-q8_0.gguf",
      "PaddleOCR-VL 1.6 NaViT+ERNIE-4.5 SOTA 109-lang OCR", "1.4 GB", "apache-2.0",
      "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6" },

    { "paddleocr-vl-1.6-q4k", "paddleocr-vl-1.6-q4_k.gguf",
      "https://huggingface.co/cstr/paddleocr-vl-1.6-GGUF/resolve/main/paddleocr-vl-1.6-q4_k.gguf",
      "PaddleOCR-VL 1.6 NaViT+ERNIE-4.5 SOTA Q4_K 109-lang OCR", "1.3 GB", "apache-2.0",
      "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6" },

    // LFM2.5-Embedding-350M — LiquidAI bidirectional hybrid (10 ShortConv + 6 GQA)
    // 1024-dim CLS pooling, 11 languages (EN/ES/DE/FR/IT/PT/AR/SV/NO/JA/KO)
    // License: LFM Open License v1.0 (commercial use requires separate agreement)
    // Default = best flavor (Q4_K+imatrix, A/B winner). -q4k now serves the
    // imatrix build (same size, strictly better than the old plain Q4_K).
    { "lfm2-embed", "lfm2-embed-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/lfm2-embed-GGUF/resolve/main/lfm2-embed-q4_k-imatrix.gguf",
      "LFM2.5 1024d 11-lang CLS hybrid (350M, Q4_K+imatrix)", "235 MB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M" },

    { "lfm2-embed-q4k", "lfm2-embed-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/lfm2-embed-GGUF/resolve/main/lfm2-embed-q4_k-imatrix.gguf",
      "LFM2.5 1024d 11-lang CLS hybrid Q4_K+imatrix (350M)", "235 MB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M" },

    { "lfm2-embed-iq4xs", "lfm2-embed-iq4_xs.gguf",
      "https://huggingface.co/cstr/lfm2-embed-GGUF/resolve/main/lfm2-embed-iq4_xs.gguf",
      "LFM2.5 1024d 11-lang CLS hybrid IQ4_XS+imatrix (350M)", "226 MB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M" },

    { "lfm2-embed-q8", "lfm2-embed-q8_0.gguf",
      "https://huggingface.co/cstr/lfm2-embed-GGUF/resolve/main/lfm2-embed-q8_0.gguf",
      "LFM2.5 1024d 11-lang CLS hybrid Q8_0 (350M)", "379 MB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M" },

    // LFM2.5-ColBERT-350M — LiquidAI ColBERT multi-vector (per-token 128d)
    // Same backbone as LFM2.5-Embedding + Dense projection head
    // License: LFM Open License v1.0
    { "lfm2-colbert", "lfm2-colbert-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/lfm2-colbert-GGUF/resolve/main/lfm2-colbert-q4_k-imatrix.gguf",
      "LFM2.5 ColBERT 128d multi-vector hybrid 350M (Q4_K+imatrix, per-token cos 0.9975)", "234 MB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M" },

    { "lfm2-colbert-q4k", "lfm2-colbert-q4_k.gguf",
      "https://huggingface.co/cstr/lfm2-colbert-GGUF/resolve/main/lfm2-colbert-q4_k.gguf",
      "LFM2.5 ColBERT 128d multi-vector Q4_K (350M)", "254 MB", "lfm1.0",
      "https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M" },

    { "gliner-lfm", "gliner-lfm-q8_0.gguf",
      "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF/resolve/main/gliner-lfm-q8_0.gguf",
      "GLiNER zero-shot NER (LFM2.5-350M bidirectional, 5 languages)", "419 MB", "lfm1.0",
      "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF" },

    { "gliner-lfm-q4k", "gliner-lfm-q4_k.gguf",
      "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF/resolve/main/gliner-lfm-q4_k.gguf",
      "GLiNER zero-shot NER (LFM2.5-350M, Q4_K compact)", "254 MB", "lfm1.0",
      "https://huggingface.co/cstr/sauerkraut-gliner-lfm-GGUF" },

    { "gliner-deberta", "gliner-deberta-iq4_xs.gguf",
      "https://huggingface.co/cstr/gliner-deberta-GGUF/resolve/main/gliner-deberta-iq4_xs.gguf",
      "GLiNER zero-shot NER DeBERTa-v3-base 209M (IQ4_XS+imatrix, span-F1 1.0)", "159 MB", "apache-2.0",
      "https://huggingface.co/cstr/gliner-deberta-GGUF" },

    { "gliner-deberta-q4k", "gliner-deberta-q4_k.gguf",
      "https://huggingface.co/cstr/gliner-deberta-GGUF/resolve/main/gliner-deberta-q4_k.gguf",
      "GLiNER zero-shot NER (DeBERTa-v3-base, Q4_K compact)", "152 MB", "apache-2.0",
      "https://huggingface.co/cstr/gliner-deberta-GGUF" },

    { "lilt-funsd", "lilt-funsd-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/lilt-funsd-GGUF/resolve/main/lilt-funsd-q4_k-imatrix.gguf",
      "LiLT FUNSD form understanding (130M, Q4_K+imatrix — 16/16 KIE labels vs f32)", "94 MB", "mit",
      "https://huggingface.co/cstr/lilt-funsd-GGUF" },
    { "lilt-funsd-iq4xs", "lilt-funsd-iq4_xs.gguf",
      "https://huggingface.co/cstr/lilt-funsd-GGUF/resolve/main/lilt-funsd-iq4_xs.gguf", "LiLT FUNSD (IQ4_XS+imatrix)",
      "92 MB", "mit", "https://huggingface.co/cstr/lilt-funsd-GGUF" },
    { "lilt-funsd-q8", "lilt-funsd-q8_0.gguf",
      "https://huggingface.co/cstr/lilt-funsd-GGUF/resolve/main/lilt-funsd-q8_0.gguf", "LiLT FUNSD (Q8_0)", "140 MB",
      "mit", "https://huggingface.co/cstr/lilt-funsd-GGUF" },
    { "lilt-funsd-f32", "lilt-funsd-f32.gguf",
      "https://huggingface.co/cstr/lilt-funsd-GGUF/resolve/main/lilt-funsd-f32.gguf", "LiLT FUNSD (F32)", "497 MB",
      "mit", "https://huggingface.co/cstr/lilt-funsd-GGUF" },

    { "lilt-base", "lilt-base-f32.gguf", "https://huggingface.co/cstr/lilt-base-GGUF/resolve/main/lilt-base-f32.gguf",
      "LiLT base encoder (130M params, MIT)", "497 MB", "mit", "https://huggingface.co/cstr/lilt-base-GGUF" },

    // BERT fixed-label NER (CoNLL-03: PER/LOC/ORG/MISC)
    { "bert-base-ner", "bert-base-ner-iq4_xs.gguf",
      "https://huggingface.co/cstr/bert-base-NER-GGUF/resolve/main/bert-base-ner-iq4_xs.gguf",
      "BERT NER 110M CoNLL-03 9 labels (IQ4_XS+imatrix, span-F1 1.0)", "70 MB", "mit",
      "https://huggingface.co/cstr/bert-base-NER-GGUF" },

    { "xlmr-ner-hrl", "xlmr-ner-hrl-iq4_xs.gguf",
      "https://huggingface.co/cstr/xlmr-ner-hrl-GGUF/resolve/main/xlmr-ner-hrl-iq4_xs.gguf",
      "XLM-R multilingual NER 278M 10 langs 9 labels (IQ4_XS+imatrix, span-F1 1.0)", "256 MB", "other*",
      "https://huggingface.co/cstr/xlmr-ner-hrl-GGUF" },

    // Text language identification
    { "cld3", "cld3-f16.gguf", "https://huggingface.co/cstr/cld3-GGUF/resolve/main/cld3-f16.gguf",
      "Google CLD3 text LID (109 languages, Apache-2.0)", "1.2 MB", "apache-2.0",
      "https://huggingface.co/cstr/cld3-GGUF" },

    // Two bugs were fixed here. The `lid-` prefix is CrispASR's naming
    // convention and never existed in this repo (published files are
    // glotlid-{f16,q8_0,q5_k,q4_k}.gguf), and the old "3.3 MB" was wrong by
    // ~250x — GlotLID-V3 is a 2102-language FastText model whose f16 is
    // 848 MB. The default keeps the f16 the entry always named rather than
    // silently switching precision; the quants are registered so 848 MB is
    // not the only option. Quant accuracy for LID is unmeasured here: this
    // repo has no LID engine (text_lid_dispatch.h is an optional CrispASR
    // header behind __has_include), so none of these was run.
    { "glotlid", "glotlid-f16.gguf", "https://huggingface.co/cstr/glotlid-GGUF/resolve/main/glotlid-f16.gguf",
      "GlotLID-V3 text LID (2102 ISO 639-3 languages, F16)", "848 MB", "apache-2.0",
      "https://huggingface.co/cstr/glotlid-GGUF" },
    { "glotlid-q8", "glotlid-q8_0.gguf", "https://huggingface.co/cstr/glotlid-GGUF/resolve/main/glotlid-q8_0.gguf",
      "GlotLID-V3 text LID (2102 ISO 639-3 languages, Q8_0)", "455 MB", "apache-2.0",
      "https://huggingface.co/cstr/glotlid-GGUF" },
    { "glotlid-q4k", "glotlid-q4_k.gguf", "https://huggingface.co/cstr/glotlid-GGUF/resolve/main/glotlid-q4_k.gguf",
      "GlotLID-V3 text LID (2102 ISO 639-3 languages, Q4_K)", "246 MB", "apache-2.0",
      "https://huggingface.co/cstr/glotlid-GGUF" },

    // LightOnOCR-2-1B — Pixtral ViT + Qwen3 decoder (OCR Arena #2)
    { "lightonocr", "lightonocr-1b-q4_k.gguf",
      "https://huggingface.co/cstr/lightonocr-GGUF/resolve/main/lightonocr-1b-q4_k.gguf",
      "LightOnOCR-2-1B (1B, Pixtral+Qwen3, Apache-2.0)", "622 MB", "apache-2.0",
      "https://huggingface.co/cstr/lightonocr-GGUF" },

    // Tesseract LSTM line OCR — lightweight multilingual (from tessdata_best)
    { "tesseract-eng", "tesseract-eng-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-eng-q8_0.gguf",
      "Tesseract LSTM English line OCR (1.5M, CTC, 126-lang family)", "1.5 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-deu", "tesseract-deu-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-deu-q8_0.gguf",
      "Tesseract LSTM German line OCR (940K, CTC)", "976 KB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-fra", "tesseract-fra-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-fra-q8_0.gguf",
      "Tesseract LSTM French line OCR (391K, CTC)", "435 KB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-spa", "tesseract-spa-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-spa-q8_0.gguf",
      "Tesseract LSTM Spanish line OCR (1.5M, CTC)", "1.5 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-ita", "tesseract-ita-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-ita-q8_0.gguf",
      "Tesseract LSTM Italian line OCR (822K, CTC)", "860 KB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-por", "tesseract-por-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-por-q8_0.gguf",
      "Tesseract LSTM Portuguese line OCR (822K, CTC)", "860 KB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-nld", "tesseract-nld-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-nld-q8_0.gguf",
      "Tesseract LSTM Dutch line OCR (408K, CTC)", "449 KB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin" },

    { "tesseract-rus", "tesseract-rus-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-rus-q8_0.gguf",
      "Tesseract LSTM Russian line OCR (1.5M, CTC, Cyrillic)", "1.5 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin+cyrillic" },

    { "tesseract-ara", "tesseract-ara-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-ara-q8_0.gguf",
      "Tesseract LSTM Arabic line OCR (1.4M, CTC, RTL)", "1.5 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin+arabic" },

    { "tesseract-chi-sim", "tesseract-chi_sim-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-chi_sim-q8_0.gguf",
      "Tesseract LSTM Chinese Simplified line OCR (1.5M, CTC, CJK)", "1.6 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin+cjk" },

    { "tesseract-jpn", "tesseract-jpn-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-jpn-q8_0.gguf",
      "Tesseract LSTM Japanese line OCR (1.6M, CTC, CJK+Kana)", "1.7 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin+cjk+kana" },

    { "tesseract-kor", "tesseract-kor-q8_0.gguf",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF/resolve/main/tesseract-kor-q8_0.gguf",
      "Tesseract LSTM Korean line OCR (1.5M, CTC, Hangul)", "1.5 MB", "apache-2.0",
      "https://huggingface.co/cstr/tesseract-lstm-GGUF", "latin+hangul" },

    // Punctuation restoration models
    { "fireredpunc", "fireredpunc-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q4_k-imatrix.gguf",
      "FireRedPunc punctuation (chinese-bert-wwm-ext, 5 classes, Q4_K+imatrix — 2.8x lower KL vs f16)", "58 MB",
      "apache-2.0", "https://huggingface.co/cstr/fireredpunc-GGUF" },
    { "fireredpunc-iq4xs", "fireredpunc-iq4_xs.gguf",
      "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-iq4_xs.gguf",
      "FireRedPunc punctuation (IQ4_XS+imatrix)", "43 MB", "apache-2.0",
      "https://huggingface.co/cstr/fireredpunc-GGUF" },
    { "fireredpunc-q4k", "fireredpunc-q4_k.gguf",
      "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q4_k.gguf",
      "FireRedPunc punctuation (Q4_K, no imatrix)", "58 MB", "apache-2.0",
      "https://huggingface.co/cstr/fireredpunc-GGUF" },
    { "fireredpunc-q8", "fireredpunc-q8_0.gguf",
      "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q8_0.gguf",
      "FireRedPunc punctuation (Q8_0)", "109 MB", "apache-2.0", "https://huggingface.co/cstr/fireredpunc-GGUF" },

    { "fullstop-punc", "fullstop-punc-q4_k.gguf",
      "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF/resolve/main/fullstop-punc-q4_k.gguf",
      "Fullstop punctuation restoration (XLM-R-large, multilingual)", "321 MB", "mit",
      "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF" },

    { "fullstop-punc-q8", "fullstop-punc-q8_0.gguf",
      "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF/resolve/main/fullstop-punc-q8_0.gguf",
      "Fullstop punctuation restoration (XLM-R-large, q8_0 — exact HF parity)", "567 MB", "mit",
      "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF" },

    { "pcs", "pcs-xlmr-base-q4_k-imatrix.gguf",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k-imatrix.gguf",
      "PCS punct+caps+segmentation (XLM-R-base, Q4_K+imatrix — 4.2x lower KL vs f32)", "163 MB", "mit",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF" },
    { "pcs-iq4xs", "pcs-xlmr-base-iq4_xs.gguf",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-iq4_xs.gguf",
      "PCS punct+caps+segmentation (IQ4_XS+imatrix)", "154 MB", "mit",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF" },
    { "pcs-q4k", "pcs-xlmr-base-q4_k.gguf",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k.gguf",
      "PCS punct+caps+segmentation (Q4_K, no imatrix)", "163 MB", "mit",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF" },

    { "pcs-q8", "pcs-xlmr-base-q8_0.gguf",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q8_0.gguf",
      "PCS punct+caps+segmentation (XLM-R-base, q8_0 — exact ONNX parity)", "287 MB", "mit",
      "https://huggingface.co/cstr/pcs-xlmr-base-GGUF" },

    // Uni-MuMER handwritten math OCR (Qwen3-VL / Qwen2.5-VL fine-tunes)
    { "uni-mumer-qwen3-vl-2b", "uni-mumer-qwen3-vl-2b-q4_k.gguf",
      "https://huggingface.co/cstr/uni-mumer-qwen3-vl-2b-GGUF/resolve/main/uni-mumer-qwen3-vl-2b-q4_k.gguf",
      "Uni-MuMER handwritten math→LaTeX (Qwen3-VL-2B, 82% CROHME)", "1509 MB", "apache-2.0",
      "https://huggingface.co/cstr/uni-mumer-qwen3-vl-2b-GGUF" },

    { "uni-mumer-qwen2.5-vl-3b", "uni-mumer-qwen2.5-vl-3b-q4_k.gguf",
      "https://huggingface.co/cstr/uni-mumer-qwen2.5-vl-3b-GGUF/resolve/main/uni-mumer-qwen2.5-vl-3b-q4_k.gguf",
      "Uni-MuMER handwritten math→LaTeX (Qwen2.5-VL-3B, 82.25% CROHME)", "2614 MB", "apache-2.0",
      "https://huggingface.co/cstr/uni-mumer-qwen2.5-vl-3b-GGUF" },

    // TexTeller 3.0 math OCR (ViT + TrOCR, 310M, printed + handwritten)
    { "texteller-3", "texteller-3-q8_0.gguf",
      "https://huggingface.co/cstr/texteller-3-GGUF/resolve/main/texteller-3-q8_0.gguf",
      "TexTeller 3.0 math→LaTeX (ViT+TrOCR, 310M, EN+CN)", "302 MB", "apache-2.0",
      "https://huggingface.co/cstr/texteller-3-GGUF" },

    // Sheet Music Transformer — Optical Music Recognition (ConvNext + transformer,
    // 21.4M). q8_0 decodes identically to HF; q4_k is too lossy for the AR decode.
    { "smt-grandstaff", "smt-grandstaff-q8_0.gguf",
      "https://huggingface.co/cstr/smt-grandstaff-GGUF/resolve/main/smt-grandstaff-q8_0.gguf",
      "Sheet Music Transformer OMR: staff notation→bekern (pianoform, 21.4M)", "24 MB", "mit",
      "https://huggingface.co/antoniorv6/smt-grandstaff" },

    // SMT++ full-page pianoform OMR (antoniorv6/SMT rewrite, 10.9M). Same engine
    // as smt-grandstaff but scaled config (maxlen 4353, 181-token vocab) + smt-main
    // forward (scaled attn, no pre-head ReLU) + reduce_ratio=1.0/invert preproc.
    // q8_0 per-stage cos ≥0.9998; greedy decode byte-identical to HF at f32/q8_0.
    { "smt-fp", "smt-fp-grandstaff-q8_0.gguf",
      "https://huggingface.co/cstr/smt-fp-grandstaff-GGUF/resolve/main/smt-fp-grandstaff-q8_0.gguf",
      "SMT++ full-page OMR: whole pianoform page→bekern (10.9M)", "16 MB", "mit",
      "https://huggingface.co/PRAIG/smt-fp-grandstaff" },

    // Polyphonic-TrOMR — Optical Music Recognition (ResNetV2+ViT encoder +
    // x-transformers decoder, ~22M). q8_0 decodes byte-identically to the
    // reference; camera/photo-robust. rhythm/pitch/lift streams merged to notation.
    { "tromr", "tromr-q8_0.gguf", "https://huggingface.co/cstr/tromr-GGUF/resolve/main/tromr-q8_0.gguf",
      "Polyphonic-TrOMR OMR: staff image→rhythm/pitch/lift notation (~22M, camera-robust)", "31 MB", "apache-2.0",
      "https://github.com/NetEase/Polyphonic-TrOMR" },

    // Flova/omr_transformer — handwritten/whiteboard OMR (DonutSwin + 4L mBART VED,
    // 143M). q8_0 decodes byte-identically to HF. Only permissive handwritten-music
    // OMR model; "simple notes" → LilyPond.
    { "flova", "flova-q8_0.gguf", "https://huggingface.co/cstr/flova-omr-GGUF/resolve/main/flova-q8_0.gguf",
      "Flova/omr_transformer OMR: handwritten/whiteboard music→LilyPond (143M)", "162 MB", "apache-2.0",
      "https://huggingface.co/Flova/omr_transformer" },
    // q4_k is half the size and decoded byte-exact on all three sample images
    // (the encoder's per-token cosine drifts but the greedy decode never flips).
    // q8_0 stays the default; q4_k is the smaller-download option (e.g. browser).
    { "flova-q4k", "flova-q4_k.gguf", "https://huggingface.co/cstr/flova-omr-GGUF/resolve/main/flova-q4_k.gguf",
      "Flova/omr_transformer OMR: handwritten music→LilyPond (Q4_K, byte-exact on samples)", "88 MB", "apache-2.0",
      "https://huggingface.co/Flova/omr_transformer" },
    { "flova-q8", "flova-q8_0.gguf", "https://huggingface.co/cstr/flova-omr-GGUF/resolve/main/flova-q8_0.gguf",
      "Flova/omr_transformer OMR: handwritten music→LilyPond (Q8_0)", "162 MB", "apache-2.0",
      "https://huggingface.co/Flova/omr_transformer" },

    // Transcoda-59M — zero-shot full-page OMR (ConvNeXt-V2 enc + 8L RoPE cross-attn
    // decoder, 59M). Score image→Humdrum **kern. OMR-NED SOTA on real historical
    // scans. Weights cc-by-4.0 (attribution required).
    { "transcoda", "transcoda-q8_0.gguf",
      "https://huggingface.co/cstr/transcoda-omr-GGUF/resolve/main/transcoda-q8_0.gguf",
      "Transcoda-59M zero-shot OMR: full-page score→Humdrum **kern (59M)", "69 MB", "cc-by-4.0",
      "https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1" },

    { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }
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
    const std::filesystem::path requested(path);
    const std::filesystem::file_status link_status = std::filesystem::symlink_status(requested, ec);
    if (!ec && std::filesystem::is_symlink(link_status)) {
        const std::filesystem::path target = std::filesystem::read_symlink(requested, ec);
        if (!ec) {
            const std::filesystem::path resolved = target.is_absolute() ? target : requested.parent_path() / target;
            std::filesystem::create_directories(resolved, ec);
            return;
        }
    }
    std::filesystem::create_directories(requested, ec);
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
// Accepts the payload only when its SHA-256 matches the pin for `source_url`
// in the generated model_hashes.h. A download that "succeeded" says nothing
// about what arrived; a GGUF is a graph this process then executes, so an
// unverified swap at the re-host is a code-execution path, not a data bug.
static bool download_file(const std::string & source_url, const std::string & dest_path) {
#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    (void)source_url;
    (void)dest_path;
    return false;
#else
    if (!url_is_https(source_url)) {
        fprintf(stderr,
                "crispembed: refusing to download over a non-HTTPS URL:\n  %s\n"
                "Model payloads are executed as ggml graphs; plaintext transport is not\n"
                "an acceptable channel for them.\n",
                source_url.c_str());
        return false;
    }

    const char * expected_sha = model_pinned_sha256(source_url.c_str());
    if (!expected_sha && !unpinned_downloads_allowed()) {
        fprintf(stderr,
                "crispembed: no SHA-256 pin for\n  %s\n"
                "Refusing to install an unverified model payload. If this URL was just\n"
                "added to the registry, regenerate the pins:\n"
                "    python tools/fetch_model_hashes.py\n"
                "To override for a one-off, set CRISPEMBED_ALLOW_UNPINNED_MODEL=1.\n",
                source_url.c_str());
        return false;
    }

    std::string tmp = dest_path + ".tmp";

    // Ensure the destination directory exists — curl/wget will not create it,
    // and a missing cache dir otherwise fails with an opaque "No such file or
    // directory" while writing the .tmp.
    {
        std::filesystem::path parent = std::filesystem::path(dest_path).parent_path();
        if (!parent.empty()) mkdirs(parent.string());
    }

    // Resume-aware: if a previous attempt left a partial .tmp, log the
    // recovery and pass `-C -` / `-c` to curl / wget so we resume from
    // there instead of starting over. Hours of bandwidth saved on flaky
    // links. On success the .tmp is renamed; on failure we deliberately
    // KEEP the .tmp so the next attempt can resume rather than redo.
    long long resume_from = file_size(tmp);
    if (resume_from > 0) {
        fprintf(stderr, "Resuming download from %.1f MB...\n", resume_from / (1024.0 * 1024.0));
    }

    // `-C -` (curl) and `-c` (wget) tell the client to ask for HTTP Range
    // bytes=N- where N is the existing local size. Both handle the
    // already-complete case (HTTP 416) gracefully.
#ifdef _WIN32
    // Windows 10+ bundles curl.exe — supports -C -.
    std::string cmd = "curl.exe -fL -C - --progress-bar -o \"" + tmp + "\" \"" + source_url + "\"";
#else
    std::string cmd = "curl -fL -C - --progress-bar -o \"" + tmp + "\" \"" + source_url + "\"";
#endif
    // Hash the .tmp before it is renamed into the cache, so a payload that
    // fails the pin never occupies the path a later run would treat as a
    // valid cache hit. A mismatch deletes the .tmp rather than keeping it for
    // resume: resuming onto wrong bytes only produces more wrong bytes.
    auto install_if_verified = [&]() -> bool {
        if (!file_exists(tmp)) return false;
        if (!expected_sha) { // only reachable with CRISPEMBED_ALLOW_UNPINNED_MODEL=1
            fprintf(stderr, "crispembed: WARNING — installing UNVERIFIED model payload from %s\n", source_url.c_str());
            rename(tmp.c_str(), dest_path.c_str());
            return true;
        }
        fprintf(stderr, "Verifying SHA-256...\n");
        const std::string actual = sha256_file(tmp);
        if (actual.empty()) {
            fprintf(stderr, "crispembed: could not read %s to verify it.\n", tmp.c_str());
            return false;
        }
        if (actual != expected_sha) {
            fprintf(stderr,
                    "crispembed: SHA-256 MISMATCH for\n  %s\n"
                    "  expected %s\n"
                    "  actual   %s\n"
                    "Discarding the download. The re-host content changed, the transfer was\n"
                    "corrupted, or a stale partial was resumed onto. If the model was\n"
                    "legitimately re-uploaded, re-run tools/fetch_model_hashes.py.\n",
                    source_url.c_str(), expected_sha, actual.c_str());
            std::remove(tmp.c_str());
            return false;
        }
        rename(tmp.c_str(), dest_path.c_str());
        return true;
    };

    // NOLINTNEXTLINE(bugprone-command-processor)
    int ret = system(cmd.c_str());
    if (ret == 0 && install_if_verified()) return true;

#ifndef _WIN32
    // wget fallback (Linux/macOS only). `-c` resumes the partial .tmp.
    cmd = "wget -c -q --show-progress -O \"" + tmp + "\" \"" + source_url + "\"";
    // NOLINTNEXTLINE(bugprone-command-processor)
    ret = system(cmd.c_str());
    if (ret == 0 && install_if_verified()) return true;
#endif

    // Both attempts failed. Keep the .tmp so the next invocation can
    // resume from wherever curl/wget reached. If the partial is corrupt
    // (server changed, mid-byte error) the caller can `rm <cache>/*.tmp`
    // manually — bandwidth-cheap to lose, expensive to redo from zero.
    long long partial = file_size(tmp);
    if (partial > 0) {
        fprintf(stderr, "Download failed; partial %.1f MB kept at %s — re-run to resume.\n",
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
        "gemma",    "llama2",        "llama3",
        "llama3.1", "llama3.2",      "llama3.3",
        "llama4",   "qwen-research", "mistral-ai-research",
        "lfm1.0",   "other",         nullptr,
    };
    for (const char ** p = restricted; *p; ++p) {
        if (strcmp(spdx, *p) == 0) return true;
    }
    return false;
}

bool accept_biometric_use(const char * model_label, bool accepted_flag) {
    // Every success path also arms the library-level gate in
    // crispembed_face_init(), so the acknowledgement is made once and honoured
    // by both the CLI's own cnn_embed calls and the public C ABI.
    if (accepted_flag) {
        crispembed_accept_biometric_use();
        return true;
    }
    const char * env = std::getenv("CRISPEMBED_ACCEPT_BIOMETRIC");
    if (env && *env && strcmp(env, "0") != 0) {
        crispembed_accept_biometric_use();
        return true;
    }

    const char * label = (model_label && *model_label) ? model_label : "this model";
    fprintf(stderr, "\n'%s' is a FACE RECOGNITION model.\n", label);
    fprintf(stderr, "Its output is a biometric template — special-category personal data under\n");
    fprintf(stderr, "GDPR Art. 9, which generally needs an Art. 9(2) basis (e.g. explicit consent)\n");
    fprintf(stderr, "before you process it.\n");
    fprintf(stderr, "Using it to search a gallery (1:N identification) builds a biometric\n");
    fprintf(stderr, "identification system: high-risk under EU AI Act Annex III §1 from\n");
    fprintf(stderr, "2 December 2027, and prohibited outright in some settings (Art. 5).\n");
    fprintf(stderr, "See POLICY.md.\n\n");

    if (isatty(fileno(stdin))) {
        fprintf(stderr, "Acknowledge and continue? [y/N] ");
        char c = 0;
        if (scanf(" %c", &c) != 1 || (c != 'y' && c != 'Y')) return false;
        crispembed_accept_biometric_use();
        return true;
    }

    fprintf(stderr, "error: refusing to run a face recognition model without acknowledgement.\n");
    fprintf(stderr, "       Pass --accept-biometric (or set CRISPEMBED_ACCEPT_BIOMETRIC=1).\n");
    return false;
}

static bool license_accepted(const char * spdx, const std::string & accepted_arg) {
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

// Install an entry's companion file (an mmproj vision tower, say) next to the
// main GGUF. The engine finds it by scanning the model's own directory, so the
// two must land in the same place. Returns false only when the companion was
// wanted and could not be installed; the caller warns rather than failing the
// main resolve, so the engine reports the specific missing piece.
static bool ensure_companion(const ModelEntry * entry, const std::string & dir) {
    if (!entry->companion_url || !entry->companion_file) return true;

    const std::string path = dir + "/" + entry->companion_file;
    if (file_exists(path)) return true;

    if (!download_supported()) {
        fprintf(stderr, "Model '%s' needs companion file %s, and auto-download is unavailable.\n", entry->name,
                entry->companion_file);
        return false;
    }

    // No second license prompt: the companion ships under the same terms as
    // the model it belongs to, and the main file's gate has already run.
    fprintf(stderr, "Downloading companion %s (%s)...\n", entry->companion_file,
            entry->companion_size ? entry->companion_size : entry->approx_size);
    if (!download_file(entry->companion_url, path)) {
        fprintf(stderr, "Companion download failed (%s).\n", entry->companion_file);
        return false;
    }
    fprintf(stderr, "Downloaded to %s\n", path.c_str());
    return true;
}

std::string resolve_model(const std::string & arg, bool auto_download, const std::string & accepted_license) {
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
        // The model is cached but a companion may not be — a cache populated
        // before the companion was registered, or an interrupted install.
        ensure_companion(entry, dir);
        return cached;
    }

    const bool restricted = license_requires_acceptance(entry->license);
    const bool accepted = license_accepted(entry->license, accepted_license);
    const bool is_tty = isatty(fileno(stdin));

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
                fprintf(stderr, "Model '%s' is released under a restricted license:\n", entry->name);
                fprintf(stderr, "  License:    %s\n", entry->license);
                fprintf(stderr, "  Model card: %s\n", entry->model_card_url ? entry->model_card_url : "(unknown)");
                if (strncmp(entry->license, "cc-by-nc", 8) == 0) {
                    fprintf(stderr, "  Notice:     non-commercial use only — see upstream model card for terms.\n");
                } else if (strcmp(entry->license, "gemma") == 0) {
                    fprintf(stderr, "  Notice:     governed by Google's Gemma Terms of Use & Prohibited Use Policy.\n");
                } else {
                    fprintf(stderr, "  Notice:     review the upstream model card for the full license terms.\n");
                }
                fprintf(stderr, "Download %s (%s) and accept this license? [y/N] ", entry->filename,
                        entry->approx_size);
                char c = 0;
                if (scanf(" %c", &c) != 1 || (c != 'y' && c != 'Y')) {
                    return "";
                }
            } else {
                fprintf(stderr, "error: model '%s' is released under '%s' (restricted).\n", entry->name,
                        entry->license);
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
            fprintf(stderr, "  License: %s   (%s)\n", entry->license ? entry->license : "?",
                    entry->model_card_url ? entry->model_card_url : "");
            fprintf(stderr, "Download %s (%s) from HuggingFace? [y/N] ", entry->filename, entry->approx_size);
            char c = 0;
            if (scanf(" %c", &c) != 1 || (c != 'y' && c != 'Y')) {
                return "";
            }
        } else {
            fprintf(stderr, "Model '%s' not found. Use --auto-download to download automatically.\n", entry->name);
            return "";
        }
    }

    if (!download_supported()) {
        fprintf(stderr, "Model '%s' is not cached, and auto-download is unavailable on iOS builds.\n", entry->name);
        return "";
    }

    mkdirs(dir);
    fprintf(stderr, "Downloading %s (%s, license: %s)...\n", entry->filename, entry->approx_size,
            entry->license ? entry->license : "?");
    if (download_file(entry->url, cached)) {
        fprintf(stderr, "Downloaded to %s\n", cached.c_str());
        ensure_companion(entry, dir);
        return cached;
    } else {
        fprintf(stderr, "Download failed.\n");
        return "";
    }
}

void list_models() {
    fprintf(stderr, "Available models:\n");
    fprintf(stderr, "  %-40s %-14s %-9s %-22s %s\n", "Name", "License", "Size", "Scripts", "Description");
    fprintf(stderr, "  %-40s %-14s %-9s %-22s %s\n", "----", "-------", "----", "-------", "-----------");
    for (const ModelEntry * e = k_registry; e->name; e++) {
        std::string cached = cache_dir() + "/" + e->filename;
        const char * status = file_exists(cached) ? " [cached]" : "";
        const char * license = e->license ? e->license : "?";
        const char * marker = license_requires_acceptance(e->license) ? "*" : " ";
        // Blank for every non-OCR-recognizer row: the column states scanned
        // dictionary coverage, and an unscanned model must not read as one
        // with no coverage.
        const char * scripts = (e->languages && *e->languages) ? e->languages : "";
        fprintf(stderr, " %s%-40s %-14s %-9s %-22s %s%s\n", marker, e->name, license, e->approx_size, scripts, e->desc,
                status);
    }
    fprintf(stderr, "\n  * = restricted license (non-commercial or vendor terms); "
                    "requires --accept-license <spdx> or interactive consent.\n");
    fprintf(stderr, "  Scripts = characters the recognizer's dictionary can emit, scanned from the shipped\n"
                    "  GGUF (tools/scan_model_languages.py). Coverage is necessary, NOT sufficient, for\n"
                    "  quality; a blank means not scanned, not \"none\". See docs/LANGUAGES.md.\n");
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

const char * model_languages(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->languages;
    return nullptr;
}

const char * get_query_prefix(const char * model) {
    return query_prefix(model);
}
const char * get_passage_prefix(const char * model) {
    return passage_prefix(model);
}

} // namespace crispembed_mgr
