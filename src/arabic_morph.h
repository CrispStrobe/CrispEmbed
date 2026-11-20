// Native Camel Morph / camel_tools morphological analyzer (C ABI).
//
// A C++ port of the camel_tools (MIT, (c) 2018-2026 New York University Abu
// Dhabi) MorphologyDB loader and Analyzer for ALMOR-format databases such as
// Camel Morph MSA v1.0 (CC BY 4.0 data, downloaded separately as a bundle).
// The analyzer returns every analysis with every feature, in camel_tools'
// order, as a JSON array of objects, or as a compact binary buffer. A
// version-2 bundle holds the binary database format
// (docs/arabic-morph-binary-format.md): memory-mapped, and reduced to a
// feature profile, so analyses carry exactly camel_tools' values of those
// features. See src/arabic_morph.cpp for the port notes and
// docs/validation/camel-morph-msa-20261119/ and
// docs/validation/camel-morph-binary-20261119/ for the exactness evidence.
#ifndef CRISPEMBED_ARABIC_MORPH_H
#define CRISPEMBED_ARABIC_MORPH_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(CRISPEMBED_AR_MORPH_SHARED)
#if defined(CRISPEMBED_AR_MORPH_BUILD)
#define CRISPEMBED_AR_MORPH_API __declspec(dllexport)
#else
#define CRISPEMBED_AR_MORPH_API __declspec(dllimport)
#endif
#else
#define CRISPEMBED_AR_MORPH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Version 2 adds crispembed_ar_morph_analyze_compact, crispembed_ar_morph_buffer_free
// and binary (format_version 2) bundles; every version-1 function is unchanged.
#define CRISPEMBED_AR_MORPH_ABI_VERSION 2
// Largest accepted word in bytes, before stripping.
#define CRISPEMBED_AR_MORPH_MAX_WORD_BYTES 4096u

typedef struct crispembed_ar_morph crispembed_ar_morph;

enum crispembed_ar_morph_status {
    CRISPEMBED_AR_MORPH_OK = 0,
    // CRISPEMBED_AR_MORPH=0 (or false/off/no) disables the service.
    CRISPEMBED_AR_MORPH_DISABLED = 1,
    CRISPEMBED_AR_MORPH_INVALID_ARGUMENT = 2,
    // Hash, size, manifest or database format check failed.
    CRISPEMBED_AR_MORPH_DATA_ERROR = 3,
    CRISPEMBED_AR_MORPH_ALLOCATION_ERROR = 4,
    // Invalid UTF-8, an over-long word, or an input for which camel_tools
    // itself raises (e.g. an invalid re.sub template in backoff mode).
    CRISPEMBED_AR_MORPH_INPUT_ERROR = 5,
    CRISPEMBED_AR_MORPH_RUNTIME_ERROR = 6
};

// camel_tools Analyzer backoff modes; NONE is camel_tools' default.
enum crispembed_ar_morph_backoff {
    CRISPEMBED_AR_MORPH_BACKOFF_NONE = 0,
    CRISPEMBED_AR_MORPH_BACKOFF_NOAN_ALL = 1,
    CRISPEMBED_AR_MORPH_BACKOFF_NOAN_PROP = 2,
    CRISPEMBED_AR_MORPH_BACKOFF_ADD_ALL = 3,
    CRISPEMBED_AR_MORPH_BACKOFF_ADD_PROP = 4
};

CRISPEMBED_AR_MORPH_API uint32_t crispembed_ar_morph_abi_version(void);

// Opens a bundle through its manifest.json. manifest_sha256 is the externally
// trusted lowercase SHA-256 of manifest.json; every file the manifest lists
// (database, NOTICE, LICENSE) must sit next to it and match its size and
// SHA-256 before the database is parsed. Never downloads. Returns null on
// error; errors are NUL-terminated when error_capacity is nonzero.
// format_version 1 bundles hold the ALMOR text database (parsed into heap
// tables); format_version 2 bundles hold the binary database, which is
// memory-mapped (CRISPEMBED_AR_MORPH_MMAP=0, and Emscripten, read it into one
// heap buffer instead) and used in place.
CRISPEMBED_AR_MORPH_API crispembed_ar_morph * crispembed_ar_morph_open(const char * manifest_path,
                                                                       const char * manifest_sha256, int backoff,
                                                                       char * error, size_t error_capacity);

// Opens a bare ALMOR text or binary database (detected by its magic) whose
// lowercase SHA-256 the caller trusts.
CRISPEMBED_AR_MORPH_API crispembed_ar_morph * crispembed_ar_morph_open_db(const char * db_path, const char * db_sha256,
                                                                          int backoff, char * error,
                                                                          size_t error_capacity);

// Analyzes one word (UTF-8, word_len bytes; embedded NUL is allowed). On
// success *json receives a NUL-terminated JSON array with one object per
// analysis, keys in camel_tools' insertion order, string/null/float values
// exactly as camel_tools' dicts hold them; free it with
// crispembed_ar_morph_string_free. On error *json is null. The context is
// immutable after open, so concurrent calls on one context are safe.
CRISPEMBED_AR_MORPH_API int crispembed_ar_morph_analyze(const crispembed_ar_morph * morph, const char * word,
                                                        size_t word_len, char ** json, size_t * json_len, char * error,
                                                        size_t error_capacity);

CRISPEMBED_AR_MORPH_API void crispembed_ar_morph_string_free(char * text);

// Compact output: no JSON, one call per batch of words.
#define CRISPEMBED_AR_MORPH_COMPACT_MAGIC 0x31434D41u /* "AMC1" little-endian */
// Drop analyses whose (key, value) sequence repeats an earlier one of the same word.
#define CRISPEMBED_AR_MORPH_COMPACT_DEDUP 1u

// Analyzes n_words words given as concatenated UTF-8 bytes (`words`) and their
// byte lengths (`word_lens`). On success *out receives one malloc'ed buffer
// (free with crispembed_ar_morph_buffer_free), all integers little-endian:
//
//   u32 magic (CRISPEMBED_AR_MORPH_COMPACT_MAGIC), u32 n_words, then per word:
//     u32 status   0 = OK, CRISPEMBED_AR_MORPH_INPUT_ERROR = rejected word
//     u32 bytes    size of the record that follows (0 when rejected)
//     record:
//       u32 n_analyses, u32 n_strings, u8 wide, u8[3] zero
//       n_strings x { u8 kind (0 string, 1 float as Python repr), u32 length, UTF-8 bytes }
//       n_analyses x { u16 n_features, n_features x feature }
//       feature: wide == 0: u8 key id, u16 string index (0xFFFF = null)
//                wide == 1: u16 key id, u32 string index (0xFFFFFFFF = null)
//
// Key ids index the "keys" array of crispembed_ar_morph_info; only output
// features appear (the profile of a binary database). Strings are unique per
// word. Features, analyses and values are those of crispembed_ar_morph_analyze,
// in the same order, except that COMPACT_DEDUP drops repeated analyses. Any
// error other than a rejected word fails the whole call (*out stays null).
CRISPEMBED_AR_MORPH_API int crispembed_ar_morph_analyze_compact(const crispembed_ar_morph * morph, const char * words,
                                                                const uint32_t * word_lens, uint32_t n_words,
                                                                uint32_t flags, uint8_t ** out, size_t * out_len,
                                                                char * error, size_t error_capacity);

CRISPEMBED_AR_MORPH_API void crispembed_ar_morph_buffer_free(uint8_t * buffer);

// JSON description of the loaded database (variant, db_format, hashes, counts,
// backoff, Unicode table version, storage "mapped"/"heap", "keys" by key id and
// the output "profile", null for a text database). Owned by the context;
// valid until free.
CRISPEMBED_AR_MORPH_API const char * crispembed_ar_morph_info(const crispembed_ar_morph * morph);

CRISPEMBED_AR_MORPH_API void crispembed_ar_morph_free(crispembed_ar_morph * morph);

#ifdef __cplusplus
}
#endif

#endif // CRISPEMBED_ARABIC_MORPH_H
