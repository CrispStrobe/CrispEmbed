// Native Camel Morph / camel_tools morphological analyzer (C ABI).
//
// A C++ port of the camel_tools (MIT, (c) 2018-2026 New York University Abu
// Dhabi) MorphologyDB loader and Analyzer for ALMOR-format databases such as
// Camel Morph MSA v1.0 (CC BY 4.0 data, downloaded separately as a bundle).
// The analyzer returns every analysis with every feature, in camel_tools'
// order, as a JSON array of objects. See src/arabic_morph.cpp for the port
// notes and docs/validation/camel-morph-msa-20261119/README.md for the
// exactness evidence.
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

#define CRISPEMBED_AR_MORPH_ABI_VERSION 1
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
CRISPEMBED_AR_MORPH_API crispembed_ar_morph * crispembed_ar_morph_open(const char * manifest_path,
                                                                       const char * manifest_sha256, int backoff,
                                                                       char * error, size_t error_capacity);

// Opens a bare ALMOR database whose lowercase SHA-256 the caller trusts.
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

// JSON description of the loaded database (variant, hashes, counts, backoff,
// Unicode table version). Owned by the context; valid until free.
CRISPEMBED_AR_MORPH_API const char * crispembed_ar_morph_info(const crispembed_ar_morph * morph);

CRISPEMBED_AR_MORPH_API void crispembed_ar_morph_free(crispembed_ar_morph * morph);

#ifdef __cplusplus
}
#endif

#endif // CRISPEMBED_ARABIC_MORPH_H
