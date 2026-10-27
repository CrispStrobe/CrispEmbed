#pragma once
#include "crispembed.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Separate raw-text/role ABI. No generic dense pooling or token dropping.
typedef struct crispembed_moedern_context crispembed_moedern_context;
enum { CRISPEMBED_MOEDERN_QUERY = 0, CRISPEMBED_MOEDERN_DOCUMENT = 1 };
typedef struct crispembed_moedern_result {
    uint32_t abi_version, struct_bytes;
    int32_t role, n_tokens, n_selected, dimension, expansion_count;
    const int32_t * ids;
    const uint8_t * supplied_mask;
    const uint8_t * effective_mask;
    const uint8_t * scoring_mask;
    const uint8_t * expansion_positions; // NULL/count0 for documents.
    const float * all_vectors;           // [n_tokens,128], row-major.
    const float * selected_vectors;      // [n_selected,128], row-major.
} crispembed_moedern_result;
// Local self-contained packed F32 GGUF; threads0=>2, otherwise1..2.
CRISPEMBED_API crispembed_moedern_context * crispembed_moedern_init(const char *, int threads);
CRISPEMBED_API void crispembed_moedern_free(crispembed_moedern_context *);
// NUL-free UTF8 text, <=64KiB. Borrowed result/buffers expire at next encode,
// score or free, including failed calls. One call per context at a time.
CRISPEMBED_API const crispembed_moedern_result * crispembed_moedern_encode(crispembed_moedern_context *, const char *,
                                                                           int role);
// Raw text on both sides, no vector/ID injection. Returns1 on success; failure
// clears *score when non-NULL. All context result buffers expire on this call.
CRISPEMBED_API int crispembed_moedern_score(crispembed_moedern_context *, const char * query, const char * document,
                                            float * score);
// NULL context returns this thread's last initialization/invalid-handle error.
CRISPEMBED_API const char * crispembed_moedern_error(const crispembed_moedern_context *);
#ifdef __cplusplus
}
#endif
