// Prepared-input, CPU-only Proofbird German GECToR v2.1 interface.
// This experimental library does not tokenize, decode edits or modify text.
#ifndef CRISPEMBED_GECTOR_H
#define CRISPEMBED_GECTOR_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(CRISPEMBED_GECTOR_SHARED)
# if defined(CRISPEMBED_GECTOR_BUILD)
#  define GECTOR_API __declspec(dllexport)
# else
#  define GECTOR_API __declspec(dllimport)
# endif
#else
# define GECTOR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct crispembed_gector_context crispembed_gector_context;

enum crispembed_gector_status {
    CRISPEMBED_GECTOR_OK = 0,
    CRISPEMBED_GECTOR_INVALID_ARGUMENT = 1,
    CRISPEMBED_GECTOR_DISABLED = 2,
    CRISPEMBED_GECTOR_MODEL_ERROR = 3,
    CRISPEMBED_GECTOR_RUNTIME_ERROR = 4,
    CRISPEMBED_GECTOR_ALLOCATION_ERROR = 5,
    CRISPEMBED_GECTOR_TRACE_ERROR = 6
};

enum {
    CRISPEMBED_GECTOR_MAX_TOKENS = 512,
    CRISPEMBED_GECTOR_VOCAB_SIZE = 31102,
    CRISPEMBED_GECTOR_HIDDEN_SIZE = 768,
    CRISPEMBED_GECTOR_LABEL_COUNT = 5000,
    CRISPEMBED_GECTOR_DETECT_COUNT = 2
};

// Model-free shape/value validation. Exactly one unpadded sequence is supported:
// 1..512 token IDs in [0,31102), mask_count == id_count, every mask value == 1.
// No implicit truncation, token insertion, padding or int64-to-int32 narrowing.
// A non-null error buffer receives a nul-terminated diagnostic when capacity>0.
GECTOR_API int crispembed_gector_validate_inputs(
    const int64_t * input_ids, size_t id_count,
    const int64_t * attention_mask, size_t mask_count, size_t batch_size,
    char * error, size_t error_capacity);

// Requires CRISPEMBED_GECTOR=1 (value-parsed, default off). Only the exact pinned
// source profile, its complete tensor census and supported precisions are admitted.
// n_threads must be 1..64. The context owns weights and cached graph workspace.
// Returns null on failure and reports the reason through error, if supplied.
GECTOR_API crispembed_gector_context * crispembed_gector_create(
    const char * model_path, int n_threads, char * error, size_t error_capacity);

// A null context is harmless. The caller must stop all uses before freeing it.
GECTOR_API void crispembed_gector_free(crispembed_gector_context * context);

// Returns zero for null, otherwise the fixed thread count used by this context.
GECTOR_API int crispembed_gector_threads(const crispembed_gector_context * context);

// Output arrays are caller-owned and token-major: [T,5000], [T,2], [T,768].
// Both logits are required. encoder_output is optional; use null with capacity 0
// when unwanted. Capacities are FLOAT ELEMENT COUNTS, never bytes. Output ranges
// must not overlap. Outputs are unchanged on error. Calls on one context serialize;
// independent contexts may run concurrently. No global tokenizer/model registry.
GECTOR_API int crispembed_gector_forward(
    crispembed_gector_context * context,
    const int64_t * input_ids, size_t id_count,
    const int64_t * attention_mask, size_t mask_count, size_t batch_size,
    float * label_logits, size_t label_capacity,
    float * detection_logits, size_t detection_capacity,
    float * encoder_output, size_t encoder_capacity,
    char * error, size_t error_capacity);

// Optional diagnostic callback. Values are contiguous token-major F32, valid
// only for the duration of the call. Stages: embedding, layer.0 ... layer.11,
// labels, detect. Return nonzero to abort delivery. Do not re-enter or destroy
// this context from the callback. Ordinary forward retains no per-layer outputs.
typedef int (*crispembed_gector_trace_callback)(
    void * user, const char * stage, const float * values, size_t rows, size_t columns);

GECTOR_API int crispembed_gector_forward_trace(
    crispembed_gector_context * context,
    const int64_t * input_ids, size_t id_count,
    const int64_t * attention_mask, size_t mask_count, size_t batch_size,
    float * label_logits, size_t label_capacity,
    float * detection_logits, size_t detection_capacity,
    float * encoder_output, size_t encoder_capacity,
    crispembed_gector_trace_callback trace, void * trace_user,
    char * error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
