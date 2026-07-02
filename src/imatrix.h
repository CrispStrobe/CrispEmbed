// imatrix.h — importance-matrix (activation-statistics) collector for CrispEmbed.
//
// When the environment variable CRISPEMBED_IMATRIX_OUT is set to a file path,
// installing the collector on a ggml scheduler makes every ggml_mul_mat node
// whose src0 is a model weight accumulate the per-column sum-of-squares of its
// activation input (src1). At process exit the accumulated statistics are merged
// with any pre-existing file and written back, so a calibration corpus can be
// streamed across many CLI invocations that all point at the same output file.
//
// The resulting file is consumed by crispembed-quantize --imatrix <file>, which
// feeds the per-tensor importance vector to ggml_quantize_chunk so that k-quant
// / IQ-quant error is minimised on the weights the calibration data exercised.
//
// This is a no-op unless CRISPEMBED_IMATRIX_OUT is set; production paths are
// unaffected.

#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Install the imatrix eval-callback on a scheduler. Safe no-op when
// CRISPEMBED_IMATRIX_OUT is unset. Call once, right after ggml_backend_sched_new.
void crispembed_imatrix_install(ggml_backend_sched_t sched);

// Merge the accumulated statistics with any existing output file and write it.
// Registered via atexit() on first install; may also be called explicitly.
void crispembed_imatrix_flush(void);

#ifdef __cplusplus
}
#endif
