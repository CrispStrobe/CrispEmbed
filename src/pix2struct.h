// pix2struct.h -- Pix2Struct image-to-text (Google, Apache-2.0).
//
// Variable-resolution ViT encoder + T5-style decoder. Handles documents,
// charts, tables, infographics at any aspect ratio. 282M params (base).
// Source: google/pix2struct-base.

#ifndef PIX2STRUCT_H
#define PIX2STRUCT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pix2struct_context pix2struct_context;

pix2struct_context * pix2struct_init(const char * model_path, int n_threads);
void pix2struct_free(pix2struct_context * ctx);

/// Process image and generate text output.
/// Input: uint8 RGB [h, w, 3].
/// Output: UTF-8 text (caller frees with pix2struct_free_text).
/// max_tokens: maximum number of tokens to generate (0 = default 256).
const char * pix2struct_generate(pix2struct_context * ctx,
                                 const uint8_t * image, int width, int height,
                                 int max_tokens);

/// Free text returned by pix2struct_generate.
void pix2struct_free_text(const char * text);

/// Process pre-computed patches (for parity testing).
/// patches: [n_patches, 770] float (row_id, col_id, 768 pixel values).
/// Returns encoder output [n_patches, hidden_size].
const float * pix2struct_encode_patches(pix2struct_context * ctx,
                                         const float * patches, int n_patches,
                                         int * out_dim);

#ifdef __cplusplus
}
#endif

#endif
