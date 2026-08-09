#ifndef FIREREDPUNC_H
#define FIREREDPUNC_H

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

struct fireredpunc_context;

// Load a FireRedPunc GGUF model.
struct fireredpunc_context * fireredpunc_init(const char * model_path);

// Add punctuation to unpunctuated text. Returns newly allocated string (caller frees).
char * fireredpunc_process(struct fireredpunc_context * ctx, const char * text);

// Free context.
void fireredpunc_free(struct fireredpunc_context * ctx);

// Parity hook: the token ids this engine feeds the model for `text`. Its
// tokenizer is private to fireredpunc.cpp (a second WordPiece implementation,
// separate from the shared one in tokenizer.h), so this is how
// tests/firered_tokenizer_parity.py compares it against HuggingFace. The
// pointer is owned by the context and valid until the next call.
const int * fireredpunc_debug_token_ids(struct fireredpunc_context * ctx, const char * text, int * out_n);

#ifdef __cplusplus
}
#endif

#endif // FIREREDPUNC_H
