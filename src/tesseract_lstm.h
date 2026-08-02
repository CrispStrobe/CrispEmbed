// tesseract_lstm.h — Tesseract LSTM line-recognition engine via ggml.
//
// Loads a GGUF model produced by convert-tesseract-to-gguf.py.
// Runs the VGSL forward pass (Convolve stacking → FC+tanh → MaxPool →
// SummLSTM → LSTMs → Softmax → CTC decode) on a pre-cropped text line.
//
// Supports 126 languages via tessdata_best/tessdata_fast .traineddata files.
// Typical model size: ~2-4 MB Q4_K, ~1.5 MB F16.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tesseract_lstm_context tesseract_lstm_context;

/// Load a Tesseract LSTM GGUF model. Returns NULL on failure.
tesseract_lstm_context * tesseract_lstm_init(const char * model_path, int n_threads);

/// Free the context and all associated memory.
void tesseract_lstm_free(tesseract_lstm_context * ctx);

/// Recognize a single text line image (grayscale uint8, height-normalized).
///
/// [pixels]   — row-major grayscale pixels, [0..255].
/// [width]    — image width in pixels.
/// [height]   — image height in pixels (will be resized to model height if needed).
/// [out_len]  — if non-NULL, receives the length of the returned string.
///
/// Returns a null-terminated UTF-8 string owned by the context.
/// Valid until the next call to tesseract_lstm_recognize or tesseract_lstm_free.
/// Spaces are NOT produced by the LSTM — they come from Tesseract's word
/// segmentation which is not part of this engine.
const char * tesseract_lstm_recognize(tesseract_lstm_context * ctx, const uint8_t * pixels, int width, int height,
                                      int * out_len);

/// Get per-character confidence scores from the last recognition.
/// Returns a float array of length *n_chars, or NULL if no recognition
/// has been performed. Each value is the softmax probability of the
/// winning character class at its CTC timestep.
const float * tesseract_lstm_confidences(const tesseract_lstm_context * ctx, int * n_chars);

/// Get sequence-level confidence for the last recognition. Greedy decoding
/// returns the mean selected CTC probability; beam decoding returns a
/// length-normalized sequence probability. Beam per-character confidence is
/// intentionally unavailable because its labels have no single timestep.
float tesseract_lstm_mean_confidence(const tesseract_lstm_context * ctx);

/// Tesseract-style greedy word confidence in [0,1]. This uses the minimum
/// log-probability over the selected CTC path, matching NetworkIO's certainty
/// conversion and LTRResultIterator's 100 + 5*certainty rule. Returns 0 for
/// beam decoding because a beam path has no stable per-symbol certainty.
float tesseract_lstm_word_confidence(const tesseract_lstm_context * ctx);

/// Get model info.
int tesseract_lstm_input_height(const tesseract_lstm_context * ctx);
int tesseract_lstm_num_classes(const tesseract_lstm_context * ctx);
const char * tesseract_lstm_vgsl_spec(const tesseract_lstm_context * ctx);
/// Number of optional DAWG graphs loaded from GGUF metadata. Loading is
/// enabled only with CRISPEMBED_TESSERACT_DAWG_LOAD.
int tesseract_lstm_dawg_count(const tesseract_lstm_context * ctx);

/// Query a loaded DAWG by name using unicharset IDs. Returns 1 for a match,
/// 0 for a non-match, and -1 when the graph is not loaded or arguments are
/// invalid. `complete=0` accepts a valid prefix; `complete=1` requires a word.
int tesseract_lstm_dawg_matches(const tesseract_lstm_context * ctx, const char * name, const int * unichars,
                                int n_unichars, int complete);

/// UTF-8 convenience wrapper around tesseract_lstm_dawg_matches(). It
/// segments the string against the loaded tokenizer tokens and returns the
/// same 1/0/-1 status.
int tesseract_lstm_dawg_matches_utf8(const tesseract_lstm_context * ctx, const char * name, const char * text,
                                     int complete);

/// Number of preserved DAWG components in the GGUF metadata. This is
/// provenance/introspection only; dictionary scoring is not enabled.
int tesseract_lstm_dawg_component_count(const tesseract_lstm_context * ctx);
int tesseract_lstm_dawg_contains(const tesseract_lstm_context * ctx, const char * component, const int * unichar_ids,
                                 size_t count);
int tesseract_lstm_dawg_has_prefix(const tesseract_lstm_context * ctx, const char * component, const int * unichar_ids,
                                   size_t count);

/// Enable/disable capture of per-stage intermediates for parity testing.
void tesseract_lstm_set_dump(tesseract_lstm_context * ctx, int enabled);

/// Get a captured intermediate buffer by name (e.g. "after_conv_fc").
/// Returns NULL if dump mode is off or the name is unknown.
/// Valid until the next call to tesseract_lstm_recognize.
const float * tesseract_lstm_get_capture(const tesseract_lstm_context * ctx, const char * name, int * n_elem);

#ifdef __cplusplus
}
#endif
