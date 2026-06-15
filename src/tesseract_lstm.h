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
const char * tesseract_lstm_recognize(
    tesseract_lstm_context * ctx,
    const uint8_t * pixels,
    int width, int height,
    int * out_len
);

/// Get per-character confidence scores from the last recognition.
/// Returns a float array of length *n_chars, or NULL if no recognition
/// has been performed. Each value is the softmax probability of the
/// winning character class at its CTC timestep.
const float * tesseract_lstm_confidences(
    const tesseract_lstm_context * ctx,
    int * n_chars
);

/// Get model info.
int tesseract_lstm_input_height(const tesseract_lstm_context * ctx);
int tesseract_lstm_num_classes(const tesseract_lstm_context * ctx);
const char * tesseract_lstm_vgsl_spec(const tesseract_lstm_context * ctx);

#ifdef __cplusplus
}
#endif
