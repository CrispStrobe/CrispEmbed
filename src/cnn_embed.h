// cnn_embed.h — CNN image encoder for face detection/recognition models.
//
// Loads a GGUF produced by convert-face-to-gguf.py and replays the
// ONNX graph using ggml ops. Supports Conv2d, Conv2d_dw, ReLU, PReLU,
// Add (residual), Pool, Flatten, Gemm (FC), BatchNorm.
//
// Usage:
//   cnn_embed::context ctx;
//   if (!cnn_embed::load(&ctx, "sface.gguf")) return 1;
//
//   // For face recognition: aligned 112×112 face crop
//   auto emb = cnn_embed::encode(&ctx, pixels, 112, 112);
//   // emb.size() == 128 (SFace) or 512 (AuraFace)
//
//   // For face detection: full image
//   auto dets = cnn_embed::detect(&ctx, pixels, 640, 640);
//   // dets: vector of {x, y, w, h, confidence, landmarks[10]}

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cnn_embed {

struct face_detection {
    float x, y, w, h;        // bounding box
    float confidence;
    float landmarks[10];     // 5 points × (x, y)
};

struct context;

// Load CNN GGUF. Returns true on success.
bool load(context** ctx, const char* path, int n_threads = 4);

// Encode a face image (recognition). pixels: [3, H, W] CHW float32.
// Returns embedding vector (128-D for SFace, 512-D for AuraFace).
std::vector<float> encode(context* ctx, const float* pixels, int H, int W);

// Detect faces (detection). Returns bounding boxes + landmarks.
std::vector<face_detection> detect(context* ctx, const float* pixels, int H, int W,
                                    float conf_threshold = 0.5f);

// Encode from image file (loads + resizes + normalizes).
std::vector<float> encode_file(context* ctx, const char* path);

// Get embedding dimension (recognition models only).
int dim(const context* ctx);

// Get model type: "detection" or "recognition".
const char* model_type(const context* ctx);

// Free resources.
void free(context* ctx);

} // namespace cnn_embed
