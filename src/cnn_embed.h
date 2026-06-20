// cnn_embed.h — CNN face detection & recognition via ggml.
//
// Loads GGUF models (from convert-face-to-gguf.py) and replays ONNX
// graphs. Supports full face pipeline: detect → align → encode.
//
// Usage:
//   // Detection + recognition pipeline
//   cnn_embed::context *det, *rec;
//   cnn_embed::load(&det, "scrfd.gguf");
//   cnn_embed::load(&rec, "arcface.gguf");
//   auto results = cnn_embed::face_pipeline(det, rec, "photo.jpg", 0.5f);
//   // results[i].embedding = 512-D L2-normalized vector

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cnn_embed {

struct face_detection {
    float x, y, w, h;        // bounding box in original image coordinates
    float confidence;
    float landmarks[10];     // 5 points × (x, y) in original image coords
};

struct face_result {
    face_detection det;
    std::vector<float> embedding;
};

struct context;

// Load CNN GGUF. Returns true on success.
bool load(context** ctx, const char* path, int n_threads = 1);

// Encode a face image (recognition). pixels: [3, H, W] CHW float32.
// Returns embedding vector (128-D for SFace, 512-D for AuraFace).
std::vector<float> encode(context* ctx, const float* pixels, int H, int W);

// Detect faces in preprocessed pixels [3, H, W]. Coordinates are in
// the pixel space of the input (i.e. 0..H, 0..W). Use detect_file()
// for automatic letterbox + rescaling to original image coordinates.
std::vector<face_detection> detect(context* ctx, const float* pixels, int H, int W,
                                    float conf_threshold = 0.5f);

// Detect faces from image file. Handles letterbox resize to det_size
// (default 640) and scales coordinates back to original image space.
std::vector<face_detection> detect_file(context* ctx, const char* path,
                                         float conf_threshold = 0.5f,
                                         int det_size = 640);

// Encode from image file (loads + resizes + normalizes).
// For face recognition, prefer encode_aligned() with proper landmarks.
std::vector<float> encode_file(context* ctx, const char* path);

// Encode an aligned face crop. Takes raw RGB uint8 image + 5-point
// landmarks (from detection), performs similarity-transform alignment
// to 112×112, normalizes, and runs the recognition model.
std::vector<float> encode_aligned(context* ctx,
                                   const unsigned char* image, int img_w, int img_h,
                                   const float* landmarks_10);

// Like encode_aligned but loads the image from a file path.
std::vector<float> encode_face_file(context* ctx, const char* image_path,
                                     const float* landmarks_10);

// Full pipeline: detect → align → encode. Takes a detector context
// and recognition context, runs the complete face pipeline on an
// image file. Returns face detections with embeddings.
std::vector<face_result> face_pipeline(context* det_ctx, context* rec_ctx,
                                        const char* image_path,
                                        float conf_threshold = 0.5f,
                                        int det_size = 640);

// Get embedding dimension (recognition models only).
int dim(const context* ctx);

// Get model type: "detection" or "recognition".
const char* model_type(const context* ctx);

// Free resources.
void free(context* ctx);

} // namespace cnn_embed
