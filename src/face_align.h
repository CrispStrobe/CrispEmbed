// face_align.h — 5-point face alignment for ArcFace-style recognition.
//
// Takes a detected face with 5 landmarks (left eye, right eye, nose,
// left mouth, right mouth) and produces an aligned 112×112 crop via
// affine transformation.
//
// Usage:
//   float landmarks[10] = { ... };  // from SCRFD
//   std::vector<float> aligned = face_align::align(
//       image_data, img_w, img_h, landmarks, 112, 112);
//   // aligned: [3, 112, 112] CHW float32, normalized (x-127.5)/127.5

#pragma once

#include <vector>

namespace face_align {

// ArcFace standard 112×112 reference landmarks
// (left eye, right eye, nose, left mouth, right mouth)
static const float REF_PTS_112[10] = {
    38.2946f, 51.6963f,   // left eye
    73.5318f, 51.5014f,   // right eye
    56.0252f, 71.7366f,   // nose tip
    41.5493f, 92.3655f,   // left mouth corner
    70.7299f, 92.2041f,   // right mouth corner
};

// Compute similarity transform (scale + rotation + translation)
// from src_pts[10] → dst_pts[10] (5 points × 2 coords each).
// Returns 2×3 affine matrix [a b tx; c d ty].
void estimate_affine(const float* src, const float* dst, float matrix[6]);

// Align a face from an RGB image using 5 landmarks.
// Input: image [W×H×3] uint8 or float, landmarks[10] in image coords.
// Output: [3, out_h, out_w] CHW float32, normalized (x-127.5)/127.5.
std::vector<float> align(
    const unsigned char* image, int img_w, int img_h,
    const float* landmarks,
    int out_w = 112, int out_h = 112);

// Align from CHW float input (already preprocessed)
std::vector<float> align_float(
    const float* image_chw, int img_w, int img_h,
    const float* landmarks,
    int out_w = 112, int out_h = 112);

} // namespace face_align
