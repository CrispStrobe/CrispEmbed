// face_align.cpp — 5-point similarity transform for face alignment.
//
// Implements the standard ArcFace face alignment:
// 1. Compute similarity transform from 5 detected landmarks to reference
// 2. Apply affine warp to produce aligned 112×112 crop
// 3. Normalize to (x-127.5)/127.5 for ArcFace/SFace input

#include "face_align.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace face_align {

// Estimate similarity transform: src → dst
// Solves the least-squares problem for [s*cos(θ), -s*sin(θ), tx; s*sin(θ), s*cos(θ), ty]
void estimate_affine(const float* src, const float* dst, float matrix[6]) {
    // Similarity transform: 4 unknowns (a, b, tx, ty)
    // dst_x = a * src_x - b * src_y + tx
    // dst_y = b * src_x + a * src_y + ty
    //
    // Solve via normal equations (5 points → overdetermined)
    int n = 5;
    double A[4][4] = {}, B[4] = {};

    for (int i = 0; i < n; i++) {
        double sx = src[i*2], sy = src[i*2+1];
        double dx = dst[i*2], dy = dst[i*2+1];

        // Design matrix rows: [sx, -sy, 1, 0] (x-eq) and [sy, sx, 0, 1] (y-eq)
        // Normal equations: A^T A x = A^T b
        A[0][0] += sx*sx + sy*sy;
        A[0][1] += 0;
        A[0][2] += sx;
        A[0][3] += sy;
        B[0] += sx*dx + sy*dy;

        A[1][0] += 0;
        A[1][1] += sx*sx + sy*sy;
        A[1][2] += -sy;
        A[1][3] += sx;
        B[1] += sx*dy - sy*dx;

        A[2][0] += sx;
        A[2][1] += -sy;
        A[2][2] += 1;
        A[2][3] += 0;
        B[2] += dx;

        A[3][0] += sy;
        A[3][1] += sx;
        A[3][2] += 0;
        A[3][3] += 1;
        B[3] += dy;
    }

    // Solve 4×4 system via Gaussian elimination
    double aug[4][5];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) aug[i][j] = A[i][j];
        aug[i][4] = B[i];
    }

    for (int col = 0; col < 4; col++) {
        // Pivot
        int mx = col;
        for (int row = col+1; row < 4; row++)
            if (std::fabs(aug[row][col]) > std::fabs(aug[mx][col])) mx = row;
        std::swap(aug[col], aug[mx]);

        if (std::fabs(aug[col][col]) < 1e-12) continue;

        for (int row = col+1; row < 4; row++) {
            double f = aug[row][col] / aug[col][col];
            for (int j = col; j < 5; j++) aug[row][j] -= f * aug[col][j];
        }
    }

    // Back-substitute
    double x[4] = {};
    for (int i = 3; i >= 0; i--) {
        x[i] = aug[i][4];
        for (int j = i+1; j < 4; j++) x[i] -= aug[i][j] * x[j];
        x[i] /= aug[i][i];
    }

    // matrix = [a, -b, tx; b, a, ty]
    matrix[0] = (float)x[0];   // a = s*cos(θ)
    matrix[1] = (float)(-x[1]); // -b = -s*sin(θ)
    matrix[2] = (float)x[2];   // tx
    matrix[3] = (float)x[1];   // b = s*sin(θ)
    matrix[4] = (float)x[0];   // a = s*cos(θ)
    matrix[5] = (float)x[3];   // ty
}

std::vector<float> align(
    const unsigned char* image, int img_w, int img_h,
    const float* landmarks,
    int out_w, int out_h)
{
    const bool bench = (std::getenv("CRISPEMBED_FACE_ALIGN_BENCH") != nullptr);
    auto t_total = std::chrono::steady_clock::now();

    // Compute affine transform: landmarks → reference points
    float M[6];
    auto t_aff0 = std::chrono::steady_clock::now();
    estimate_affine(landmarks, REF_PTS_112, M);
    if (bench) {
        auto t_aff1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[face_align-bench] estimate_affine: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_aff1 - t_aff0).count());
    }

    // Invert: we need src→dst mapping, but warpAffine uses dst→src
    // For similarity transform [a -b tx; b a ty]:
    // Inverse: [a b -(a*tx+b*ty); -b a (b*tx-a*ty)] / (a²+b²)
    float a = M[0], b = M[3], tx = M[2], ty = M[5];
    float det = a*a + b*b;
    float inv[6];
    inv[0] = a / det;
    inv[1] = b / det;
    inv[2] = -(a*tx + b*ty) / det;
    inv[3] = -b / det;
    inv[4] = a / det;
    inv[5] = (b*tx - a*ty) / det;

    // Warp: for each output pixel, sample from input
    auto t_warp0 = std::chrono::steady_clock::now();
    std::vector<float> result(3 * out_h * out_w);

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            // Map dst→src using inverse
            float sx = inv[0] * x + inv[1] * y + inv[2];
            float sy = inv[3] * x + inv[4] * y + inv[5];

            // Bilinear interpolation
            int x0 = (int)sx, y0 = (int)sy;
            float fx = sx - x0, fy = sy - y0;

            for (int c = 0; c < 3; c++) {
                float v = 0;
                auto sample = [&](int px, int py) -> float {
                    if (px < 0 || px >= img_w || py < 0 || py >= img_h) return 127.5f;
                    return (float)image[(py * img_w + px) * 3 + c];
                };

                v = sample(x0, y0) * (1-fx) * (1-fy)
                  + sample(x0+1, y0) * fx * (1-fy)
                  + sample(x0, y0+1) * (1-fx) * fy
                  + sample(x0+1, y0+1) * fx * fy;

                // Normalize for ArcFace: (pixel - 127.5) / 127.5
                result[c * out_h * out_w + y * out_w + x] = (v - 127.5f) / 127.5f;
            }
        }
    }

    if (bench) {
        auto t_warp1 = std::chrono::steady_clock::now();
        auto t_total1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[face_align-bench] warp: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_warp1 - t_warp0).count());
        fprintf(stderr, "[face_align-bench] total: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_total1 - t_total).count());
    }

    return result;
}

std::vector<float> align_float(
    const float* image_chw, int img_w, int img_h,
    const float* landmarks,
    int out_w, int out_h)
{
    // For pre-normalized float input, same warp but no normalization
    float M[6];
    estimate_affine(landmarks, REF_PTS_112, M);

    float a = M[0], b = M[3], tx = M[2], ty = M[5];
    float det = a*a + b*b;
    float inv[6];
    inv[0] = a / det;  inv[1] = b / det;  inv[2] = -(a*tx + b*ty) / det;
    inv[3] = -b / det; inv[4] = a / det;  inv[5] = (b*tx - a*ty) / det;

    std::vector<float> result(3 * out_h * out_w, 0.0f);

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float sx = inv[0] * x + inv[1] * y + inv[2];
            float sy = inv[3] * x + inv[4] * y + inv[5];
            int x0 = (int)sx, y0 = (int)sy;
            float fx = sx - x0, fy = sy - y0;

            for (int c = 0; c < 3; c++) {
                auto sample = [&](int px, int py) -> float {
                    if (px < 0 || px >= img_w || py < 0 || py >= img_h) return 0.0f;
                    return image_chw[c * img_h * img_w + py * img_w + px];
                };
                result[c * out_h * out_w + y * out_w + x] =
                    sample(x0, y0) * (1-fx) * (1-fy)
                  + sample(x0+1, y0) * fx * (1-fy)
                  + sample(x0, y0+1) * (1-fx) * fy
                  + sample(x0+1, y0+1) * fx * fy;
            }
        }
    }
    return result;
}

} // namespace face_align
