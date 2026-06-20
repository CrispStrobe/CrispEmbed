// tps_warp.cpp — Thin-Plate Spline spatial transformer.
//
// Solves the TPS interpolation system and applies the warp to an image.
//
// The TPS function for 2D point mapping is:
//   f(p) = a_0 + a_1*x + a_2*y + sum_i( w_i * U(|p - c_i|) )
// where U(r) = r^2 * ln(r) is the TPS radial basis function (r > 0),
// c_i are the control points, w_i are the basis weights, and a_0..a_2
// are affine coefficients.
//
// The system is solved by constructing the (N+3) x (N+3) linear system:
//   [ K  P ] [ w ]   [ v ]
//   [ P' 0 ] [ a ] = [ 0 ]
// where K[i,j] = U(|c_i - c_j|), P = [1, x_i, y_i], and v = target coords.
// Solved separately for x-target and y-target to get two sets of coefficients.

#include "tps_warp.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// TPS model structure
// ---------------------------------------------------------------------------

struct tps_model {
    int n;                     // number of control points
    std::vector<float> cx, cy; // source control point coords (length n)
    std::vector<float> wx, wy; // radial basis weights for x, y mapping (length n)
    float ax[3], ay[3];        // affine coefficients: [const, x_coeff, y_coeff]
};

// ---------------------------------------------------------------------------
// TPS radial basis function: U(r) = r^2 * ln(r)
// ---------------------------------------------------------------------------
// Convention: U(0) = 0 (limit as r→0 of r^2*ln(r) = 0).

static inline float tps_basis(float r) {
    if (r < 1e-10f) return 0.0f;
    return r * r * std::log(r);
}

// ---------------------------------------------------------------------------
// Solve (N+3) x (N+3) linear system via Gaussian elimination with partial
// pivoting. The system has M equations and `nrhs` right-hand sides.
// ---------------------------------------------------------------------------
// A is row-major M x M, B is row-major M x nrhs.
// On success, B is overwritten with the solution. Returns true on success.

static bool solve_linear(std::vector<double> & A, std::vector<double> & B,
                         int M, int nrhs) {
    for (int col = 0; col < M; col++) {
        // Partial pivoting
        int pivot = col;
        double best = std::abs(A[col * M + col]);
        for (int r = col + 1; r < M; r++) {
            double v = std::abs(A[r * M + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-12) return false; // singular

        if (pivot != col) {
            for (int c = col; c < M; c++)
                std::swap(A[col * M + c], A[pivot * M + c]);
            for (int c = 0; c < nrhs; c++)
                std::swap(B[col * nrhs + c], B[pivot * nrhs + c]);
        }

        double div = A[col * M + col];
        for (int c = col; c < M; c++) A[col * M + c] /= div;
        for (int c = 0; c < nrhs; c++) B[col * nrhs + c] /= div;

        for (int r = 0; r < M; r++) {
            if (r == col) continue;
            double factor = A[r * M + col];
            if (factor == 0.0) continue;
            for (int c = col; c < M; c++)
                A[r * M + c] -= factor * A[col * M + c];
            for (int c = 0; c < nrhs; c++)
                B[r * nrhs + c] -= factor * B[col * nrhs + c];
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

tps_model * tps_solve(const float * src_x, const float * src_y,
                      const float * dst_x, const float * dst_y,
                      int n) {
    if (n < 3 || !src_x || !src_y || !dst_x || !dst_y) return nullptr;

    const int M = n + 3; // system size

    // Build the (N+3) x (N+3) matrix:
    //   [ K  P ]
    //   [ P' 0 ]
    // K[i,j] = U(|src_i - src_j|)
    // P[i,:] = [1, src_x[i], src_y[i]]
    std::vector<double> A(M * M, 0.0);

    // Fill K block (upper-left NxN)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dx = src_x[i] - src_x[j];
            float dy = src_y[i] - src_y[j];
            float r = std::sqrt(dx * dx + dy * dy);
            double u = (double)tps_basis(r);
            A[i * M + j] = u;
            A[j * M + i] = u;
        }
        // K[i,i] = 0 (already initialized)
    }

    // Fill P block (upper-right Nx3) and P' block (lower-left 3xN)
    for (int i = 0; i < n; i++) {
        A[i * M + n + 0] = 1.0;
        A[i * M + n + 1] = (double)src_x[i];
        A[i * M + n + 2] = (double)src_y[i];

        A[(n + 0) * M + i] = 1.0;
        A[(n + 1) * M + i] = (double)src_x[i];
        A[(n + 2) * M + i] = (double)src_y[i];
    }
    // Lower-right 3x3 block is zero (already initialized)

    // Right-hand side: [dst_x; 0; 0; 0] and [dst_y; 0; 0; 0]
    std::vector<double> B(M * 2, 0.0);
    for (int i = 0; i < n; i++) {
        B[i * 2 + 0] = (double)dst_x[i];
        B[i * 2 + 1] = (double)dst_y[i];
    }
    // Last 3 rows of B are zero (already initialized)

    if (!solve_linear(A, B, M, 2)) {
        return nullptr;
    }

    // Extract solution
    tps_model * model = new tps_model;
    model->n = n;
    model->cx.assign(src_x, src_x + n);
    model->cy.assign(src_y, src_y + n);
    model->wx.resize(n);
    model->wy.resize(n);

    for (int i = 0; i < n; i++) {
        model->wx[i] = (float)B[i * 2 + 0];
        model->wy[i] = (float)B[i * 2 + 1];
    }
    model->ax[0] = (float)B[n * 2 + 0]; // constant
    model->ax[1] = (float)B[(n + 1) * 2 + 0]; // x coefficient
    model->ax[2] = (float)B[(n + 2) * 2 + 0]; // y coefficient

    model->ay[0] = (float)B[n * 2 + 1];
    model->ay[1] = (float)B[(n + 1) * 2 + 1];
    model->ay[2] = (float)B[(n + 2) * 2 + 1];

    return model;
}

void tps_map_point(const tps_model * model,
                   float x, float y,
                   float * out_x, float * out_y) {
    if (!model) return;

    float fx = model->ax[0] + model->ax[1] * x + model->ax[2] * y;
    float fy = model->ay[0] + model->ay[1] * x + model->ay[2] * y;

    for (int i = 0; i < model->n; i++) {
        float dx = x - model->cx[i];
        float dy = y - model->cy[i];
        float r = std::sqrt(dx * dx + dy * dy);
        float u = tps_basis(r);
        fx += model->wx[i] * u;
        fy += model->wy[i] * u;
    }

    if (out_x) *out_x = fx;
    if (out_y) *out_y = fy;
}

// Grid step for coarse displacement field. TPS is smooth so 8-pixel bilinear
// interpolation introduces < 0.5px error in practice.
static constexpr int TPS_GRID_STEP = 8;

void tps_warp(const uint8_t * src, int src_w, int src_h,
              const tps_model * model,
              uint8_t * dst, int dst_w, int dst_h,
              uint8_t bg) {
    if (!src || !model || !dst) return;

    // Pre-compute displacement on a coarse grid to avoid O(W*H*N) TPS evaluations.
    // Grid covers [0..dst_w] × [0..dst_h] with step TPS_GRID_STEP.
    int gw = (dst_w  + TPS_GRID_STEP - 1) / TPS_GRID_STEP + 1;
    int gh = (dst_h  + TPS_GRID_STEP - 1) / TPS_GRID_STEP + 1;
    std::vector<float> gdx(gw * gh), gdy(gw * gh);
    for (int gy = 0; gy < gh; gy++) {
        float wy = (float)(gy * TPS_GRID_STEP);
        for (int gx = 0; gx < gw; gx++) {
            float wx = (float)(gx * TPS_GRID_STEP);
            float sx, sy;
            tps_map_point(model, wx, wy, &sx, &sy);
            gdx[gy * gw + gx] = sx;
            gdy[gy * gw + gx] = sy;
        }
    }

    // Render: bilinearly interpolate the coarse grid for each output pixel.
    for (int yo = 0; yo < dst_h; yo++) {
        float gy_f = (float)yo / TPS_GRID_STEP;
        int   gy0  = (int)gy_f;
        int   gy1  = std::min(gy0 + 1, gh - 1);
        float gy_t = gy_f - gy0;

        for (int xo = 0; xo < dst_w; xo++) {
            float gx_f = (float)xo / TPS_GRID_STEP;
            int   gx0  = (int)gx_f;
            int   gx1  = std::min(gx0 + 1, gw - 1);
            float gx_t = gx_f - gx0;

            float sx = (1.0f - gy_t) * ((1.0f - gx_t) * gdx[gy0*gw + gx0] + gx_t * gdx[gy0*gw + gx1])
                     +         gy_t  * ((1.0f - gx_t) * gdx[gy1*gw + gx0] + gx_t * gdx[gy1*gw + gx1]);
            float sy = (1.0f - gy_t) * ((1.0f - gx_t) * gdy[gy0*gw + gx0] + gx_t * gdy[gy0*gw + gx1])
                     +         gy_t  * ((1.0f - gx_t) * gdy[gy1*gw + gx0] + gx_t * gdy[gy1*gw + gx1]);

            if (sx < -0.5f || sy < -0.5f ||
                sx > src_w - 0.5f || sy > src_h - 0.5f) {
                dst[yo * dst_w + xo] = bg;
                continue;
            }

            int x0 = std::max(0, std::min((int)std::floor(sx), src_w - 1));
            int x1 = std::max(0, std::min(x0 + 1,             src_w - 1));
            int y0 = std::max(0, std::min((int)std::floor(sy), src_h - 1));
            int y1 = std::max(0, std::min(y0 + 1,             src_h - 1));
            float fx = sx - std::floor(sx);
            float fy = sy - std::floor(sy);

            float v = (1.0f - fy) * ((1.0f - fx) * src[y0*src_w + x0] + fx * src[y0*src_w + x1])
                    +         fy  * ((1.0f - fx) * src[y1*src_w + x0] + fx * src[y1*src_w + x1]);
            dst[yo * dst_w + xo] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
        }
    }
}

int tps_warp_points(const uint8_t * src, int src_w, int src_h,
                    const float * src_x, const float * src_y,
                    const float * dst_x, const float * dst_y,
                    int n_points,
                    uint8_t * dst, int dst_w, int dst_h,
                    uint8_t bg) {
    tps_model * model = tps_solve(src_x, src_y, dst_x, dst_y, n_points);
    if (!model) return 1;
    tps_warp(src, src_w, src_h, model, dst, dst_w, dst_h, bg);
    tps_free(model);
    return 0;
}

void tps_free(tps_model * model) {
    delete model;
}
