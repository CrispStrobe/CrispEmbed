#pragma once
// Scalar adaptation of SLEEF AVX2 xexp/xtanh and small-range sin/cos arithmetic.
// Source: shibatch/sleef 5a1d179df9cf652951b59010a2d2075372d67f68,
// src/common/df.h and src/libm/sleefsimdsp.c.
// Copyright Naoki Shibata and contributors 2010 - 2024.
// Distributed under Boost Software License 1.0, core/third_party/sleef/LICENSE.txt.
#include <cmath>
namespace qwen35_ocr {
namespace lighton4b_avx2_math {
// Preserve each nonfused source operation even under a native compiler target.
inline float add(float a, float b) {
    volatile float x = a + b;
    return x;
}
inline float sub(float a, float b) {
    volatile float x = a - b;
    return x;
}
inline float mul(float a, float b) {
    volatile float x = a * b;
    return x;
}
struct df {
    float x, y;
};
inline df add2(df x, float y) {
    float s = add(x.x, y), v = sub(s, x.x), t = add(sub(x.x, sub(s, v)), sub(y, v));
    return { s, add(t, x.y) };
}
inline df add2(df x, df y) {
    float s = add(x.x, y.x), v = sub(s, x.x), t = add(sub(x.x, sub(s, v)), sub(y.x, v));
    return { s, add(t, add(x.y, y.y)) };
}
inline df add_ordered(float x, df y) {
    float s = add(x, y.x);
    return { s, add(add(sub(x, s), y.x), y.y) };
}
inline df add_ordered(df x, df y) {
    float s = add(x.x, y.x);
    return { s, add(add(add(sub(x.x, s), y.x), x.y), y.y) };
}
inline df multiply(df x, float y) {
    float s = mul(x.x, y);
    return { s, std::fma(x.y, y, std::fma(x.x, y, -s)) };
}
inline df multiply(df x, df y) {
    float s = mul(x.x, y.x);
    return { s, std::fma(x.x, y.y, std::fma(x.y, y.x, std::fma(x.x, y.x, -s))) };
}
inline df square(df x) {
    float s = mul(x.x, x.x);
    return { s, std::fma(add(x.x, x.x), x.y, std::fma(x.x, x.x, -s)) };
}
inline df reciprocal(df d) {
    float s = 1.0f / d.x;
    return { s, mul(s, std::fma(-d.y, s, std::fma(-d.x, s, 1.0f))) };
}
inline df divide(df n, df d) {
    float t = 1.0f / d.x, s = mul(n.x, t), u = std::fma(t, n.x, -s), v = std::fma(-d.y, t, std::fma(-d.x, t, 1.0f));
    return { s, std::fma(s, v, std::fma(n.y, t, u)) };
}
inline float scale2(float x, int q) {
    int half = q >= 0 ? q / 2 : -((-q + 1) / 2);
    return mul(mul(x, std::ldexp(1.0f, half)), std::ldexp(1.0f, q - half));
}
inline df exp_pair(float x) {
    int q = int(std::nearbyint(mul(x, 1.442695040888963407359924681001892137426645954152985934135449406931f)));
    df s = add2(df{ x, 0 }, mul(float(q), -0.693145751953125f));
    s = add2(s, mul(float(q), -1.428606765330187045e-06f));
    float u = 0.1980960224e-3f;
    u = std::fma(u, s.x, 0.1394256484e-2f);
    u = std::fma(u, s.x, 0.8333456703e-2f);
    u = std::fma(u, s.x, 0.4166637361e-1f);
    df t = add2(multiply(s, u), 0.166666659414234244790680580464f);
    t = add2(multiply(s, t), 0.5f);
    t = add2(s, multiply(square(s), t));
    t = add_ordered(1.0f, t);
    return { scale2(t.x, q), scale2(t.y, q) };
}
inline float tanh(float x) {
    if (std::abs(x) > 8.664339742f) return std::copysign(1.0f, x);
    df d = exp_pair(std::abs(x)), e = reciprocal(d);
    d = divide(add_ordered(d, df{ -e.x, -e.y }), add_ordered(d, e));
    return std::copysign(add(d.x, d.y), x);
}
inline float gelu(float x) {
    const float cube = mul(mul(x, x), x);
    const float inner = mul(0.7978845608028654f, add(x, mul(0.044715f, cube)));
    return mul(mul(0.5f, x), add(1.0f, tanh(inner)));
}
// Valid only for finite |angle| < 125, original AVX2 small-range branch.
inline float trig(float angle, bool cosine) {
    float scaled = mul(angle, 0.31830988618379067154f);
    int q = int(std::nearbyint(cosine ? sub(scaled, 0.5f) : scaled));
    if (cosine) q = q * 2 + 1;
    float d = std::fma(float(q), cosine ? -3.1414794921875f * 0.5f : -3.1414794921875f, angle);
    d = std::fma(float(q), cosine ? -0.00011315941810607910156f * 0.5f : -0.00011315941810607910156f, d);
    d = std::fma(float(q), cosine ? -1.9841872589410058936e-09f * 0.5f : -1.9841872589410058936e-09f, d);
    float s = mul(d, d);
    if (cosine ? !(q & 2) : (q & 1)) d = -d;
    float u = 2.6083159809786593541503e-06f;
    u = std::fma(u, s, -0.0001981069071916863322258f);
    u = std::fma(u, s, 0.00833307858556509017944336f);
    u = std::fma(u, s, -0.166666597127914428710938f);
    u = add(mul(s, mul(u, d)), d);
    return !cosine && angle == 0 ? angle : u;
}
} // namespace lighton4b_avx2_math
} // namespace qwen35_ocr
