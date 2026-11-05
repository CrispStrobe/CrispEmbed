// Test-only native attention observer and source-fed primitive alternatives.
#include "lighton4b_vision_bf16.cpp"
#include <fstream>
#include <stdexcept>
using namespace qwen35_ocr;
constexpr int N = 560, W = 1024, H = 16, D = 64;
static float reduce8(const std::array<float, 8> & s) {
    return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string dir = argv[1];
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f || f.peek() != EOF) throw std::runtime_error("input differs");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!f) throw std::runtime_error("output failed");
    };
    auto qkv = read("source.qkv", N * W * 3), co = read("source.cos", N * D), si = read("source.sin", N * D);
    vision_config c;
    c.width = W;
    c.heads = H;
    auto gate = [](const char * value) {
#ifdef _WIN32
        _putenv_s("CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION", value ? value : "");
#else
        if (value)
            setenv("CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION", value, 1);
        else
            unsetenv("CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION");
#endif
    };
    gate(nullptr);
    emit("actual.pooled", attention(qkv, c, N, { 0, N }, co, si));
    for (const char * value : { "", "0" }) {
        gate(value);
        emit(std::string("disabled.") + (value[0] ? "zero" : "empty"), attention(qkv, c, N, { 0, N }, co, si));
    }
    gate("1");
    emit("actual.corrected_pooled", attention(qkv, c, N, { 0, N }, co, si));
    gate(nullptr);
    std::vector<float> q(N * W), k(N * W), v(N * W);
    for (int t = 0; t < N; ++t)
        for (int h = 0; h < H; ++h) {
            size_t b = size_t(t) * W + h * D, s = size_t(t) * 3 * W + h * D;
            std::copy_n(qkv.data() + s, D, q.data() + b);
            std::copy_n(qkv.data() + s + W, D, k.data() + b);
            std::copy_n(qkv.data() + s + 2 * W, D, v.data() + b);
            for (int i = 0; i < D / 2; ++i)
                for (auto * x : { &q, &k }) {
                    float left = (*x)[b + i], right = (*x)[b + i + D / 2], a = co[t * D + i], z = si[t * D + i];
                    (*x)[b + i] = lighton4b_bf16(left * a - right * z);
                    (*x)[b + i + D / 2] = lighton4b_bf16(right * a + left * z);
                }
        }
    emit("native.qrot", q);
    emit("native.krot", k);
    auto oq = read("source.qrot", N * W), ok = read("source.krot", N * W);
    auto scores = [&](const std::vector<float> & a, const std::vector<float> & b, int mode) {
        std::vector<float> y(size_t(H) * N * N);
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < N; ++t)
                for (int s = 0; s < N; ++s) {
                    const float *x = a.data() + size_t(t) * W + h * D, *z = b.data() + size_t(s) * W + h * D;
                    float dot;
                    if (mode == 8)
                        dot = source_bf16_dot<true>(x, z, D);
                    else if (mode == 4)
                        dot = source_bf16_conv_dot<true>(x, z, D);
                    else {
                        double exact = 0;
                        for (int i = 0; i < D; ++i) exact += double(x[i]) * double(z[i]);
                        dot = float(exact);
                    }
                    y[(size_t(h) * N + t) * N + s] = lighton4b_bf16(lighton4b_bf16(dot) * 0.125f);
                }
        return y;
    };
    auto ns = scores(q, k, 0);
    emit("native.scores", ns);
    emit("sourcefed.dot8", scores(oq, ok, 8));
    emit("sourcefed.dot4", scores(oq, ok, 4));
    emit("sourcefed.dotdouble", scores(oq, ok, 0));
    auto os = read("source.scores", size_t(H) * N * N), oe = read("source.exp", size_t(H) * N * N);
    auto softmax = [&](const std::vector<float> & s, bool vecsum, const std::vector<float> * exps) {
        std::vector<float> y(s.size());
        for (int r = 0; r < H * N; ++r) {
            size_t base = size_t(r) * N;
            float maximum = -INFINITY;
            for (int j = 0; j < N; ++j) maximum = std::max(maximum, s[base + j]);
            std::array<float, 8> sum{};
            double exact = 0;
            for (int j = 0; j < N; ++j) {
                float e = exps ? (*exps)[base + j] : std::exp(s[base + j] - maximum);
                y[base + j] = e;
                exact += double(e);
                sum[j % 8] += e;
            }
            float inverse = 1.0f / (vecsum ? reduce8(sum) : float(exact));
            for (int j = 0; j < N; ++j) y[base + j] = lighton4b_bf16(y[base + j] * inverse);
        }
        return y;
    };
    auto np = softmax(ns, false, nullptr);
    emit("native.prob", np);
    emit("sourcefed.softmax_std_double", softmax(os, false, nullptr));
    emit("sourcefed.softmax_std_vec8", softmax(os, true, nullptr));
    emit("sourcefed.softmax_sourceexp_vec8", softmax(os, true, &oe));
    auto pool = [&](const std::vector<float> & p, int mode) {
        std::vector<float> y(N * W);
        std::array<float, N> g{};
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < D; ++i) {
                for (int s = 0; s < N; ++s) g[s] = v[size_t(s) * W + h * D + i];
                for (int t = 0; t < N; ++t) {
                    const float * a = p.data() + (size_t(h) * N + t) * N;
                    float dot;
                    if (mode == 4) {
                        std::array<float, 4> sum{};
                        for (int s = 0; s < N; s += 4)
                            for (int j = 0; j < 4; ++j) sum[j] = std::fma(a[s + j], g[s + j], sum[j]);
                        dot = ((sum[0] + sum[1]) + sum[2]) + sum[3];
                    } else if (mode == 8) {
                        dot = source_bf16_dot<false>(a, g.data(), 512);
                        std::array<float, 8> tail{};
                        for (int s = 512; s < N; s += 16)
                            for (int j = 0; j < 8; ++j) {
                                tail[j] = std::fma(a[s + j], g[s + j], tail[j]);
                                tail[j] = std::fma(a[s + j + 8], g[s + j + 8], tail[j]);
                            }
                        dot += reduce8(tail);
                    } else {
                        double exact = 0;
                        for (int s = 0; s < N; ++s) exact += double(a[s]) * double(g[s]);
                        dot = float(exact);
                    }
                    y[size_t(t) * W + h * D + i] = lighton4b_bf16(dot);
                }
            }
        return y;
    };
    emit("trace.pooled", pool(np, 0));
    emit("trace.corrected_pooled", pool(softmax(os, false, nullptr), 4));
    std::vector<float> std_exp(os.size());
    for (int r = 0; r < H * N; ++r) {
        size_t base = size_t(r) * N;
        float maximum = -INFINITY;
        for (int j = 0; j < N; ++j) maximum = std::max(maximum, os[base + j]);
        for (int j = 0; j < N; ++j) std_exp[base + j] = std::exp(os[base + j] - maximum);
    }
    emit("sourcefed.std_exp", std_exp);
    auto op = read("source.prob", size_t(H) * N * N);
    emit("sourcefed.pv4", pool(op, 4));
    emit("sourcefed.pv8", pool(op, 8));
    emit("sourcefed.pvdouble", pool(op, 0));
    return 0;
}
