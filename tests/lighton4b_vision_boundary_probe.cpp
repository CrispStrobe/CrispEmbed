// Test-only same-input production expression observer, not a production target.
#include "lighton4b_vision_bf16.cpp"
#include "core/clean_exit.h"
#include <fstream>
#include <stdexcept>
using namespace qwen35_ocr;
#include "vision_boundary_expressions.inc"
static float trig_nonfused(float angle, bool cosine) {
    using namespace lighton4b_avx2_math;
    float scaled = mul(angle, 0.31830988618379067154f);
    int q = int(std::nearbyint(cosine ? sub(scaled, 0.5f) : scaled));
    if (cosine) q = q * 2 + 1;
    const float factor = cosine ? 0.5f : 1.0f;
    float d = add(mul(float(q), mul(-3.1414794921875f, factor)), angle);
    d = add(mul(float(q), mul(-0.00011315941810607910156f, factor)), d);
    d = add(mul(float(q), mul(-1.9841872589410058936e-09f, factor)), d);
    float square = mul(d, d);
    if (cosine ? !(q & 2) : (q & 1)) d = -d;
    float u = 2.6083159809786593541503e-06f;
    u = add(mul(u, square), -0.0001981069071916863322258f);
    u = add(mul(u, square), 0.00833307858556509017944336f);
    u = add(mul(u, square), -0.166666597127914428710938f);
    u = add(mul(square, mul(u, d)), d);
    return !cosine && angle == 0 ? angle : u;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const std::string dir = argv[1], mode = argv[2];
    auto read = [&](const std::string & name, size_t count) {
        std::vector<float> x(count);
        std::ifstream in(dir + "/" + name + ".f32", std::ios::binary);
        in.read(reinterpret_cast<char *>(x.data()), count * 4);
        if (!in || in.peek() != EOF) throw std::runtime_error("source length differs");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream out(dir + "/" + name + ".f32", std::ios::binary);
        out.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!out) throw std::runtime_error("output failed");
    };
    if (mode == "activation") {
        auto projected = read("activation.input", 560 * 4096);
        production_activation(projected);
        round_bf16(projected);
        emit("activation.output", projected);
        return 0;
    }
    if (mode != "rotary") return 2;
    vision_config c;
    c.width = 1024;
    c.heads = 16;
    c.theta = 10000;
    const int tokens = 560, d = 64;
    std::vector<int> rows, columns, offsets, heights, widths;
    if (!geometry(c, tokens, { 1, 20, 28 }, rows, columns, offsets, heights, widths)) return 3;
    vision_outputs output;
    output.rotary_cos.resize(tokens * d);
    output.rotary_sin.resize(tokens * d);
    production_rotary(c, tokens, d, rows, columns, output);
    std::vector<float> inverse(16), trig_cos(125 * 16), trig_sin(trig_cos.size()), nonfused_cos(trig_cos.size()),
        nonfused_sin(trig_cos.size());
    for (int f = 0; f < 16; ++f) {
        inverse[f] = 1.0f / std::pow(10000.0f, float(2 * f) / 32.0f);
        for (int coordinate = 0; coordinate < 125; ++coordinate) {
            float angle = inverse[f] * float(coordinate);
            trig_cos[coordinate * 16 + f] = lighton4b_avx2_math::trig(angle, true);
            trig_sin[coordinate * 16 + f] = lighton4b_avx2_math::trig(angle, false);
            nonfused_cos[coordinate * 16 + f] = trig_nonfused(angle, true);
            nonfused_sin[coordinate * 16 + f] = trig_nonfused(angle, false);
        }
    }
    emit("native.inverse_frequency", inverse);
    emit("helper.cos_grid", trig_cos);
    emit("helper.sin_grid", trig_sin);
    emit("nonfused.cos_grid", nonfused_cos);
    emit("nonfused.sin_grid", nonfused_sin);
    emit("native.cos", output.rotary_cos);
    emit("native.sin", output.rotary_sin);
    auto qkv = read("source.qkv", tokens * 3072);
    emit("native.rotary_pooled", attention(qkv, c, tokens, offsets, output.rotary_cos, output.rotary_sin));
    emit("source.rotary_pooled",
         attention(qkv, c, tokens, offsets, read("source.cos", tokens * d), read("source.sin", tokens * d)));
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
