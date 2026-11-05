// Test-only same-input production expression observer, not a production target.
#include "lighton4b_vision_bf16.cpp"
#include <fstream>
#include <stdexcept>
using namespace qwen35_ocr;
#include "vision_boundary_expressions.inc"
int main(int argc, char ** argv) {
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
    emit("native.cos", output.rotary_cos);
    emit("native.sin", output.rotary_sin);
    auto qkv = read("source.qkv", tokens * 3072);
    emit("native.rotary_pooled", attention(qkv, c, tokens, offsets, output.rotary_cos, output.rotary_sin));
    emit("source.rotary_pooled",
         attention(qkv, c, tokens, offsets, read("source.cos", tokens * d), read("source.sin", tokens * d)));
    return 0;
}
