// Test-only exact installed PyTorch vector-header softmax control.
#include <ATen/cpu/vec/functional.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
namespace vx = at::vec::CPU_CAPABILITY;
int main(int argc, char ** argv) {
    using V = vx::Vectorized<float>;
    if (argc != 3) return 2;
    const std::string dir = argv[1], name = argv[2];
    constexpr int rows = 8960, width = 560;
    std::vector<float> x(rows * width), exps(x.size()), prob(x.size()), sum(rows), inverse(rows);
    std::ifstream f(dir + "/source.scores.f32", std::ios::binary);
    f.read(reinterpret_cast<char *>(x.data()), x.size() * 4);
    if (!f || f.peek() != EOF) return 3;
    std::vector<float> zero(V::size());
    V(0).exp().store(zero.data());
    bool exp0 = true;
    for (float z : zero) exp0 = exp0 && z == 1.0f;
    for (int r = 0; r < rows; ++r) {
        const float * input = x.data() + r * width;
        float * output = exps.data() + r * width;
        float maximum = at::vec::reduce_all<float>([](V a, V b) { return vx::maximum(a, b); }, input, width);
        at::vec::map<float>([maximum](V a) { return (a - V(maximum)).exp(); }, output, input, width);
        sum[r] = at::vec::reduce_all<float>([](V a, V b) { return a + b; }, output, width);
        inverse[r] = 1.0f / sum[r];
        at::vec::map<float>([factor = inverse[r]](V a) { return a * V(factor); }, prob.data() + r * width, output,
                            width);
    }
    auto emit = [&](const std::string & suffix, const std::vector<float> & y) {
        std::ofstream out(dir + "/" + name + "." + suffix + ".f32", std::ios::binary);
        out.write(reinterpret_cast<const char *>(y.data()), y.size() * 4);
        if (!out) throw std::runtime_error("write failed");
    };
    emit("exp", exps);
    emit("sum", sum);
    emit("inverse", inverse);
    emit("prob", prob);
    std::ofstream report(dir + "/" + name + ".json");
    report << "{\"lanes\":" << V::size() << ",\"exp_zero_exact_one\":" << (exp0 ? "true" : "false") << "}\n";
    return exp0 ? 0 : 4;
}
