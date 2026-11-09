// Installed original PyTorch header control, test-only.
#include <cmath>
#include <ATen/native/cpu/Gelu.h>
#include <fstream>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    using B = c10::BFloat16;
    using V = at::vec::CPU_CAPABILITY::Vectorized<B>;
    std::vector<float> x(560 * 4096), y(x.size());
    std::ifstream f(argv[1], std::ios::binary);
    f.read(reinterpret_cast<char *>(x.data()), x.size() * 4);
    if (!f || f.peek() != EOF) return 3;
    std::vector<B> b(x.begin(), x.end()), o(b.size());
    for (size_t i = 0; i < b.size(); i += V::size()) {
        auto z = at::native::CPU_CAPABILITY::vectorized_gelu_approximated_with_tanh<B>(V::loadu(b.data() + i));
        z.store(o.data() + i);
    }
    for (size_t i = 0; i < y.size(); ++i) y[i] = float(o[i]);
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char *>(y.data()), y.size() * 4);
    return out ? 0 : 4;
}
