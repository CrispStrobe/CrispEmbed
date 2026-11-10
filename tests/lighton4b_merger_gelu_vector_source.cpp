// Installed original Gelu.h default-erf control, test-only.
#include <cmath>
#include "core/clean_exit.h"
#include <ATen/native/cpu/Gelu.h>
#include <fstream>
#include <vector>
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    using B = c10::BFloat16;
    using V = at::vec::CPU_CAPABILITY::Vectorized<B>;
    std::vector<float> x(140 * 4096), y(x.size());
    std::ifstream in(argv[1], std::ios::binary);
    in.read(reinterpret_cast<char *>(x.data()), x.size() * 4);
    if (!in || in.peek() != EOF) return 3;
    std::vector<B> b(x.begin(), x.end()), o(b.size());
    for (size_t i = 0; i < b.size(); i += V::size())
        at::native::CPU_CAPABILITY::vectorized_gelu<B>(V::loadu(b.data() + i)).store(o.data() + i);
    for (size_t i = 0; i < y.size(); ++i) y[i] = float(o[i]);
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<char *>(y.data()), y.size() * 4);
    return out ? 0 : 4;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
