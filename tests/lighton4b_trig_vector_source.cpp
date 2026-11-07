// Test-only installed vector-header trig/dispatch control.
#include <ATen/cpu/vec/functional.h>
#include <fstream>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    using V = at::vec::CPU_CAPABILITY::Vectorized<float>;
    std::vector<float> x(2000), y(x.size()), z(x.size());
    std::ifstream in(argv[1], std::ios::binary);
    in.read(reinterpret_cast<char *>(x.data()), x.size() * 4);
    if (!in || in.peek() != EOF) return 3;
    for (size_t i = 0; i < x.size(); i += V::size()) {
        V v = V::loadu(x.data() + i);
        v.cos().store(y.data() + i);
        v.sin().store(z.data() + i);
    }
    std::ofstream co(argv[2], std::ios::binary), si(argv[3], std::ios::binary);
    co.write(reinterpret_cast<const char *>(y.data()), y.size() * 4);
    si.write(reinterpret_cast<const char *>(z.data()), z.size() * 4);
    return co && si ? 0 : 4;
}
