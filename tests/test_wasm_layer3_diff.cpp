// Test-only comparison of actual named subnodes; never linked into production.
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include <cstdio>
static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 4) return 2;
    crispembed_diff::Ref native, wasm;
    if (!native.load(argv[1]) || !wasm.load(argv[2])) return 2;
    for (int i = 3; i < argc; ++i) {
        auto values = wasm.get_f32(argv[i]);
        if (!values.first || !values.second) return 3;
        auto r = native.compare(argv[i], values.first, values.second);
        if (!r.found) return 3;
        printf("{\"stage\":\"%s\",\"n\":%zu,\"max_abs\":%.12g,\"rms\":%.12g,\"cos_min\":%.12g,\"cos_global\":%.12g}\n",
               argv[i], r.n_elem, r.max_abs, r.rms, r.cos_min, r.cos_global);
    }
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
