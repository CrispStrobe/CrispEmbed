// Test-only stage comparison: no production library or model probes.
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include <cstdio>
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    crispembed_diff::Ref native, wasm;
    if (!native.load(argv[1]) || !wasm.load(argv[2])) return 2;
    const char * names[] = { "emb_ln_out", "attn_out_0", "layer_0", "layer_1",
                             "layer_2",    "layer_3",    "layer_4", "encoder_out" };
    for (const char * name : names) {
        auto values = wasm.get_f32(name);
        if (!values.first || !values.second) return 3;
        auto r = native.compare(name, values.first, values.second);
        if (!r.found) return 3;
        printf("{\"stage\":\"%s\",\"n\":%zu,\"max_abs\":%.12g,\"mean_abs\":%.12g,\"rms\":%.12g,\"cos_min\":%.12g,\"cos_"
               "global\":%.12g,\"native_l2\":%.12g,\"wasm_l2\":%.12g}\n",
               name, r.n_elem, r.max_abs, r.mean_abs, r.rms, r.cos_min, r.cos_global, r.ref_norm, r.mine_norm);
    }
    return 0; // Attribution records differences; unchanged final gates decide qualification elsewhere.
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
