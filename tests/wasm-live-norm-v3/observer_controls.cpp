// Weight-free v3 observer controls: actual ggml graph metadata, no compute.
// Reproduces the 41418312256 native abort (unevaluated reservation build at
// bucket 16 rows reaching the observer) and proves the v3 bracket admits only
// the evaluated 12-row graph with the unchanged v1 rules.
#include "observer.h"
#include <cstdio>

static int failures = 0;
static void expect(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main(int argc, char ** argv) {
    const char * dir = argc > 1 ? argv[1] : ".";
    ggml_init_params params = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * g = ggml_init(params);
    auto norm_input = [&](int rows) {
        ggml_tensor * a = ggml_new_tensor_2d(g, GGML_TYPE_F32, 384, rows);
        ggml_tensor * b = ggml_new_tensor_2d(g, GGML_TYPE_F32, 384, rows);
        return ggml_add(g, a, b);  // actual first native norm input op: ADD
    };
    auto fresh = [&](live_norm_observer & o) {
        o = live_norm_observer();
        o.configure("bert", 384, 6, 12, 1536, false, 1e-12f, true, false);
        o.directory = dir;  // active(); begin() needs a scheduler and is not called here
    };
    live_norm_observer o;

    // 1. v1 behaviour on the reservation graph: rejected (the v2 abort).
    fresh(o);
    ggml_tensor * x16 = norm_input(16);
    expect(!o.add_norm(x16, ggml_norm(g, x16, 1e-12f)), "unbracketed bucket-16 reservation norm is rejected (reproduces 41418312256)");

    // 2. v3 bracket: reservation build neither registers nor resets state.
    fresh(o);
    ggml_tensor * x12 = norm_input(12);
    expect(o.add_norm(x12, ggml_norm(g, x12, 1e-12f)) && o.norms == 1 && o.entries.size() == 2, "evaluated 12-row norm registers");
    o.measuring = true;
    o.begin(nullptr);
    expect(o.measured_builds == 1 && o.norms == 1 && o.entries.size() == 2, "measuring begin() does not reset evaluated state");
    ggml_tensor * y16 = norm_input(16);
    expect(o.add_norm(y16, ggml_norm(g, y16, 1e-12f)) && o.norms == 1 && o.entries.size() == 2, "measuring add_norm is a no-op");
    expect(o.add("layer3_input", y16) && o.entries.size() == 2, "measuring add is a no-op");
    o.measuring = false;

    // 3. Unchanged v1 rules for evaluated graphs.
    ggml_tensor * z16 = norm_input(16);
    expect(!o.add_norm(z16, ggml_norm(g, z16, 1e-12f)), "evaluated 16-row norm still rejected");
    ggml_tensor * z13 = norm_input(13);
    expect(!o.add_norm(z13, ggml_norm(g, z13, 1e-12f)), "evaluated 13-row norm still rejected");
    ggml_tensor * w = norm_input(12);
    expect(!o.add_norm(w, ggml_add(g, w, w)), "wrong output op still rejected");
    ggml_tensor * v = norm_input(12);
    expect(!o.add_norm(norm_input(12), ggml_norm(g, v, 1e-12f)), "disconnected src[0] still rejected");
    expect(!o.add("norm00_input", x12), "duplicate name still rejected");
    fresh(o);
    o.norms = 13;
    ggml_tensor * u = norm_input(12);
    expect(!o.add_norm(u, ggml_norm(g, u, 1e-12f)), "fourteenth norm still rejected");

    ggml_free(g);
    std::printf("observer controls: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
