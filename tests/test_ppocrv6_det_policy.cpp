#include "core/clean_exit.h"
#include "ppocrv6_det_policy.h"

#include <cstdio>
#include <limits>
#include <vector>

static int run() {
    using namespace ppocrv6_det::policy;
    int failures = 0, checks = 0;
    auto check = [&](bool ok, const char * message) {
        ++checks;
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", message);
        }
    };
    check(graph_shape(false, false, 0, 0, 32, 64, false) == shape_action::build, "first shape builds");
    check(graph_shape(true, true, 32, 64, 32, 64, false) == shape_action::reuse, "same shape reuses");
    check(graph_shape(true, true, 32, 64, 64, 32, false) == shape_action::fallback, "different shape defaults off");
    check(graph_shape(true, true, 32, 64, 64, 32, true) == shape_action::rebuild, "different shape rebuilds opt-in");
    check(graph_shape(true, false, 32, 64, 32, 64, true) == shape_action::fallback, "failed shape is not retried");
    check(graph_shape(true, false, 32, 64, 64, 32, true) == shape_action::rebuild, "new shape retries after failure");
    const std::vector<float> valid = { 0.0f, 0.2f, 0.6f, 1.0f };
    check(!accept_graph(false, false, 0, valid, 2, 2), "empty graph falls back by default");
    check(accept_graph(false, true, 0, valid, 2, 2), "finite empty result accepted opt-in");
    check(!accept_graph(true, true, 0, valid, 2, 2), "compare mode always runs reference");
    check(!accept_graph(true, true, 1, valid, 2, 2), "compare mode never accepts nonempty graph");
    check(accept_graph(false, false, 1, valid, 2, 2), "existing nonempty graph accepted");
    check(!accept_graph(false, true, 0, {}, 2, 2), "missing probability map rejected");
    check(!accept_graph(false, true, 0, valid, 0, 2), "invalid dimensions rejected");
    check(!accept_graph(false, true, 0, valid, 1, 2), "mismatched map size rejected");
    auto invalid = valid;
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    check(!accept_graph(false, true, 0, invalid, 2, 2), "NaN rejected");
    invalid[0] = std::numeric_limits<float>::infinity();
    check(!accept_graph(false, true, 0, invalid, 2, 2), "infinity rejected");
    invalid[0] = -0.01f;
    check(!accept_graph(false, true, 0, invalid, 2, 2), "negative probability rejected");
    invalid[0] = 1.01f;
    check(!accept_graph(false, true, 0, invalid, 2, 2), "probability above one rejected");
    std::printf("ppocrv6 detector policy: %d/%d passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(run());
}
