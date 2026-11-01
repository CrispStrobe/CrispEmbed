// Small, weight-free analytical checks for the actual native lambda/GLA
// kernels. Build/run on hosted CI; this does not qualify either full model.
#include "latent_inpaint_unet.h"
#include "core/clean_exit.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using latent_inpaint::Tensor;

namespace {

void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

Tensor tensor(int c, int h, int w, std::initializer_list<float> values) {
    Tensor t(c, h, w);
    t.v.assign(values);
    check(t.v.size() == size_t(c) * h * w, "bad test fixture size");
    return t;
}

void close(const Tensor & result, std::initializer_list<float> expected, float tolerance = 2e-5f) {
    check(result.v.size() == expected.size(), "output size differs");
    size_t i = 0;
    for (float value : expected) {
        if (!std::isfinite(result.v[i]) || std::abs(result.v[i] - value) > tolerance)
            throw std::runtime_error("analytical component mismatch at " + std::to_string(i));
        ++i;
    }
}

void rejects(const std::function<void()> & fn) {
    try {
        fn();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}

void test_timestep() {
    using latent_inpaint::detail::timestep_embedding;
    const auto zero = timestep_embedding(0);
    check(zero.v.size() == 320, "time embedding width");
    for (size_t i = 0; i < 160; ++i) check(zero.v[i] == 1 && zero.v[i + 160] == 0, "cos/sin order");
    // Four dimensions have frequencies 1 and 1/100, not 1 and 1/10000.
    close(timestep_embedding(25, 4), { std::cos(25.0f), std::cos(0.25f), std::sin(25.0f), std::sin(0.25f) });
    rejects([] { timestep_embedding(-1); });
    rejects([] { timestep_embedding(1000); });
    rejects([] { timestep_embedding(1, 3); });
}

void test_gla() {
    using latent_inpaint::detail::gla_recurrent;
    const Tensor q = tensor(4, 1, 3, { 1, 2, 3, 0, 1, -1, 2, 1, 0, 1, -1, 2 });
    const Tensor k = tensor(4, 1, 3, { 1, 0, 2, 1, 2, -1, 0, 1, 2, 2, 1, 1 });
    const Tensor v = tensor(4, 1, 3, { 1, 3, 2, -1, 2, 4, 2, 0, 1, 1, 2, -2 });
    const Tensor logits(4, 1, 3);
    // With zero logits and normalizer 1 each key-state row decays by 1/2.
    // Gold uses the independent closed form:
    // sum_{s<=t} (q_t . k_s)/sqrt(2) * (1/2)^(t-s) * v_s.
    const auto got = gla_recurrent(q, k, v, logits, 2, 1.0f);
    close(got, { 0.7071067812f, 5.303300859f, 8.131727984f, -0.7071067812f, 1.767766953f, 18.03122292f, 2.828427125f,
                 -1.414213562f, 2.828427125f, 1.414213562f, -0.7071067812f, -0.7071067812f });
    check(got.v == gla_recurrent(q, k, v, logits, 2, 1.0f).v, "GLA leaked state across invocations");
    Tensor future = v;
    for (int c = 0; c < 4; ++c) future.v[size_t(c) * 3 + 2] += 100.0f;
    const auto changed = gla_recurrent(q, k, future, logits, 2, 1.0f);
    for (int c = 0; c < 4; ++c)
        for (int t = 0; t < 2; ++t)
            check(got.v[size_t(c) * 3 + t] == changed.v[size_t(c) * 3 + t], "GLA future token leakage");
    Tensor extreme = logits;
    for (size_t i = 0; i < extreme.v.size(); ++i) extreme.v[i] = i % 2 ? -1000.0f : 1000.0f;
    check(latent_inpaint::finite(gla_recurrent(q, k, v, extreme, 2)), "unstable key log-sigmoid");
    rejects([&] { gla_recurrent(q, k, v, logits, 3); });
    rejects([&] { gla_recurrent(q, k, v, logits, 2, 0.0f); });
    Tensor bad = q;
    bad.v[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { gla_recurrent(bad, k, v, logits, 2); });
}

void test_lambda() {
    using latent_inpaint::detail::lambda_content;
    using latent_inpaint::detail::lambda_relative;
    const Tensor q = tensor(4, 1, 2, { 1, 2, -1, 3, 2, -1, 1, 2 });
    const Tensor k = tensor(2, 1, 3, { 0, std::log(2.0f), std::log(3.0f), std::log(3.0f), std::log(2.0f), 0 });
    const Tensor v = tensor(2, 1, 3, { 1, 2, 4, -1, 3, 2 });
    // Softmax is over key positions; queries use eight heads in the model but
    // share one K/V pair. This two-head fixture catches softmax-axis/head bugs.
    close(lambda_content(q, k, v, 2), { 1, 11.166666667f, 1, 6.166666667f, 7.5f, 0.833333333f, 4.5f, -0.166666667f });
    std::vector<float> relative_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    latent_inpaint::Weight relative{ { 2, 3, 2, 1 }, relative_data.data(), relative_data.size() };
    // Gold is direct sum_{k,m} Q[h,k,n] E[n,m,k,0] V[v,m].
    close(lambda_relative(q, v, relative, 2), { -7, 366, -4, 222, 88, 83, 58, 50 });
    rejects([&] { lambda_content(q, k, v, 1); });
    relative.shape[0] = 3;
    rejects([&] { lambda_relative(q, v, relative, 2); });
    relative.shape = { 2, 3, 2, 1 };
    relative.size = 11;
    rejects([&] { lambda_relative(q, v, relative, 2); });
}

} // namespace

static int crispembed_test_main() {
    try {
        test_timestep();
        test_gla();
        test_lambda();
        std::cout << "latent inpaint attention: analytical, causality, reset and rejection checks passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

int main() {
    const int result = crispembed_test_main();
    std::cout.flush();
    std::cerr.flush();
    core_util::clean_exit(result);
}
