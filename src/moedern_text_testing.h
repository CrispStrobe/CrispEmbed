#pragma once
#ifndef CRISPEMBED_MOEDERN_COMPOSED_TESTING
#error "MoEdern reference hooks are test-build-only"
#endif
#include "moedern_text.h"
#include <functional>
namespace moedern_text_test {
enum class scalar { f32, i64 };
struct tensor_view {
    const void * data = nullptr;
    scalar type = scalar::f32;
    std::vector<int64_t> shape;
    size_t bytes = 0;
};
using observer = std::function<bool(const std::string &, const tensor_view &)>;
bool encode_observed(moedern_text::context *, const std::string &, moedern_text::role, moedern_text::result &,
                     const observer &);
} // namespace moedern_text_test
