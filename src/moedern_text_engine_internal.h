#pragma once
#include "moedern_text.h"
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
#include "moedern_text_testing.h"
#endif
namespace moedern_text_detail {
using role = moedern_text::role;
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
using scalar = moedern_text_test::scalar;
using tensor_view = moedern_text_test::tensor_view;
using observer = moedern_text_test::observer;
#endif
struct result {
    std::vector<float> vectors;
    std::vector<uint8_t> supplied_mask, effective_mask;
};
struct context;
context * init(const char *, int, std::string &);
void free(context *);
const char * error(const context *);
bool encode_tokens(context *, const int32_t *, const uint8_t *, size_t, role, result &
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
                   ,
                   const observer & = {}
#endif
);
bool scoring_mask(const std::vector<int32_t> &, const std::vector<uint8_t> &, role, const std::vector<int32_t> &,
                  std::vector<uint8_t> &, std::string &);
bool select_rows(const std::vector<float> &, const std::vector<uint8_t> &, std::vector<float> &);
bool maxsim(const std::vector<float> &, const std::vector<float> &, float &);
} // namespace moedern_text_detail
